import itertools
import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import sounddevice
import soundfile
import torch

from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator
from InferenceInterfaces.InferenceArchitectures.InferenceBigVGAN import BigVGAN
from TrainingInterfaces.Text_to_Spectrogram.UnetTTS.UnetTTS import UnetTTS
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.storage_config import MODELS_DIR
from Utility.utils import float2pcm


class UnetTTSInterface(torch.nn.Module):

    def __init__(self,
                 device="cpu",  # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
                 tts_model_path=os.path.join(MODELS_DIR, "UnetTTS_Meta", "Content_Encoder", "best.pt"),  # path to the ContentPretrainTTS checkpoint or just a shorthand if run standalone
                 embedding_model_path=None,
                 vocoder_model_path=None,  # path to the hifigan/avocodo/bigvgan checkpoint
                 faster_vocoder=True,  # whether to use the quicker HiFiGAN or the better BigVGAN
                 language="qu",  # initial language of the model, can be changed later with the setter methods
                 ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            # default to shorthand system
            content_encoder_path = os.path.join(MODELS_DIR, f"UnetTTS_{tts_model_path}","Content_Encoder", "best.pt")
            tts_model_path = os.path.join(MODELS_DIR, f"UnetTTS_{tts_model_path}", "best.pt")
        if vocoder_model_path is None:
            if faster_vocoder:
                vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
            else:
                vocoder_model_path = os.path.join(MODELS_DIR, "BigVGAN", "best.pt")

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)

        ################################
        #   load weights               #
        ################################
        checkpoint = torch.load(tts_model_path, map_location='cpu')

        ################################
        #   load phone to mel model    #
        ################################
        self.use_lang_id = True
        self.phone2mel = UnetTTS(content_encoder_path, weights=checkpoint["model"])  # multi speaker multi language
       
        with torch.no_grad():
            self.phone2mel.store_inverse_all()  # this also removes weight norm
        self.phone2mel = self.phone2mel.to(torch.device(device))

        #################################
        #  load mel to style models     #
        #################################
        self.style_embedding_function = StyleEmbedding()
        if embedding_model_path is None:
            check_dict = torch.load(os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"), map_location="cpu")
        else:
            check_dict = torch.load(embedding_model_path, map_location="cpu")
        self.style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        self.style_embedding_function.to(self.device)

        ################################
        #  load mel to wave model      #
        ################################
        if faster_vocoder:
            self.mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(torch.device(device))
        else:
            self.mel2wav = BigVGAN(path_to_weights=vocoder_model_path).to(torch.device(device))
        self.mel2wav.remove_weight_norm()

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True, device=self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.style_embedding_function.eval()
        if self.use_lang_id:
            self.lang_id = get_language_id(language)
        else:
            self.lang_id = None
        self.to(torch.device(device))
        self.eval()

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        assert os.path.exists(path_to_reference_audio)
        wave, sr = soundfile.read(path_to_reference_audio)
        if sr != self.audio_preprocessor.sr:
            self.audio_preprocessor = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True, device=self.device)
        spec = self.audio_preprocessor.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        spec_len = torch.LongTensor([len(spec)])
        self.default_utterance_embedding = self.style_embedding_function(spec.unsqueeze(0).to(self.device),
                                                                         spec_len.unsqueeze(0).to(self.device)).squeeze()

    def set_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """
        self.set_phonemizer_language(lang_id=lang_id)
        self.set_accent_language(lang_id=lang_id)

    def set_phonemizer_language(self, lang_id):
        self.text2phone = ArticulatoryCombinedTextFrontend(language=lang_id, add_silence_to_end=True)

    def set_accent_language(self, lang_id):
        if self.use_lang_id:
            self.lang_id = get_language_id(lang_id).to(self.device)
        else:
            self.lang_id = None

    def forward(self,
                text,
                target_speech,
                view=False,
                durations=None,
                input_is_phones=False,
                return_plot_as_filepath=False):
        """
        duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        """
        with torch.inference_mode():
            phones = self.text2phone.string_to_tensor(text, input_phonemes=input_is_phones).to(torch.device(self.device))
            _, mel, durations = self.phone2mel.inference(phones,
                                                        speech=target_speech,
                                                        utterance_embedding=self.default_utterance_embedding,
                                                        gold_durations=durations,
                                                        lang_id=self.lang_id)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)

        if view or return_plot_as_filepath:
            from Utility.utils import cumsum_durations
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(),
                         ax=ax[1],
                         sr=16000,
                         cmap='GnBu',
                         y_axis='mel',
                         x_axis=None,
                         hop_length=256)
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax[1].xaxis.grid(True, which='minor')
            ax[1].set_xticks(label_positions, minor=False)
            if input_is_phones:
                phones = text.replace(" ", "|")
            else:
                phones = self.text2phone.get_phone_string(text, for_plot_labels=True)
            ax[1].set_xticklabels(phones)
            word_boundaries = list()
            for label_index, phone in enumerate(phones):
                if phone == "|":
                    word_boundaries.append(label_positions[label_index])

            try:
                prev_word_boundary = 0
                word_label_positions = list()
                for word_boundary in word_boundaries:
                    word_label_positions.append((word_boundary + prev_word_boundary) / 2)
                    prev_word_boundary = word_boundary
                word_label_positions.append((duration_splits[-1] + prev_word_boundary) / 2)

                secondary_ax = ax[1].secondary_xaxis('bottom')
                secondary_ax.tick_params(axis="x", direction="out", pad=24)
                secondary_ax.set_xticks(word_label_positions, minor=False)
                secondary_ax.set_xticklabels(text.split())
                secondary_ax.tick_params(axis='x', colors='orange')
                secondary_ax.xaxis.label.set_color('orange')
            except ValueError:
                ax[0].set_title(text)
            except IndexError:
                ax[0].set_title(text)

            ax[1].vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
            ax[1].vlines(x=word_boundaries, colors="orange", linestyles="solid", ymin=0.0, ymax=8000, linewidth=1.2)
            plt.subplots_adjust(left=0.05, bottom=0.12, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            if not return_plot_as_filepath:
                plt.show()
            else:
                plt.savefig("tmp.png")
                return wave, "tmp.png"
        return wave

    def read_to_file(self,
                     text_list,
                     file_location,
                     speaker_reference,
                     silent=False,
                     dur_list=None,
                     increased_compatibility_mode=False):
        """
        Args:
            increased_compatibility_mode: Whether to export audio as 16bit integer 48kHz audio for maximum compatibility across systems and devices
            silent: Whether to be verbose about the process
            text_list: A list of strings to be read
            file_location: The path and name of the file it should be saved to
            dur_list: list of duration tensors to be used for the texts
        """
        if not dur_list:
            dur_list = []
        silence = torch.zeros([10600])
        wav = silence.clone()
        
        assert os.path.exists(speaker_reference)
        wave, sr = soundfile.read(speaker_reference)
        if sr != self.audio_preprocessor.sr:
            self.audio_preprocessor = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True, device=self.device)
        target_speech = self.audio_preprocessor.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        
        for (text, durations) in itertools.zip_longest(text_list, dur_list):
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                spoken_sentence = self(text,
                                       target_speech = target_speech.to(self.device),
                                       durations=durations.to(self.device) if durations is not None else None).cpu()
                wav = torch.cat((wav, spoken_sentence, silence), 0)
        if increased_compatibility_mode:
            wav = [val for val in wav.numpy() for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
            soundfile.write(file=file_location, data=float2pcm(wav), samplerate=48000, subtype="PCM_16")
        else:
            soundfile.write(file=file_location, data=wav, samplerate=24000)

    def read_aloud(self,
                   text,
                   speaker_reference,
                   view=False,
                   blocking=False,
                   increased_compatibility_mode=False):
        if text.strip() == "":
            return
        assert os.path.exists(speaker_reference)
        wave, sr = soundfile.read(speaker_reference)
        if sr != self.audio_preprocessor.sr:
            self.audio_preprocessor = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True, device=self.device)
        target_speech = self.audio_preprocessor.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        wav = self(text,
                   target_speech.to(self.device),
                   view).cpu()
        wav = torch.cat((wav, torch.zeros([12000])), 0).numpy()
        if increased_compatibility_mode:
            wav = [val for val in wav for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
            sounddevice.play(float2pcm(wav), samplerate=48000)
        else:
            sounddevice.play(wav, samplerate=24000)
        if blocking:
            sounddevice.wait()

import torch
from torch.nn import Linear

from Layers.Conformer import Conformer
from Layers.ContentEncoder import ContentEncoder
from Layers.PostNet import PostNet
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.ContentPreTrain.ContentPreTrainLoss import ContentPreTrainLoss
from Utility.utils import initialize
from Utility.utils import make_non_pad_mask


class ContentPreTrain(torch.nn.Module):

    def __init__(self,
                 # network structure related
                 output_spectrogram_channels=80,
                 attention_dimension=192,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 init_type="xavier_uniform",
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,

                 # decoder
                 decoder_layers=6,
                 decoder_units=1536,
                 decoder_concat_after=False,
                 conformer_decoder_kernel_size=31,
                 decoder_normalize_before=True,
                 transformer_dec_dropout_rate=0.2,
                 transformer_dec_positional_dropout_rate=0.2,
                 transformer_dec_attn_dropout_rate=0.2,
                 
                 weights = None
                ):
        super().__init__()

        self.output_spectrogram_channels = output_spectrogram_channels

        self.encoder = ContentEncoder()

        self.decoder = Conformer(idim=0,
                                 attention_dim=attention_dimension,
                                 attention_heads=attention_heads,
                                 linear_units=decoder_units,
                                 num_blocks=decoder_layers,
                                 input_layer=None,
                                 dropout_rate=transformer_dec_dropout_rate,
                                 positional_dropout_rate=transformer_dec_positional_dropout_rate,
                                 attention_dropout_rate=transformer_dec_attn_dropout_rate,
                                 normalize_before=decoder_normalize_before,
                                 concat_after=decoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_decoder_kernel_size,
                                 use_output_norm=False)

        self.feat_out = Linear(attention_dimension, output_spectrogram_channels)

        self.conv_postnet = PostNet(idim=0,
                                    odim=output_spectrogram_channels,
                                    n_layers=5,
                                    n_chans=256,
                                    n_filts=5,
                                    use_batch_norm=True,
                                    dropout_rate=0.5)

        self.post_flow = Glow(
            in_channels=output_spectrogram_channels,
            hidden_channels=192,  # post_glow_hidden
            kernel_size=5,  # post_glow_kernel_size
            dilation_rate=1,
            n_blocks=18,  # post_glow_n_blocks (original 12 in paper)
            n_layers=4,  # post_glow_n_block_layers (original 3 in paper)
            n_split=4,
            n_sqz=2,
            text_condition_channels=attention_dimension,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,
            sigmoid_scale=False,
            condition_integration_projection=torch.nn.Conv1d(output_spectrogram_channels + attention_dimension, attention_dimension, 5, padding=2)
        )
        if weights:
            self.load_state_dict(weights)
            self.eval()
        else:
            # initialize parameters
            self._reset_parameters(init_type=init_type)

            self.criterion = ContentPreTrainLoss()

    def forward(self,
                text_tensors,
                text_lengths,
                gold_speech,
                speech_lengths,
                gold_durations,
                utterance_embedding,
                return_mels=False,
                lang_ids=None,
                run_glow=True
                ):
        """
        Args:
            return_mels (Boolean): whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            run_glow (Boolean): Whether to run the PostNet. There should be a warmup phase in the beginning.
            lang_ids (LongTensor): The language IDs used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Batch of embeddings to condition the TTS on, if the model is multispeaker
        """
        before_outs, \
        after_outs, \
        predicted_durations, \
        glow_loss = self._forward(text_tensors=text_tensors,
                                  text_lengths=text_lengths,
                                  gold_speech=gold_speech,
                                  speech_lengths=speech_lengths,
                                  gold_durations=gold_durations,
                                  utterance_embedding=utterance_embedding,
                                  is_inference=False,
                                  lang_ids=lang_ids,
                                  run_glow=run_glow)

        # calculate loss
        l1_loss, duration_loss = self.criterion(after_outs=after_outs,
                                                # if a regular PostNet is used, the post-PostNet outs have to go here. The flow has its own loss though, so we hard-code this to None
                                                before_outs=before_outs,
                                                gold_spectrograms=gold_speech,
                                                spectrogram_lengths=speech_lengths,
                                                text_lengths=text_lengths,
                                                gold_durations=gold_durations,
                                                predicted_durations=predicted_durations)

        if return_mels:
            if after_outs is None:
                after_outs = before_outs
            return l1_loss, duration_loss, glow_loss, after_outs
        return l1_loss, duration_loss, glow_loss

    def _forward(self,
                 text_tensors,
                 text_lengths,
                 gold_durations,
                 gold_speech=None,
                 speech_lengths=None,
                 content_training=True,
                 is_inference=False,
                 utterance_embedding=None,
                 lang_ids=None,
                 run_glow=True):

        if not self.encoder.multilingual_model:
            lang_ids = None

        if not self.encoder.multispeaker_model:
            utterance_embedding = None
        else:
            utterance_embedding = torch.nn.functional.normalize(utterance_embedding)

        encoded_texts, _, predicted_durations = self.encoder(text_tensors,text_lengths,utterance_embedding,lang_ids,gold_durations,content_training)

        # decoding spectrogram
        decoder_masks = make_non_pad_mask(speech_lengths, device=speech_lengths.device).unsqueeze(-2) if speech_lengths is not None and not is_inference else None
        decoded_speech, _ = self.decoder(encoded_texts, decoder_masks)
        decoded_spectrogram = self.feat_out(decoded_speech).view(decoded_speech.size(0), -1, self.output_spectrogram_channels)

        refined_spectrogram = decoded_spectrogram + self.conv_postnet(decoded_spectrogram.transpose(1, 2)).transpose(1, 2)

        # refine spectrogram further with a normalizing flow (requires warmup, so it's not always on)
        glow_loss = None
        if run_glow:
            if is_inference:
                refined_spectrogram = self.post_flow(tgt_mels=None,
                                                     infer=is_inference,
                                                     mel_out=refined_spectrogram,
                                                     encoded_texts=encoded_texts,
                                                     tgt_nonpadding=None).squeeze()
            else:
                glow_loss = self.post_flow(tgt_mels=gold_speech,
                                           infer=is_inference,
                                           mel_out=refined_spectrogram.detach().clone(),
                                           encoded_texts=encoded_texts.detach().clone(),
                                           tgt_nonpadding=decoder_masks)
        if is_inference:
            return decoded_spectrogram.squeeze(), \
                   refined_spectrogram.squeeze(), \
                   predicted_durations.squeeze(),
        else:
            return decoded_spectrogram, \
                   refined_spectrogram, \
                   predicted_durations, \
                   glow_loss

    @torch.inference_mode()
    def inference(self,
                  text,
                  gold_durations,
                  speech=None,
                  utterance_embedding=None,
                  return_duration=False,
                  lang_id=None,
                  run_postflow=True):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            return_duration_pitch_energy (Boolean): whether to return the list of predicted durations for nicer plotting
            run_postflow (Boolean): Whether to run the PostNet. There should be a warmup phase in the beginning.
            lang_id (LongTensor): The language ID used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Embedding to condition the TTS on, if the model is multispeaker
        """
        self.eval()
        x, y = text, speech

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)
        utterance_embeddings = utterance_embedding.unsqueeze(0) if utterance_embedding is not None else None

        before_outs, \
        after_outs, \
        duration_predictions = self._forward(xs,
                                           ilens,
                                           gold_durations,
                                           ys,
                                           content_training=False,
                                           is_inference=True,
                                           utterance_embedding=utterance_embeddings,
                                           lang_ids=lang_id,
                                           run_glow=run_postflow)  # (1, L, odim)
        #self.train()
        if after_outs is None:
            after_outs = before_outs
        if return_duration:
            return before_outs, after_outs, duration_predictions
        return after_outs

    def _reset_parameters(self, init_type):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)
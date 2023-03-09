import torch
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

from Layers.Conformer import Conformer
from Layers.LengthRegulator import LengthRegulator
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.StochasticVariancePredictor import StochasticVariancePredictor
from Utility.utils import make_non_pad_mask


class ToucanTTS(torch.nn.Module):

    def __init__(self,
                 # network structure related
                 input_feature_dimensions=62,
                 output_spectrogram_channels=80,
                 attention_dimension=192,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 use_scaled_positional_encoding=True,
                 # encoder / decoder
                 encoder_layers=6,
                 encoder_units=1536,
                 encoder_normalize_before=True,
                 encoder_concat_after=False,
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,
                 conformer_encoder_kernel_size=7,
                 decoder_layers=6,
                 decoder_units=1536,
                 decoder_concat_after=False,
                 conformer_decoder_kernel_size=31,
                 decoder_normalize_before=True,
                 # pitch predictor
                 pitch_embed_kernel_size=1,
                 pitch_embed_dropout=0.0,
                 # training related
                 transformer_enc_dropout_rate=0.2,
                 transformer_enc_positional_dropout_rate=0.2,
                 transformer_enc_attn_dropout_rate=0.2,
                 transformer_dec_dropout_rate=0.2,
                 transformer_dec_positional_dropout_rate=0.2,
                 transformer_dec_attn_dropout_rate=0.2,
                 # additional features
                 utt_embed_dim=64,
                 lang_embs=8000,
                 weights=None):
        super().__init__()

        # store hyperparameters
        self.idim = input_feature_dimensions
        self.odim = output_spectrogram_channels
        self.adim = attention_dimension
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None

        # define encoder
        embed = Sequential(Linear(input_feature_dimensions, 100),
                           Tanh(),
                           Linear(100, attention_dimension))
        self.encoder = Conformer(idim=input_feature_dimensions,
                                 attention_dim=attention_dimension,
                                 attention_heads=attention_heads,
                                 linear_units=encoder_units,
                                 num_blocks=encoder_layers,
                                 input_layer=embed,
                                 dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate,
                                 attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before,
                                 concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_encoder_kernel_size,
                                 zero_triu=False,
                                 utt_embed=None,
                                 lang_embs=lang_embs)

        self.duration_predictor = StochasticVariancePredictor(in_channels=attention_dimension,
                                                              kernel_size=3,
                                                              p_dropout=0.5,
                                                              n_flows=4,
                                                              conditioning_signal_channels=utt_embed_dim)

        self.pitch_predictor = StochasticVariancePredictor(in_channels=attention_dimension,
                                                           kernel_size=3,
                                                           p_dropout=0.5,
                                                           n_flows=4,
                                                           conditioning_signal_channels=utt_embed_dim)

        self.pitch_embed = Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=attention_dimension,
                            kernel_size=pitch_embed_kernel_size,
                            padding=(pitch_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(pitch_embed_dropout))

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
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
                                 utt_embed=None)

        # define final projection
        self.feat_out = Linear(attention_dimension, output_spectrogram_channels)

        # define speaker embedding integrations
        if self.multispeaker_model:
            self.decoder_in_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim, attention_dimension), LayerNorm(attention_dimension))
            self.decoder_out_embedding_projection = Sequential(Linear(output_spectrogram_channels + utt_embed_dim, output_spectrogram_channels), LayerNorm(output_spectrogram_channels))

        # post net is realized as a flow
        gin_channels = attention_dimension
        self.post_flow = Glow(
            in_channels=output_spectrogram_channels,
            hidden_channels=192,  # post_glow_hidden  (original 192 in paper)
            kernel_size=3,  # post_glow_kernel_size
            dilation_rate=1,
            n_blocks=16,  # post_glow_n_blocks (original 12 in paper)
            n_layers=3,  # post_glow_n_block_layers (original 3 in paper)
            n_split=4,
            n_sqz=2,
            text_condition_channels=gin_channels,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,  # share_wn_layers
            sigmoid_scale=False,  # sigmoid_scale
            condition_integration_projection=torch.nn.Conv1d(output_spectrogram_channels + attention_dimension, gin_channels, 5, padding=2)
        )

        self.load_state_dict(weights)
        self.eval()

    def _forward(self,
                 text_tensors,
                 text_lens,
                 gold_durations=None,
                 gold_pitch=None,
                 duration_scaling_factor=1.0,
                 utterance_embedding=None,
                 lang_ids=None,
                 pitch_variance_scale=1.0,
                 pause_duration_scaling_factor=1.0,
                 device=None):

        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward encoder
        text_masks = self._source_mask(text_lens)

        encoded_texts, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)

        if self.multispeaker_model:
            utterance_embedding_expanded = torch.nn.functional.normalize(utterance_embedding.unsqueeze(-1))
        else:
            utterance_embedding_expanded = None

        # predicting pitch, energy and duration.
        pitch_mask = torch.ones(size=[text_tensors.size(1)], device=text_tensors.device)
        duration_mask = torch.ones(size=[text_tensors.size(1)], device=text_tensors.device)
        for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze()):
            if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                duration_mask[phoneme_index] = 0
            if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                pitch_mask[phoneme_index] = 0.0

        if gold_durations is not None:
            predicted_durations = gold_durations
        else:
            predicted_durations = self.duration_predictor(encoded_texts.transpose(1, 2), duration_mask, w=None, g=utterance_embedding_expanded, reverse=True)
            predicted_durations = torch.ceil(torch.exp(predicted_durations)).long()
        if gold_pitch is not None:
            pitch_predictions = gold_pitch
        else:
            pitch_predictions = self.pitch_predictor(encoded_texts.transpose(1, 2), pitch_mask, w=None, g=utterance_embedding_expanded, reverse=True)
            pitch_scaling_factor_to_restore_mean = 1 - (sum(pitch_predictions) / len(pitch_predictions.squeeze()))
            pitch_predictions = pitch_predictions * pitch_scaling_factor_to_restore_mean  # we make sure the sequence has a mean of 1.0 to be closer to training

        for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
            if phoneme_vector[get_feature_to_index_lookup()["questionmark"]] == 1:
                if phoneme_index - 4 >= 0:
                    pitch_predictions[0][0][phoneme_index - 1] += .3
                    pitch_predictions[0][0][phoneme_index - 2] += .3
                    pitch_predictions[0][0][phoneme_index - 3] += .2
                    pitch_predictions[0][0][phoneme_index - 4] += .1
            if phoneme_vector[get_feature_to_index_lookup()["silence"]] == 1 and pause_duration_scaling_factor != 1.0:
                predicted_durations[phoneme_index] = torch.round(predicted_durations[0][0][phoneme_index].float() * pause_duration_scaling_factor).long()
            if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                # this means the phoneme is unvoiced and should therefore not have a pitch value (undefined, but we overload this with 0)
                pitch_predictions[0][0][phoneme_index] = 0.0
            if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                predicted_durations[0][0][phoneme_index] = 0
        if duration_scaling_factor != 1.0:
            assert duration_scaling_factor > 0
            predicted_durations = torch.round(predicted_durations.float() * duration_scaling_factor).long()
        pitch_predictions = _scale_variance(pitch_predictions, pitch_variance_scale)

        embedded_pitch_curve = self.pitch_embed(pitch_predictions).transpose(1, 2)
        encoded_texts = encoded_texts + embedded_pitch_curve
        upsampled_enriched_encoded_texts = self.length_regulator(encoded_texts, predicted_durations.squeeze(0))

        if utterance_embedding is not None:
            upsampled_enriched_encoded_texts = _integrate_with_utt_embed(hs=upsampled_enriched_encoded_texts,
                                                                         utt_embeddings=utterance_embedding,
                                                                         projection=self.decoder_in_embedding_projection)

        decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, None, utterance_embedding)
        predicted_spectrogram_before_postnet = self.feat_out(decoded_speech).view(decoded_speech.size(0), -1, self.odim)

        # forward flow post-net
        if utterance_embedding is not None:
            before_enriched = _integrate_with_utt_embed(hs=predicted_spectrogram_before_postnet,
                                                        utt_embeddings=utterance_embedding,
                                                        projection=self.decoder_out_embedding_projection)
        else:
            before_enriched = predicted_spectrogram_before_postnet
        predicted_spectrogram_after_postnet = self.post_flow(tgt_mels=None,
                                                             infer=True,
                                                             mel_out=before_enriched,
                                                             encoded_texts=upsampled_enriched_encoded_texts,
                                                             tgt_nonpadding=None)

        return predicted_spectrogram_before_postnet.squeeze(), predicted_spectrogram_after_postnet.squeeze(), predicted_durations.squeeze(), pitch_predictions.squeeze()

    @torch.inference_mode()
    def forward(self,
                text,
                durations=None,
                pitch=None,
                utterance_embedding=None,
                return_duration_pitch_energy=False,
                lang_id=None,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                pause_duration_scaling_factor=1.0,
                device=None):
        """
        Generate the sequence of spectrogram frames given the sequence of vectorized phonemes.

        Args:
            text: input sequence of vectorized phonemes
            durations: durations to be used (optional, if not provided, they will be predicted)
            pitch: token-averaged pitch curve to be used (optional, if not provided, it will be predicted)
            energy: token-averaged energy curve to be used (optional, if not provided, it will be predicted)
            return_duration_pitch_energy: whether to return the list of predicted durations for nicer plotting
            utterance_embedding: embedding of speaker information
            lang_id: id to be fed into the embedding layer that contains language information
            duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
            pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
            pause_duration_scaling_factor: reasonable values are 0.6 < scale < 1.4.
                                   scales the durations of pauses on top of the regular duration scaling

        Returns:
            mel spectrogram

        """
        # setup batch axis
        ilens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        if durations is not None:
            durations = durations.unsqueeze(0).to(text.device)
        if pitch is not None:
            pitch = pitch.unsqueeze(0).to(text.device)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0).to(text.device)

        before_outs, \
        after_outs, \
        predicted_durations, \
        pitch_predictions = self._forward(text.unsqueeze(0),
                                          ilens,
                                          gold_durations=durations,
                                          gold_pitch=pitch,
                                          utterance_embedding=utterance_embedding.unsqueeze(0),
                                          lang_ids=lang_id,
                                          duration_scaling_factor=duration_scaling_factor,
                                          pitch_variance_scale=pitch_variance_scale,
                                          pause_duration_scaling_factor=pause_duration_scaling_factor,
                                          device=device)
        if return_duration_pitch_energy:
            return after_outs, predicted_durations, pitch_predictions
        return after_outs

    def _source_mask(self, ilens):
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)


def _integrate_with_utt_embed(hs, utt_embeddings, projection):
    # concat hidden states with spk embeds and then apply projection
    embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
    hs = projection(torch.cat([hs, embeddings_expanded], dim=-1))
    return hs


def _scale_variance(sequence, scale):
    if scale == 1.0:
        return sequence
    average = sequence[0][sequence[0] != 0.0].mean()
    sequence = sequence - average  # center sequence around 0
    sequence = sequence * scale  # scale the variance
    sequence = sequence + average  # move center back to original with changed variance
    for sequence_index in range(len(sequence[0])):
        if sequence[0][sequence_index] < 0.0:
            sequence[0][sequence_index] = 0.0
    return sequence

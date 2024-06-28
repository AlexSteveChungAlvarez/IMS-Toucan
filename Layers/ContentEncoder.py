from torch import nn
from torch import sum
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

from Layers.Conformer import Conformer
from Layers.DurationPredictor import DurationPredictor
from Layers.LengthRegulator import LengthRegulator
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Utility.utils import make_non_pad_mask
from Utility.utils import make_pad_mask

class ContentEncoder(nn.Module):
    """
    Content Encoder with Conformer
    """
    def __init__(self,
                 # network structure related
                 input_feature_dimensions=62,
                 attention_dimension=192,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,
                 
                 # encoder
                 encoder_layers=6,
                 encoder_units=1536,
                 encoder_normalize_before=True,
                 encoder_concat_after=False,
                 conformer_encoder_kernel_size=7,
                 transformer_enc_dropout_rate=0.1,
                 transformer_enc_positional_dropout_rate=0.1,
                 transformer_enc_attn_dropout_rate=0.1,
                 
                 # duration predictor
                 duration_predictor_layers=3,
                 duration_predictor_chans=256,
                 duration_predictor_kernel_size=3,
                 duration_predictor_dropout_rate=0.2,
                 
                 # additional features
                 utt_embed_dim=64,
                 lang_embs=8000
                 ):
        super(ContentEncoder, self).__init__()
                
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        
        articulatory_feature_embedding = Sequential(Linear(input_feature_dimensions, 100), Tanh(), Linear(100, attention_dimension))
        
        self.encoder = Conformer(idim=input_feature_dimensions,
                                attention_dim=attention_dimension,
                                attention_heads=attention_heads,
                                linear_units=encoder_units,
                                num_blocks=encoder_layers,
                                input_layer=articulatory_feature_embedding,
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
                                utt_embed=utt_embed_dim,
                                lang_embs=lang_embs,
                                use_output_norm=True)
                
        self.duration_predictor = DurationPredictor(idim=attention_dimension, n_layers=duration_predictor_layers,
                                                    n_chans=duration_predictor_chans,
                                                    kernel_size=duration_predictor_kernel_size,
                                                    dropout_rate=duration_predictor_dropout_rate,
                                                    utt_embed_dim=utt_embed_dim)
        
        self.length_regulator = LengthRegulator()
        if lang_embs is not None:
            nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)
        
    def forward(self,text_tensors,text_lengths,utterance_embedding,lang_ids,gold_durations,is_inference):
        # encoding the texts
        text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)
        padding_masks = make_pad_mask(text_lengths, device=text_lengths.device)
        
        content_latents, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)
        if is_inference:
            encoder_masks = None
            predicted_durations = self.duration_predictor.inference(content_latents, padding_mask=None, utt_embed=utterance_embedding)
            # modifying the predictions with linguistic knowledge
            for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
                if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                    predicted_durations[0][phoneme_index] = 0
            content_latents = self.length_regulator(content_latents, predicted_durations)
        else:
            encoder_masks = make_non_pad_mask(sum(gold_durations,-1), device=gold_durations.device)
            predicted_durations = self.duration_predictor(content_latents, padding_mask=padding_masks, utt_embed=utterance_embedding)
            content_latents = self.length_regulator(content_latents, gold_durations)
            
        return content_latents,encoder_masks,predicted_durations
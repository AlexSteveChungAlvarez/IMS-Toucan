import torch
from Layers.ContentEncoder import ContentEncoder
from Layers.INEncoder import INEncoder
from Layers.AdaINConformer import AdaInConformer
from Layers.PostNet import PostNet
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.UnetTTS.UnetTTSLoss import UnetTTSLoss
from Utility.utils import initialize
from Utility.utils import make_non_pad_mask

class UnetTTS(torch.nn.Module):
    
    def __init__(self,
                 # Content Encoder weights
                 content_encoder_path,
                 
                 ## instance normalization structure related
                 adain_filter_size=192,
                 ada_in_kernel_size=5,
                 n_conv_blocks=6,
                                  
                 # IN encoder
                 content_latent_dim=192,
                 
                 # Ada-IN decoder
                 num_mels=80,
                 
                 weights=None
                 ):
        super().__init__()
        
        self.content_encoder = ContentEncoder()
        
        self.in_encoder = INEncoder(input_size=num_mels,
                                    in_hidden_size=adain_filter_size,
                                    out_hidden_size=content_latent_dim,
                                    n_conv_blocks=n_conv_blocks,
                                    enc_kernel_size=ada_in_kernel_size)

        self.ada_in_decoder = AdaInConformer()
        
        self.conv_postnet = PostNet(idim=0,
                                    odim=num_mels,
                                    n_layers=5,
                                    n_chans=256,
                                    n_filts=5,
                                    use_batch_norm=True,
                                    dropout_rate=0.5)

        self.post_flow = Glow(
            in_channels=num_mels,
            hidden_channels=192,  # post_glow_hidden
            kernel_size=5,  # post_glow_kernel_size
            dilation_rate=1,
            n_blocks=18,  # post_glow_n_blocks (original 12 in paper)
            n_layers=4,  # post_glow_n_block_layers (original 3 in paper)
            n_split=4,
            n_sqz=2,
            text_condition_channels=content_latent_dim,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,
            sigmoid_scale=False,
            condition_integration_projection=torch.nn.Conv1d(num_mels + content_latent_dim, content_latent_dim, 5, padding=2)
        )
        
        # initialize parameters
        self._reset_parameters(init_type="xavier_uniform")
        
        if weights:
            self.load_state_dict(weights)
            self.eval()
        else:
            check_dict = torch.load(content_encoder_path, map_location='cpu')
            self.load_state_dict(check_dict["model"],strict=False)
            self.content_encoder.eval()
            self.in_encoder.eval()
            self.ada_in_decoder.eval()
            self.criterion = UnetTTSLoss()        
        
    def forward(self,
                text_tensors,
                text_lengths,
                gold_speech,
                speech_lengths,
                gold_durations,
                utterance_embedding,
                return_mels=False,
                lang_ids=None,
                run_glow=True,
                is_inference=False):
        """
        Args:
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            lang_ids (LongTensor): The language IDs used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Batch of embeddings to condition the TTS on, if the model is multispeaker
        """
        mel_before,         \
        mel_after,          \
        content_latents,    \
        content_latents_pred,\
        predicted_durations,\
        glow_loss            = self._forward(text_tensors,
                                             text_lengths,
                                             gold_speech,
                                             speech_lengths,
                                             gold_durations,
                                             is_inference,
                                             utterance_embedding,
                                             lang_ids,
                                             run_glow)
        
        # calculate loss
        l1_loss, l2_loss, duration_loss = self.criterion(mel_after,
                                                         mel_before,
                                                         content_latents,
                                                         content_latents_pred,
                                                         gold_speech,
                                                         speech_lengths,
                                                         text_lengths,
                                                         gold_durations,
                                                         predicted_durations)
        if return_mels:
            if mel_after is None:
                mel_after = mel_before
            return l1_loss, l2_loss, duration_loss, glow_loss, mel_after
        return l1_loss, l2_loss, duration_loss, glow_loss
        
    def _forward(self, 
                 text_tensors,
                 text_lengths,
                 gold_speech,
                 speech_lengths=None,
                 gold_durations=None,
                 is_inference=False,
                 utterance_embedding=None,
                 lang_ids=None,
                 run_glow=True):
        
        if not self.content_encoder.multilingual_model:
            lang_ids = None

        if not self.content_encoder.multispeaker_model:
            utterance_embedding = None
        else:
            utterance_embedding = torch.nn.functional.normalize(utterance_embedding)

        content_latents,encoder_masks,predicted_durations = self.content_encoder(text_tensors,text_lengths,utterance_embedding,lang_ids,gold_durations,is_inference)
        if is_inference:
            tmp_masks = torch.ones([gold_speech.size(dim=0),gold_speech.size(dim=1)],dtype=bool,device=gold_speech.device)
            encoder_masks = torch.ones([content_latents.size(dim=0),content_latents.size(dim=1)],dtype=bool,device=content_latents.device)
        content_latents_pred, means, stds = self.in_encoder(gold_speech, encoder_masks if not is_inference else tmp_masks)
        decoder_masks = make_non_pad_mask(speech_lengths, device=speech_lengths.device).unsqueeze(-2) if speech_lengths is not None and not is_inference else None
        decoded_spectrogram = self.ada_in_decoder(content_latents, decoder_masks, (content_latents_pred, means, stds), encoder_masks)
        
        refined_spectrogram = decoded_spectrogram + self.conv_postnet(decoded_spectrogram.transpose(1, 2)).transpose(1, 2)

        # refine spectrogram further with a normalizing flow (requires warmup, so it's not always on)
        glow_loss = None
        if run_glow:
            if is_inference:
                refined_spectrogram = self.post_flow(tgt_mels=None,
                                                     infer=is_inference,
                                                     mel_out=refined_spectrogram,
                                                     encoded_texts=content_latents,
                                                     tgt_nonpadding=None).squeeze()
            else:
                glow_loss = self.post_flow(tgt_mels=gold_speech,
                                           infer=is_inference,
                                           mel_out=refined_spectrogram.detach().clone(),
                                           encoded_texts=content_latents.detach().clone(),
                                           tgt_nonpadding=decoder_masks)
        if is_inference:
            return decoded_spectrogram.squeeze(), refined_spectrogram.squeeze(), predicted_durations.squeeze()
        else:
            return decoded_spectrogram, refined_spectrogram, content_latents, content_latents_pred, predicted_durations, glow_loss
        
    @torch.inference_mode()
    def inference(self,
                  text,
                  speech,
                  gold_durations,
                  utterance_embedding=None,
                  lang_id=None,
                  run_postflow=True):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            return_duration (Boolean): whether to return the list of predicted durations for nicer plotting
            lang_id (LongTensor): The language ID used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Embedding to condition the TTS on, if the model is multispeaker
        """
        self.eval()
        x, y = text, speech

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), y.unsqueeze(0)

        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)
        utterance_embeddings = utterance_embedding.unsqueeze(0) if utterance_embedding is not None else None

        mel_before, \
        mel_after,  \
        duration_predictions=self._forward(xs,
                                           ilens,
                                           ys,
                                           gold_durations=gold_durations,
                                           is_inference=True,
                                           utterance_embedding=utterance_embeddings,
                                           lang_ids=lang_id,
                                           run_glow=run_postflow)  # (1, L, odim)
        self.train()
        return mel_before, mel_after, duration_predictions
    
    def _reset_parameters(self, init_type):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)
            
    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
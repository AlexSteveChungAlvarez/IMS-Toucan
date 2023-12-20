import torch
from Layers.ContentEncoder import ContentEncoder
from Layers.INEncoder import INEncoder
from Layers.AdaINDecoder import AdaINDecoder
from TrainingInterfaces.Text_to_Spectrogram.UnetTTS.UnetTTSLoss import UnetTTSLoss

class UnetTTS(torch.nn.Module):
    
    def __init__(self,
                 # Content Encoder weights
                 content_encoder_path,
                 
                 ## instance normalization structure related
                 adain_filter_size=256,
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

        self.ada_in_decoder = AdaINDecoder(input_size=content_latent_dim,
                                           in_hidden_size=adain_filter_size,
                                           out_hidden_size=num_mels,
                                           n_conv_blocks=n_conv_blocks,
                                           dec_kernel_size=ada_in_kernel_size,
                                           gen_kernel_size=ada_in_kernel_size)
        
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
                lang_ids=None,
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
        content_latents,    \
        content_latents_pred,\
        predicted_durations  = self._forward(text_tensors,
                                             text_lengths,
                                             gold_speech,
                                             gold_durations,
                                             utterance_embedding,
                                             lang_ids,
                                             is_inference)
        
        # calculate loss
        l1_loss, l2_loss, duration_loss = self.criterion(mel_before,
                                                         content_latents,
                                                         content_latents_pred,
                                                         gold_speech,
                                                         speech_lengths,
                                                         text_lengths,
                                                         gold_durations,
                                                         predicted_durations)
        
        return l1_loss, l2_loss, duration_loss
        
    def _forward(self, 
                 text_tensors,
                 text_lengths,
                 gold_speech,
                 gold_durations=None,
                 utterance_embedding=None,
                 lang_ids=None,
                 is_inference=False):
        
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
        mel_before = self.ada_in_decoder(content_latents, (content_latents_pred, means, stds), encoder_masks)

        if is_inference:
            return mel_before.squeeze(), predicted_durations.squeeze()
        else:
            return mel_before, content_latents, content_latents_pred, predicted_durations
        
    @torch.inference_mode()
    def inference(self,
                  text,
                  speech,
                  gold_durations,
                  utterance_embedding=None,
                  lang_id=None):
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
        duration_predictions=self._forward(xs,
                                           ilens,
                                           ys,
                                           gold_durations,
                                           is_inference=True,
                                           utterance_embedding=utterance_embeddings,
                                           lang_ids=lang_id)
        #self.train()
        return mel_before, duration_predictions
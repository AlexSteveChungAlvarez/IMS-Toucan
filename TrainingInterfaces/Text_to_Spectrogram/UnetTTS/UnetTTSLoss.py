import torch

from Layers.DurationPredictor import DurationPredictorLoss
from Utility.utils import make_non_pad_mask


class UnetTTSLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_criterion = torch.nn.L1Loss(reduction="none")
        self.duration_criterion = DurationPredictorLoss(reduction="none")
        self.mse_criterion = torch.nn.MSELoss(reduction="none")
    
    def forward(self, mel_after, mel_before, gold_spectrograms, spectrogram_lengths, text_lengths, gold_durations, predicted_durations):
        """
        Args:
            mel_before (Tensor): Batch of outputs of ada_in_decoder (B, Lmax, odim).
            content_latents (Tensor): Batch of outputs of content encoder after length_regulator (B, L*, odim).
            content_latents_pred (Tensor): Batch of outputs of encoder with Instance Normalization (B, L*, odim).
            gold_spectrograms (Tensor): Batch of target features (B, Lmax, odim).
            spectrogram_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of durations (B, Tmax).
            predicted_durations (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            text_lengths (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: L2 loss value.
            Tensor: Duration loss value
        """
        
        # calculate loss
        l1_loss = self.l1_criterion(mel_before, gold_spectrograms)
        if mel_after is not None:
            l1_loss = l1_loss + self.l1_criterion(mel_after, gold_spectrograms)        
        duration_loss = self.duration_criterion(predicted_durations, gold_durations)
        # make weighted mask and apply it
        out_masks = make_non_pad_mask(spectrogram_lengths).unsqueeze(-1).to(gold_spectrograms.device)
        out_masks = torch.nn.functional.pad(out_masks.transpose(1, 2), [0, gold_spectrograms.size(1) - out_masks.size(1), 0, 0, 0, 0], value=False).transpose(1, 2)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= gold_spectrograms.size(0) * gold_spectrograms.size(2)
        
        duration_masks = make_non_pad_mask(text_lengths).to(gold_spectrograms.device)
        duration_weights = (duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float())
        
        # apply weight
        l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
        duration_loss = duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
        
        return l1_loss, duration_loss
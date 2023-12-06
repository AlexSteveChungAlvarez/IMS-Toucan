import torch
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding

from Utility.path_to_transcript_dicts import *
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec_unettts


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, language_id
    return (pad_sequence([datapoint[0].squeeze() for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]),
            pad_sequence([datapoint[2].squeeze() for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]),
            pad_sequence([datapoint[4].squeeze() for datapoint in batch], batch_first=True),
            torch.stack([datapoint[5] for datapoint in batch]))


def train_loop(net,
               datasets,
               device,
               save_directory,
               batch_size,
               steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               lang,
               path_to_embed_model,
               resume,
               fine_tune,
               swa_start,
               use_wandb
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
    net = net.to(device)

    if steps % steps_per_checkpoint == 0:
        steps = steps + 1
    else:
        steps = steps + ((steps_per_checkpoint + 1) - (steps % steps_per_checkpoint))  # making sure to stop at the closest point that makes sense to the specified stopping point

    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embed_model, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    style_embedding_function.eval()
    style_embedding_function.requires_grad_(False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loaders = list()
    train_iters = list()
    for dataset in datasets:
        train_loaders.append(DataLoader(batch_size=1,
                                        dataset=dataset,
                                        drop_last=True,
                                        num_workers=2,
                                        pin_memory=True,
                                        shuffle=True,
                                        prefetch_factor=4,
                                        collate_fn=collate_and_pad,
                                        persistent_workers=True))
        train_iters.append(iter(train_loaders[-1]))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    swa_model = AveragedModel(net)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    grad_scaler = GradScaler()
    steps_run_previously = 0
    l1_losses_total = list()
    l2_losses_total = list()
    duration_losses_total = list()

    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            steps_run_previously = check_dict["step_counter"]
            grad_scaler.load_state_dict(check_dict["scaler"])
        if steps_run_previously > steps:
            print("Desired steps already reached in loaded checkpoint.")
            return

    net.train()
    # =============================
    # Actual train loop starts here
    # =============================
    for step_counter in tqdm(range(steps_run_previously, steps)):
        batches = []
        while len(batches) < batch_size:
            for index in random.sample(list(range(len(datasets))), len(datasets)):
                if len(batches) < batch_size:
                    # we get one batch for each task (i.e. language in this case) in a randomized order
                    try:
                        batch = next(train_iters[index])
                        batches.append(batch)
                    except StopIteration:
                        train_iters[index] = iter(train_loaders[index])
                        batch = next(train_iters[index])
                        batches.append(batch)
        batch = collate_and_pad(batches)

        text_tensors = batch[0].to(device)
        text_lengths = batch[1].squeeze().to(device)
        gold_speech = batch[2].to(device)
        speech_lengths = batch[3].squeeze().to(device)
        gold_durations = batch[4].to(device)
        lang_ids = batch[5].squeeze(1).to(device)

        train_loss = 0.0
        with autocast():
            # we sum the loss for each task, as we would do for the
            # second order regular MAML, but we do it only over one
            # step (i.e. iterations of inner loop = 1)
            style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                       batch_of_spectrogram_lengths=batch[3].to(device))
            l1_loss, l2_loss, duration_loss = net(
                text_tensors=text_tensors,
                text_lengths=text_lengths,
                gold_speech=gold_speech,
                speech_lengths=speech_lengths,
                gold_durations=gold_durations,
                utterance_embedding=style_embedding,
                lang_ids=lang_ids)

            # then we directly update our meta-parameters without
            # the need for any task specific parameters

            if not torch.isnan(l1_loss):
                train_loss = train_loss + l1_loss
            if not torch.isnan(l2_loss):
                train_loss = train_loss + l2_loss
            if not torch.isnan(duration_loss):
                train_loss = train_loss + duration_loss

        l1_losses_total.append(l1_loss.item())
        l2_losses_total.append(l2_loss.item())
        duration_losses_total.append(duration_loss.item())

        optimizer.zero_grad()
        grad_scaler.scale(train_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if step_counter > swa_start:
            swa_model.update_parameters(net)
            swa_scheduler.step()
        else:
            scheduler.step()

        if step_counter % steps_per_checkpoint == 0 and step_counter != 0:
            # ==============================
            # Enough steps for some insights
            # ==============================
            if step_counter > swa_start:
                torch.optim.swa_utils.update_bn(train_loaders, swa_model)
            net.eval()
            style_embedding_function.eval()
            default_embedding = style_embedding_function(
                batch_of_spectrograms=datasets[0][0][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=datasets[0][0][3].unsqueeze(0).to(device)).squeeze()
            print("Reconstruction Loss:    {}".format(round(sum(l1_losses_total) / len(l1_losses_total), 3)))
            print("Content Encoder Loss:    {}".format(round(sum(l2_losses_total) / len(l2_losses_total), 3)))
            print("Steps:                  {}\n".format(step_counter))
            if step_counter > swa_start:
                torch.save({
                    "model"       : swa_model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "scaler"      : grad_scaler.state_dict(),
                    "scheduler"   : swa_scheduler.state_dict(),
                    "step_counter": step_counter,
                    "default_emb" : default_embedding,
                },
                    os.path.join(save_directory, "best.pt".format(step_counter)))
            else:
                torch.save({
                    "model"       : net.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "scaler"      : grad_scaler.state_dict(),
                    "scheduler"   : scheduler.state_dict(),
                    "step_counter": step_counter,
                    "default_emb" : default_embedding,
                },
                    os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(save_directory, keep=5)

            if use_wandb:
                wandb.log({
                    "l1_loss"      : round(sum(l1_losses_total) / len(l1_losses_total), 5),
                    "l2_loss"      : round(sum(l2_losses_total) / len(l1_losses_total), 5),
                    "duration_loss": round(sum(duration_losses_total) / len(duration_losses_total), 5),
                }, step=step_counter)

            try:
                path_to_most_recent_plot = plot_progress_spec_unettts(net,
                                                                      device,
                                                                      save_directory,
                                                                      step_counter,
                                                                      gold_speech,
                                                                      gold_durations,
                                                                      lang,
                                                                      default_embedding)
                if use_wandb:
                    wandb.log({
                        "progress_plot": wandb.Image(path_to_most_recent_plot)
                    }, step=step_counter)
            except IndexError:
                print("generating progress plots failed.")

            l1_losses_total = list()
            l2_losses_total = list()
            duration_losses_total = list()
            net.train()
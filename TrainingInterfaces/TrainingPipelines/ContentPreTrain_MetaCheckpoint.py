import time

import torch
import torch.multiprocessing
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.ContentPreTrain.ContentPreTrain import ContentPreTrain
from TrainingInterfaces.Text_to_Spectrogram.ContentPreTrain.content_pretrain_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_unettts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    # It is not recommended training this yourself or to finetune this, but you can.
    # The recommended use is to download the pretrained model from the GitHub release
    # page and finetune to your desired data

    datasets = list()

    base_dir = os.path.join(MODELS_DIR, "UnetTTS_Meta")
    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = os.path.join(base_dir,"Content_Encoder")
    os.makedirs(meta_save_dir, exist_ok=True)

    print("Preparing")

    english_datasets = list()
    german_datasets = list()
    spanish_datasets = list()
    french_datasets = list()
    portuguese_datasets = list()
    quechua_datasets = list()

    english_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_librittsr(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "LibriTTS_R"),
                                                      lang="en"))
    
    english_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_common_voice_english(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "cv-corpus", "english"),
                                                      lang="en"))

    german_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_common_voice_german(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "cv-corpus", "german"),
                                                     lang="de"))

    german_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_mls_german(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "MultiLingLibriSpeech", "german"),
                                                     lang="de"))

    spanish_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_quest_spanish(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "QuEsT","es"),
                                                      lang="es"))

    spanish_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_common_voice_spanish(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "cv-corpus", "spanish"),
                                                      lang="es"))

    spanish_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "MultiLingLibriSpeech", "spanish"),
                                                      lang="es"))
    
    spanish_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_pespa(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "pe_spa"),
                                                      lang="es"))

    french_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_common_voice_french(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "cv-corpus", "french"),
                                                     lang="fr"))

    french_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_mls_french(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "MultiLingLibriSpeech", "french"),
                                                     lang="fr"))

    portuguese_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese(),
                                                         corpus_dir=os.path.join(PREPROCESSING_DIR, "MultiLingLibriSpeech", "portuguese"),
                                                         lang="pt-br"))
    
    portuguese_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_common_voice_portuguese(),
                                                         corpus_dir=os.path.join(PREPROCESSING_DIR, "cv-corpus", "portuguese"),
                                                         lang="pt-br"))
    
    quechua_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_quest_quechua(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "QuEsT","qu"),
                                                      lang="qu"))
    
    quechua_datasets.append(prepare_unettts_corpus(transcript_dict=build_path_to_transcript_dict_quechua(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "QuechuaSingleSpeaker"),
                                                      lang="qu"))

    datasets.append(ConcatDataset(english_datasets))
    datasets.append(ConcatDataset(german_datasets))
    datasets.append(ConcatDataset(spanish_datasets))
    datasets.append(ConcatDataset(french_datasets))
    datasets.append(ConcatDataset(portuguese_datasets))
    datasets.append(ConcatDataset(quechua_datasets))

    model = ContentPreTrain()
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    train_loop(net=model,
               device=torch.device("cuda"),
               datasets=datasets,
               save_directory=meta_save_dir,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               resume=resume,
               fine_tune=finetune,
               steps=160000,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()

import torch
import argparse
import pickle

from transformers import (
    WhisperForConditionalGeneration,
    set_seed
)

from utils import data_handling
from utils import ft_utils

from fine_tuning.data_collator import WhisperDataCollator
from fine_tuning import ewc_trainer

def get_ewc_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--fisher_path", required=True,
                        help="Path to Fisher Information pickle file")
    p.add_argument("--ewc_lambda", type=float, default=5000,
                        help="EWC regularization strength (default: 5000)")
    
    return p.parse_args(argv)

def main():
    ft_args, rest = ft_utils.get_args()
    ewc_args = get_ewc_args(rest)

    # reproducibility
    set_seed(ft_args.seed)
    torch.manual_seed(ft_args.seed)

    # load dataset
    print(f"loading dataset from {ft_args.dataset_dir}")
    dataset = data_handling.load_local_dataset(ft_args.dataset_dir, ft_args.audio_column, ft_args.sampling_rate)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Store original parameters (BEFORE adding new token)
    print(f"Loading original model for parameter storage...")
    model_id = f"openai/whisper-{ft_args.model_size}"
    original_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    original_params = {}
    for name, param in original_model.named_parameters():
        original_params[name] = param.data.clone().to(device)

    # load model & processor
    model, processor = ft_utils.load_custom_whisper(ft_args.model_size, ft_args.lang_code)

    # prepare dataset
    dataset = data_handling.prepare_dataset(processor, dataset, ft_args.audio_column, ft_args.text_column)

    # data collator
    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # load fisher information
    print(f"Loading fisher information from {ewc_args.fisher_path}")
    with open(ewc_args.fisher_path, 'rb') as f:
        fisher_dict = pickle.load(f)

    fisher_dict = {k: v.to(device) for k, v in fisher_dict.items()}

    trainer = ewc_trainer.build_ewc_trainer(model, processor, dataset, data_collator, fisher_dict, ewc_args.ewc_lambda, original_params, ft_args)

    # save processor before training
    processor.save_pretrained(ft_args.output_dir)

    # training
    print(f"\nstarting training with custom language: {ft_args.lang_code}")
    if ft_args.lang_alias:
        print(f"  alias: {ft_args.lang_alias}")

    trainer.train()

    # save best model
    ft_utils.save_best_model(trainer, processor, ft_args.output_dir)

if __name__ == "__main__":
    main()

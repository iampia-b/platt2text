import torch
from transformers import set_seed

from utils import data_handling
from utils import ft_utils
from fine_tuning.data_collator import WhisperDataCollator

def main():
    args = ft_utils.get_args()

    # reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # load dataset
    print(f"loading dataset from {args.dataset_dir}")
    dataset = data_handling.load_local_dataset(args.dataset_dir, args.audio_column, args.sampling_rate)

    # load model & processor
    model, processor = ft_utils.load_custom_whisper(args.model_size, args.lang_code)

    # prepare dataset
    dataset = data_handling.prepare_dataset(processor, dataset, args.audio_column, args.text_column)

    # data collator
    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    trainer = ft_utils.build_trainer(model, processor, dataset, data_collator, args)

    # save processor before training
    processor.save_pretrained(args.output_dir)

    # training
    print(f"\nstarting training with custom language: {args.lang_code}")
    if args.lang_alias:
        print(f"  alias: {args.lang_alias}")

    trainer.train()

    # save best model
    ft_utils.save_best_model(trainer, processor, args.output_dir)

if __name__ == "__main__":
    main()

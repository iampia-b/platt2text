# Custom Whisper Fine-tuning with Dynamic Language Support

A framework for fine-tuning OpenAI's Whisper models to support new languages not included in the original model.
The implementation provides a custom tokenizer and processor class that integrate with the Hugging Face transformers library.

## Files

- `custom_whisper.py` - Custom tokenizer and processor classes that handle new language tokens
- `fine_tune_whisper.py` - Training script for fine-tuning Whisper with custom languages - based on the Hugging Face guide: [Fine-Tune Whisper For Multilingual ASR with Transformers](https://huggingface.co/blog/fine-tune-whisper)

## Key Features

- Adds new language tokens to existing Whisper models for fine-tuning
- Automatically handles token ID management and model vocabulary resizing

## Training Parameters

New parameters for fine-tuning with a new language:
- `--lang_code`: Language token (e.g., "nds")
- `--lang_alias`: Name for the language (e.g., "Low German")

## Evaluation

Tools for evaluating fine-tuned Whisper models and analyzing training runs.

### Files
- `model_evaluation/scripts/evaluate_wer.py` – Compute WER/CER on HF or local datasets
- `model_evaluation/scripts/evaluate_forgetting.py` – Compare runs to assess catastrophic forgetting
- `model_evaluation/scripts/analyze_training.py` – Summarize training dynamics (best WER, steps, runtime)

## Acknowledgments

Based on the Hugging Face guide by Sanchit Gandhi: [Fine-Tune Whisper For Multilingual ASR with Transformers](https://huggingface.co/blog/fine-tune-whisper)
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
- `--lang_alias`: Name for the language

## Acknowledgments

Based on the Hugging Face guide by Sanchit Gandhi: [Fine-Tune Whisper For Multilingual ASR with Transformers](https://huggingface.co/blog/fine-tune-whisper)
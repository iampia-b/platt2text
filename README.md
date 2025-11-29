# Fine-Tuning Whisper for Low German

BSc thesis project in Computer Science at TUHH: Fine-tuning OpenAI's Whisper model for Low German (Plattdeutsch) automatic speech recognition.

## Project Overview

This repository implements and compares different fine-tuning strategies for adapting Whisper to Low German, a low-resource language not included in Whisper's original training. 

## Repository Structure

```
platt2text/
├── core/                
│   └── custom_whisper.py       # Custom Whisper tokenizer and processor for new language tokens
├── fine_tuning/           
│   ├── standard.py                         # Standard fine-tuning with custom language tokens
│   └── ws_ft.py                            # Weighted-sum initialization for better zero-shot performance
│   └── ewc_ft.py                           # Elastic Weight Consolidation to reduce catastrophic forgetting
│   └── ewc_calc.py                         # calculation of fisher information for ewc
│   └── ewc_trainer.py                      # trainer for ewc_ft
└── evaluation/
│   ├── eval_error_rates.py                 # WER/CER evaluation on test sets
│   ├── eval_catastrophic_forgetting.py     # Catastrophic forgetting analysis
│   └── eval_training.py                    # Training dynamics and convergence analysis
│   └── compare_embeddings.py               # Comparison of language embeddings using cosine similarity
└── utils/
    └── data_handling.py                    # Utilities for loading and preprocessing speech datasets (local HF datasets and Common Voice TSVs)
    └── ft_utils.py                         # Helper functions for setting up a custom-language Whisper model for fine-tuning
```

## Requirements

```
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
evaluate>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
tqdm
```

## Acknowledgments

Fine-tuning code is based on the Hugging Face guide by Sanchit Gandhi: [Fine-Tune Whisper For Multilingual ASR with Transformers](https://huggingface.co/blog/fine-tune-whisper)
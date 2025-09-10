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
│   └── weighted_sum.py                     # Weighted-sum initialization for better zero-shot performance
│   └── elastic_weight_consolidation.py     # to be uploaded
└── evaluation/
    ├── eval_error_rates.py                 # WER/CER evaluation on test sets
    ├── eval_catastrophic_forgetting.py     # Catastrophic forgetting analysis
    └── eval_training.py                    # Training dynamics and convergence analysis
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
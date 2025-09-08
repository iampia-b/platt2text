import logging
from datasets import load_from_disk, load_dataset, Audio
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    mode: str = "hf"
    path: str = ""  
    hf_dataset: str = "mozilla-foundation/common_voice_17_0"
    hf_config: str = "de"
    hf_split: str = "test"
    hf_audio_col: str = "audio"
    hf_text_col: str = "sentence"
    sampling_rate: int = 16000
    sample_limit: int = 0  # 0 = no limit


def load_test_dataset(config: DatasetConfig):

    if config.mode == "local":
        return load_local_dataset(config.path, config.sampling_rate)
    else:
        return load_hf_dataset(config)


def load_local_dataset(dataset_path, sampling_rate = 16000):
    logger.info(f"Loading local dataset from: {dataset_path}")
    
    ds = load_from_disk(dataset_path)
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=sampling_rate))
    
    # test split
    if "test" in ds:
        test_ds = ds["test"]
    else:
        test_ds = ds
    
    if "audio_filepath" not in test_ds.column_names:
        if "audio" in test_ds.column_names:
            test_ds = test_ds.rename_column("audio", "audio_filepath")
    
    return test_ds


def load_hf_dataset(config: DatasetConfig):

    logger.info(
        f"Loading HF dataset: {config.hf_dataset} [{config.hf_config}] "
        f"split={config.hf_split}"
    )
    
    ds = load_dataset(
        config.hf_dataset, 
        config.hf_config, 
        split=config.hf_split,
        trust_remote_code=True
    )
    
    ds = ds.cast_column(config.hf_audio_col, Audio(sampling_rate=config.sampling_rate))
    
    keep_cols = [config.hf_audio_col, config.hf_text_col]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    
    ds = ds.rename_columns({
        config.hf_audio_col: "audio_filepath",
        config.hf_text_col: "text"
    })
    
    # sample limit
    if config.sample_limit > 0:
        logger.info(f"Limiting to {config.sample_limit} samples")
        ds = ds.select(range(min(config.sample_limit, len(ds))))
    
    return ds


def calculate_dataset_stats(dataset):
    total_seconds = 0.0
    total_words = 0
    n = len(dataset)
    
    for item in dataset:
        audio = item["audio_filepath"]
        total_seconds += len(audio["array"]) / audio["sampling_rate"]
        total_words += len(item["text"].split())
    
    return {
        "num_samples": n,
        "total_duration_hours": total_seconds / 3600.0,
        "avg_duration_seconds": (total_seconds / n) if n else 0.0,
        "avg_text_length_words": (total_words / n) if n else 0.0,
    }
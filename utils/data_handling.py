import os

from datasets import load_from_disk, load_dataset, Audio

def prepare_batch(batch, processor, audio_col, text_col):
    # load and resample audio data
    audio = batch[audio_col]

    # compute log-mel input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    ids = processor.tokenizer(batch[text_col], add_special_tokens=False).input_ids
    ids = ids + [processor.tokenizer.eos_token_id]    
    batch["labels"] = ids
    return batch

def prepare_dataset(processor, dataset, audio_column, text_column):
    print("preparing dataset...")
    columns = dataset["train"].column_names
    dataset = dataset.map(
        lambda batch: prepare_batch(batch, processor, audio_column, text_column),
        remove_columns=columns,
        num_proc=1,
    )

    input_feat = dataset['train'][0]['input_features']
    print("\nProcessed dataset check:")
    print(f"  Keys: {dataset['train'][0].keys()}")
    print(f"  Input features shape: ({len(input_feat)}, {len(input_feat[0])})")
    print(f"  Labels length: {len(dataset['train'][0]['labels'])}")
    print(f"  First 10 label IDs: {dataset['train'][0]['labels'][:10]}")

    return dataset

def load_local_dataset(dataset_dir, audio_column, sampling_rate):
    print(f"loading dataset from {dataset_dir}")
    dataset = load_from_disk(dataset_dir)
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    return dataset


def load_local_cv_dataset(root_dir, split, sample_limit= 500, seed = 42, audio_column = "audio", text_column = "sentence"):

    # load TSV metadata as a dataset
    split_tsv = split+".tsv"
    tsv_path = os.path.join(root_dir, split_tsv)

    ds_dict = load_dataset(
        "csv",
        data_files={"data": tsv_path},
        delimiter="\t",
    )
    ds = ds_dict["data"] 

    def _add_fields(batch):

        abs_paths = [os.path.join(root_dir, "clips", rel) for rel in batch["path"]]
        transcripts = batch["sentence"]

        batch[audio_column] = abs_paths
        batch[text_column] = transcripts
        return batch

    ds = ds.map(_add_fields, batched=True)
    ds = ds.cast_column(audio_column, Audio(sampling_rate=16000))

    ds = ds.shuffle(seed=seed)
    if sample_limit is not None:
        n = min(sample_limit, len(ds))
        ds = ds.select(range(n))

    keep_cols = [c for c in [audio_column, text_column] if c in ds.column_names]
    ds = ds.select_columns(keep_cols)

    print(f"{split_tsv} loaded from {root_dir}")
    print(f"final size: {len(ds)} samples")
    return ds

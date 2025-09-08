import re
from dataclasses import dataclass


@dataclass
class ExperimentInfo:

    model: str
    language: str
    learning_rate: str
    seed: int
    folder: str
    job_id: str = None
    array_idx: str = None


def parse_experiment_name(folder_name):
    # expected pattern: <model><language>_lr<lr>_<seed>[_<job_id>[_<array_idx>]]

    pattern = r'^(?P<prefix>.+?)_lr(?P<lr>[^_]+)_(?P<seed>\d+)(?:_(?P<job_id>\d+))?(?:_(?P<array_idx>\d+))?$'
    match = re.match(pattern, folder_name)
    
    if not match:
        raise ValueError(f"Unexpected experiment folder format: {folder_name}")
    
    prefix = match.group('prefix')
    
    # detecting language
    if prefix.endswith('low_german'):
        language = 'low_german'
        model = prefix[:-len('low_german')]
    elif prefix.endswith('german'):
        language = 'german'
        model = prefix[:-len('german')]
    else:
        language = 'unknown'
        model = prefix
    
    model = model.rstrip('-_')
    
    lr_token = match.group('lr')
    lr_norm = re.sub(r'^(\d+)e(\d+)$', lambda m: f"{m.group(1)}e-{m.group(2)}", lr_token)
    
    return ExperimentInfo(
        model=model,
        language=language,
        learning_rate=lr_norm,
        seed=int(match.group('seed')),
        folder=folder_name,
        job_id=match.group('job_id'),
        array_idx=match.group('array_idx')
    )


def find_model_checkpoint(exp_folder):

    best_path = exp_folder / 'best'
    if best_path.exists():
        return best_path
    
    checkpoints = list(exp_folder.glob('checkpoint-*'))
    if checkpoints:
        return max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
    
    return None


def discover_experiments(exp_dir):

    experiments = []
    for folder in sorted(exp_dir.iterdir()):
        if folder.is_dir() and find_model_checkpoint(folder):
            experiments.append(folder)
    return experiments
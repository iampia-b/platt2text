import logging

from ..corelib.models import load_model, transcribe_dataset, map_language_code
from ..corelib.metrics import normalize_text, calculate_metrics, calculate_per_sample_metrics
from ..corelib.experiment import find_model_checkpoint


logger = logging.getLogger(__name__)


def evaluate_model(
    model_path,
    dataset,
    device= "cuda",
    override_language = None
    ):
    try:
        
        model, processor = load_model(model_path, device)
        
        language = map_language_code(override_language) if override_language else None
        
        predictions, references = transcribe_dataset(
            model, processor, dataset, device, language, model_path
        )
        
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]
        
        metrics = calculate_metrics(predictions, references)
        per_sample = calculate_per_sample_metrics(predictions, references)
        
        return metrics, per_sample
        
    except Exception as e:
        logger.error(f"Failed to evaluate {model_path}: {e}")
        return None, None


def evaluate_experiment_folder(
    exp_folder,
    dataset,
    device= "cuda",
    override_language = None
    ):

    model_path = find_model_checkpoint(exp_folder)
    
    if not model_path:
        logger.warning(f"No model checkpoint found in {exp_folder}")
        return None
    
    return evaluate_model(model_path, dataset, device, override_language)
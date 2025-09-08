import evaluate
import numpy as np


def normalize_text(text):
    return ' '.join(text.lower().strip().split())

def calculate_metrics(predictions, references):

    if not predictions or not references:
        return {
            'wer': 0.0,
            'cer': 0.0,
            'num_samples': 0,
            'avg_ref_length': 0.0,
            'avg_pred_length': 0.0
        }
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    wer = wer_metric.compute(predictions=predictions, references=references) * 100
    cer = cer_metric.compute(predictions=predictions, references=references) * 100
    
    avg_ref_length = np.mean([len(ref.split()) for ref in references])
    avg_pred_length = np.mean([len(pred.split()) for pred in predictions])
    
    return {
        'wer': wer,
        'cer': cer,
        'num_samples': len(predictions),
        'avg_ref_length': avg_ref_length,
        'avg_pred_length': avg_pred_length
    }


def calculate_per_sample_metrics(predictions, references):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    results = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        results.append({
            'index': i,
            'reference': ref,
            'prediction': pred,
            'wer': wer_metric.compute(predictions=[pred], references=[ref]) * 100,
            'cer': cer_metric.compute(predictions=[pred], references=[ref]) * 100
        })
    
    return results
import json
import logging
import numpy as np


logger = logging.getLogger(__name__)


def extract_training_metrics(exp_path):
    metrics = {
        'training_completed': False,
        'best_wer': None,
        'best_epoch': None,
        'total_steps': None,
        'convergence_step': None,
        'training_time_hours': None,
        'final_train_loss': None,
        'best_eval_loss': None,
        'wer_improvement': None,
        'stability_score': None
    }
    
    # find trainer state
    trainer_state_path = find_trainer_state(exp_path)
    if not trainer_state_path:
        logger.warning(f"No trainer state found for {exp_path.name}")
        return metrics
    
    # load state
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    metrics['training_completed'] = True
    metrics['total_steps'] = trainer_state.get('global_step', 0)
    metrics['best_epoch'] = trainer_state.get('epoch', 0)
    
    # extract from log history
    curves = _extract_training_curves(trainer_state.get('log_history', []))
    
    # calculate metrics
    if curves['eval_wers']:
        best_wer = min(curves['eval_wers'], key=lambda x: x[1])
        metrics['best_wer'] = best_wer[1]
        metrics['convergence_step'] = best_wer[0]
        
        # WER improvement
        if curves['eval_wers']:
            initial_wer = curves['eval_wers'][0][1]
            metrics['wer_improvement'] = initial_wer - metrics['best_wer']
        
        # stability (variance in last quarter)
        if len(curves['eval_wers']) > 4:
            last_quarter = [w[1] for w in curves['eval_wers'][-len(curves['eval_wers'])//4:]]
            metrics['stability_score'] = np.std(last_quarter)
    
    # final losses
    if curves['train_losses']:
        metrics['final_train_loss'] = curves['train_losses'][-1][1]
    if curves['eval_losses']:
        metrics['best_eval_loss'] = min(curves['eval_losses'], key=lambda x: x[1])[1]
    
    # training time
    runtime = _extract_runtime(trainer_state)
    if runtime:
        metrics['training_time_hours'] = runtime / 3600.0
    
    return metrics


def find_trainer_state(exp_path):
    
    best_path = exp_path / 'best'
    if best_path.exists():
        return best_path
    
    checkpoints = list(exp_path.glob('checkpoint-*'))
    if checkpoints:
        return max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
    
    
    return None


def _extract_training_curves(log_history):
    train_losses = []
    eval_losses = []
    eval_wers = []
    
    for entry in log_history:
        step = entry.get('step', 0)
        
        if 'loss' in entry and 'eval_loss' not in entry:
            train_losses.append((step, entry['loss']))
        if 'eval_loss' in entry:
            eval_losses.append((step, entry['eval_loss']))
        if 'eval_wer' in entry:
            eval_wers.append((step, entry['eval_wer']))
    
    return {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'eval_wers': eval_wers
    }


def _extract_runtime(trainer_state):
    # Direct field
    runtime = trainer_state.get('train_runtime')
    if runtime:
        return runtime
    
    # metrics
    runtime = trainer_state.get('metrics', {}).get('train_runtime')
    if runtime:
        return runtime
    
    # log history
    for record in reversed(trainer_state.get('log_history', [])):
        if 'train_runtime' in record:
            return record['train_runtime']
    
    return None


def calculate_efficiency_metrics(metrics):
    efficiency = {}
    
    # efficiency score (WER improvement per hour)
    if metrics.get('wer_improvement') and metrics.get('training_time_hours'):
        efficiency['efficiency_score'] = (
            metrics['wer_improvement'] / metrics['training_time_hours']
        )
    
    # convergence ratio
    if metrics.get('convergence_step') and metrics.get('total_steps'):
        efficiency['convergence_ratio'] = (
            metrics['convergence_step'] / metrics['total_steps']
        )
    
    # overfitting score
    if metrics.get('final_train_loss') and metrics.get('best_eval_loss'):
        efficiency['overfitting_score'] = (
            metrics['best_eval_loss'] - metrics['final_train_loss']
        )
    
    return efficiency
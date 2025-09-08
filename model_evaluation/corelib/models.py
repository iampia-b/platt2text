import json
import logging
import torch
from transformers import WhisperForConditionalGeneration
from tqdm import tqdm

from custom_whisper import CustomWhisperProcessor

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    'de': 'german',
    'nde': 'low_german', 
    'en': 'english',
    'nl': 'dutch'
}

def map_language_code(code_or_name):
    x = (code_or_name or '').lower().strip()
    return LANGUAGE_MAP.get(x, x)


def load_model(
    model_path,
    device= "cuda"
    ):

    logger.info(f"Loading model from {model_path}")
    
    processor = CustomWhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # clear forced decoder IDs
    model.generation_config.forced_decoder_ids = None
    
    return model, processor

def get_model_language(model_path):

    config_path = model_path.parent / 'experiment_config.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            return (
                config.get('language', 'german'),
                config.get('language_code', 'de')
            )
    
    return 'german', 'de'

def transcribe_batch(
    model: WhisperForConditionalGeneration,
    processor: CustomWhisperProcessor,
    audio_batch,
    device= "cuda",
    language= "german"
):
    
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")
    
    transcriptions = []
    
    for audio in audio_batch:
        # process audio
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features.to(device)
        
        # generate
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=1, 
                forced_decoder_ids=None,
                suppress_tokens=None,
                do_sample=False
            )
        
        # decode
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        transcriptions.append(transcription)
    
    return transcriptions

def transcribe_dataset(
    model: WhisperForConditionalGeneration,
    processor: CustomWhisperProcessor,
    dataset,
    device = "cuda",
    language = None,
    model_path = None
):

    if language is None:
        language, _ = get_model_language(model_path)
    
    predictions = []
    references = []
    
    logger.info(f"Transcribing {len(dataset)} samples with language={language}")
    
    for item in tqdm(dataset, desc="Transcribing"):
        try:
            transcription = transcribe_batch(
                model, processor,
                [item["audio_filepath"]],
                device, language
            )[0]
            
            predictions.append(transcription)
            references.append(item["text"])
            
        except Exception as e:
            logger.error(f"Error transcribing sample: {e}")
            predictions.append("")
            references.append(item["text"])
    
    return predictions, references
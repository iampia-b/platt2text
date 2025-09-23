import os
import sys
import json
import torch
from pathlib import Path
from transformers import (
    WhisperProcessor, 
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration
)


class CustomWhisperTokenizer(WhisperTokenizer):
    
    def __init__(self, *args, custom_languages=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_languages = custom_languages or {}
        
    def save_pretrained(self, save_directory, **kwargs):
        outputs = super().save_pretrained(save_directory, **kwargs)
        
        # saving custom language info
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            config["tokenizer_class"] = "CustomWhisperTokenizer"
            config["custom_languages"] = self.custom_languages
            
            with open(config_file, 'w') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # loading custom languages if present
        config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                tokenizer.custom_languages = config.get("custom_languages", {})
        
        return tokenizer


class CustomWhisperProcessor(WhisperProcessor):
    # processor that uses CustomWhisperTokenizer and handles language token addition
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        
        # check for custom tokenizer
        config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        use_custom = False
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                if config.get("tokenizer_class") == "CustomWhisperTokenizer":
                    use_custom = True
        
        if use_custom:
            tokenizer = CustomWhisperTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        else:
            tokenizer = WhisperTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    def add_language_token(self, model, lang_code, init_vector=None):
        
        lang_code = lang_code.strip().lower()
        
        lang_token = f"<|{lang_code}|>"
        
        # add token if not present
        special_tokens = self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
        if lang_token not in special_tokens:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": special_tokens + [lang_token]
            })
            print(f"Added token: {lang_token}")
        
        # resizing embeddings
        before = model.model.decoder.embed_tokens.weight.shape[0]
        model.resize_token_embeddings(len(self.tokenizer))
        after = model.model.decoder.embed_tokens.weight.shape[0]
        print(f"Decoder embeddings: {before} -> {after}")
        
        # token ID
        token_id = self.tokenizer.convert_tokens_to_ids(lang_token)
        
        # optional initialization vector
        if init_vector is not None and token_id >= 0:
            with torch.no_grad():
                model.get_input_embeddings().weight[token_id] = init_vector.to(model.device)
                model.get_output_embeddings().weight[token_id] = init_vector.to(model.device)
        
        # updating generation_config.lang_to_id
        if hasattr(model, "generation_config"):
            if not hasattr(model.generation_config, "lang_to_id"):
                model.generation_config.lang_to_id = {}
            elif model.generation_config.lang_to_id is None:
                model.generation_config.lang_to_id = {}
            
            model.generation_config.lang_to_id[lang_token] = int(token_id)
        
        print(f"Token ID for {lang_token}: {token_id}")
        return token_id


def setup_model_with_custom_language(model_name, lang_code):
    
    processor = CustomWhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    token_id = processor.add_language_token(model, lang_code)
    
    prefix_tokens = get_prefix_tokens(processor.tokenizer, lang_code)
    prefix_pos_id = [[i, t] for i, t in enumerate(prefix_tokens)]

    model.generation_config.forced_decoder_ids = prefix_pos_id
    model.config.forced_decoder_ids = prefix_pos_id

    return model, processor, token_id


def get_prefix_tokens(tokenizer, language=None, task="transcribe", no_timestamps=True):

    # building prefix token sequence for testing
    ids = [tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    if language:
        ids.append(tokenizer.convert_tokens_to_ids(f"<|{language}|>"))
    if task:
        ids.append(tokenizer.convert_tokens_to_ids(f"<|{task}|>"))
    if no_timestamps:
        ids.append(tokenizer.convert_tokens_to_ids("<|notimestamps|>"))
    return ids


def main():

    # testing for custom language token addition

    model_name = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-tiny"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "test_custom_tokenizer"
    lang_code = sys.argv[3] if len(sys.argv) > 3 else "xx"

    print(f"Testing CustomWhisperProcessor with new language token")

    # load and setup
    print(f"\nLoading {model_name}...")
    model, processor, token_id = setup_model_with_custom_language(
        model_name, lang_code
    )
    
    # display info
    print(f"\nInitial setup:")
    print(f"    Vocab size: {len(processor.tokenizer)}")
    print(f"    Decoder embeddings: {model.model.decoder.embed_tokens.weight.shape}")
    print(f"    Token <|{lang_code}|> ID: {token_id}")
    
    # checking generation config
    print(f"    Lang_to_id entries: \n{len(model.generation_config.lang_to_id)}")
    if f"<|{lang_code}|>" in model.generation_config.lang_to_id:
        print(f"    <|{lang_code}|> -> {model.generation_config.lang_to_id[f'<|{lang_code}|>']}")
        print(f"    model.generation_config.forced_decoder_ids = {model.generation_config.forced_decoder_ids}")
        print(f"    model.config.forced_decoder_ids = {model.config.forced_decoder_ids}")
        
    
    # testing prefix generation
    print(f"\nTesting prefix generation:")
    prefix = get_prefix_tokens(processor.tokenizer, language=lang_code)
    print(f"    Prefix token IDs: {prefix}")
    prefix_tokens = [processor.tokenizer.convert_ids_to_tokens(id) for id in prefix]
    print(f"    Prefix tokens: {prefix_tokens}")
    
    # testing encoding/decoding
    print(f"\nTesting encoding/decoding:")
    test_text = "Hello world, this is a test."
    core_ids = processor.tokenizer.encode(test_text, add_special_tokens=False)
    full_ids = prefix + core_ids + [processor.tokenizer.eos_token_id]
    decoded = processor.tokenizer.decode(full_ids, skip_special_tokens=False)
    print(f"    Input: {test_text}")
    print(f"    Token IDs (first 10): {full_ids[:10]}...")
    print(f"    Decoded: {decoded[:80]}...")
    
    # saving
    print(f"\nSaving to {out_dir}...")
    Path(out_dir).mkdir(exist_ok=True)
    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    
    # saved files
    print(f"\nSaved files:")
    for file in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, file))
        print(f"  {file:30} ({size:,} bytes)")
    
    # reloading
    print(f"\nVerifying reload...")
    new_processor = CustomWhisperProcessor.from_pretrained(out_dir)
    new_model = WhisperForConditionalGeneration.from_pretrained(out_dir)
    
    # checking token
    try:
        reloaded_id = new_processor.tokenizer.convert_tokens_to_ids(f"<|{lang_code}|>")
        print(f"    Token <|{lang_code}|> found with ID: {reloaded_id}")
    except:
        print(f"    Token <|{lang_code}|> not found!")
    
    # checking custom languages
    if hasattr(new_processor.tokenizer, 'custom_languages'):
        print(f"    Custom languages: {new_processor.tokenizer.custom_languages}")
    
    # checking generation config
    if hasattr(new_model.generation_config, "lang_to_id"):
        print(f"    <|{lang_code}|> -> {model.generation_config.lang_to_id[f'<|{lang_code}|>']}")
        print(f"    model.generation_config.forced_decoder_ids = {model.generation_config.forced_decoder_ids}")
        print(f"    model.config.forced_decoder_ids = {model.config.forced_decoder_ids}")
    
    print(f"\nFinal stats:")
    print(f"    Vocab size: {len(new_processor.tokenizer)}")
    print(f"    Embeddings: {new_model.model.decoder.embed_tokens.weight.shape}")


if __name__ == "__main__":
    main()
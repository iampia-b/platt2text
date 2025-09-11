import os
import json
import torch
from transformers import (
    WhisperProcessor, 
    WhisperTokenizer,
    WhisperFeatureExtractor
)

# custom whisper tokenizer, handles custom (new) language tokens
class CustomWhisperTokenizer(WhisperTokenizer):
    
    def __init__(self, *args, custom_languages = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_languages = custom_languages or {}
        
    @property
    def prefix_tokens(self):
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        translate = self.convert_tokens_to_ids("<|translate|>")
        transcribe = self.convert_tokens_to_ids("<|transcribe|>")
        notimestamps = self.convert_tokens_to_ids("<|notimestamps|>")
        
        prefix = [bos_token_id]
        
        if self.language is not None:
            # check if language is new
            lang_code = self.custom_languages.get(self.language, self.language)
            lang_token = f"<|{lang_code}|>"
            
            lang_id = self.convert_tokens_to_ids(lang_token)
            if lang_id != self.unk_token_id: # check lang token is a known token
                prefix.append(lang_id)
        
        if self.task is not None:
            if self.task == "transcribe":
                prefix.append(transcribe)
            elif self.task == "translate":
                prefix.append(translate)
        
        if not self.predict_timestamps:
            prefix.append(notimestamps)
            
        return prefix

    def save_pretrained(self, save_directory, **kwargs):
        # saving the tokenizer
        outputs = super().save_pretrained(save_directory, **kwargs)
        
        # saving language mappings and tokenizer class info
        tokenizer_config_file = os.path.join(save_directory, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_file):
            with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config["tokenizer_class"] = "CustomWhisperTokenizer"
            config["custom_languages"] = self.custom_languages
            
            with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        # if tokenizer_config available
        tokenizer_config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        custom_languages = {}
        
        if os.path.exists(tokenizer_config_file):
            with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                custom_languages = config.get("custom_languages", {})
        
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        
        # set custom languages
        tokenizer.custom_languages = custom_languages
        
        return tokenizer


# custom whisper processor, uses CustomWhisperTokenizer
class CustomWhisperProcessor(WhisperProcessor):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):

        # check if model uses custom languages
        tokenizer_config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        use_custom_tokenizer = False
        
        if os.path.exists(tokenizer_config_file):
            with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if config.get("tokenizer_class") in ["CustomWhisperTokenizer"]:
                    use_custom_tokenizer = True
            
        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # load tokenizer
        if use_custom_tokenizer:
            tokenizer_class = CustomWhisperTokenizer
            tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            # standard tokenizer
            tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # save processor 
    def save_pretrained(self, save_directory, **kwargs):
        
        return super().save_pretrained(save_directory, **kwargs)
    
    def add_new_language_token(self, lang_code, lang_alias = None):

        lang_token = f"<|{lang_code}|>"
        
        # custom tokenizer if needed
        if not isinstance(self.tokenizer, (CustomWhisperTokenizer)):
            tokenizer_class = CustomWhisperTokenizer
            
            # new instance with same config
            custom_tokenizer = tokenizer_class.__new__(tokenizer_class)
            custom_tokenizer.__dict__.update(self.tokenizer.__dict__)
            custom_tokenizer.custom_languages = {}
            self.tokenizer = custom_tokenizer
        
        # adding the language token if needed
        if lang_token not in self.tokenizer.get_vocab():
            existing_specials = self.tokenizer.additional_special_tokens
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": existing_specials + [lang_token]
            })
        
        # register the alias
        if lang_alias:
            self.tokenizer.custom_languages[lang_alias] = lang_code
        

    # adding the custom language tag
    def add_new_language(self, model, lang_code, lang_alias = None, init_vector=None):

        # register language 
        self.add_new_language_token(lang_code, lang_alias)
        tokenizer = self.tokenizer
        new_vocab = len(tokenizer)

        # resizing
        model.resize_token_embeddings(new_vocab)
        
        # optional init for the NEW row
        tok = f"<|{lang_code}|>"
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if init_vector is not None and tok_id is not None and tok_id >= 0:
            vec = init_vector.to(model.device)
            with torch.no_grad():
                model.get_input_embeddings().weight[tok_id] = vec
                out = model.get_output_embeddings()
                out.weight[tok_id] = vec

        # update generation maps
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "lang_to_id"):
            model.generation_config.lang_to_id[lang_code] = tok_id
            if lang_alias:
                model.generation_config.lang_to_id[lang_alias] = tok_id

        print(f"New id: {tok_id} for {tok}")
        return tok_id


import evaluate
import torch

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from utils import ft_utils

class EWCTrainer(Seq2SeqTrainer):
    # custom trainer that adds EWC regularization to the loss
    
    def __init__(self, *args, fisher_dict=None, ewc_lambda=5000, ewc_lambda_decay=0.95, original_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_dict = fisher_dict
        self.ewc_lambda = ewc_lambda
        self.initial_ewc_lambda = ewc_lambda
        self.ewc_lambda = ewc_lambda
        self.ewc_lambda_decay = ewc_lambda_decay
        self.original_params = original_params
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # override compute_loss to add EWC regularization

        # get original loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # add EWC regularization
        if self.fisher_dict is not None and self.original_params is not None:
            ewc_loss = 0
            for name, param in model.named_parameters():
                # only apply EWC to encoder and early decoder layers
                if name in self.fisher_dict and name in self.original_params:
                    if "encoder" in name or "decoder.layers.0" in name or "decoder.layers.1" in name:
                        # only apply EWC to parameters that existed in original model
                        fisher = self.fisher_dict[name]
                        original_param = self.original_params[name]
                        
                        # handle size mismatch for embedding layer
                        if param.size() != original_param.size():
                            
                            if "embed_tokens" in name:
                                # only apply EWC to original token embeddings
                                min_size = min(param.size(0), original_param.size(0))
                                param_slice = param[:min_size]
                                fisher_slice = fisher[:min_size]
                                param_diff = param_slice - original_param[:min_size]
                                ewc_loss += (fisher_slice * param_diff.pow(2)).sum()
                        else:
                            # normal case: sizes match
                            param_diff = param - original_param
                            ewc_loss += (fisher * param_diff.pow(2)).sum()
            
            ewc_loss = 0.5 * self.ewc_lambda * ewc_loss
            loss = loss + ewc_loss
            
            # log EWC loss for monitoring
            if self.state.global_step % 100 == 0:
                self.ewc_lambda *= self.ewc_lambda_decay
                self.log({"ewc_lambda": self.ewc_lambda})
        
        return (loss, outputs) if return_outputs else loss

def build_ewc_trainer(model, processor, dataset, data_collator, fisher_dict, ewc_lambda, original_params, args):

    # metrics
    wer_metric = evaluate.load("wer")

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        gradient_checkpointing=True,
        seed=args.seed,
    )

    # trainer
    trainer = EWCTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: ft_utils.compute_metrics(pred, processor, wer_metric),
        processing_class=processor.tokenizer,
        fisher_dict=fisher_dict,
        ewc_lambda=ewc_lambda,
        original_params=original_params,
    )

    return trainer
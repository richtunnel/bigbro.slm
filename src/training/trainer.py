# src/training/trainer.py
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel
import mlflow
from src.models.peft_model import load_base_model, create_peft_model, load_tokenizer
from src.utils.config import ModelConfig, PeftConfig, TrainingConfig

def train(model_cfg: ModelConfig, peft_cfg: PeftConfig, train_cfg: TrainingConfig, dataset):
    tokenizer = load_tokenizer(model_cfg)
    model = load_base_model(model_cfg, quantize=True)
    model = create_peft_model(model, peft_cfg)

    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        evaluation_strategy=train_cfg.evaluation_strategy,
        bf16=train_cfg.bf16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        report_to=train_cfg.report_to,
        deepspeed=train_cfg.deepspeed,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
    )

    with mlflow.start_run():
        trainer.train()
        trainer.save_model(f"{train_cfg.output_dir}/final")
        # Optionally merge and push to registry
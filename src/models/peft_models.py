import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils.config import ModelConfig, PeftConfig

def load_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model(cfg: ModelConfig, quantize: bool = True):
    bnb_config = None
    if quantize:
        #This is what allows us to load a 7B–13B model on a single 24 GB GPU (or even smaller cards for true SLMs)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Load weights in 4-bit
            bnb_4bit_quant_type="nf4",            # NormalFloat4 quantization
            bnb_4bit_compute_dtype=torch.bfloat16,# Compute in bfloat16
            bnb_4bit_use_double_quant=True        # Extra compression
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=getattr(torch, cfg.torch_dtype),
        attn_implementation="flash_attention_2" if cfg.use_flash_attention else "eager",
    )
    return model
# PEFT Parameter-Efficient Fine-Tuning
def create_peft_model(model, peft_cfg: PeftConfig):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=peft_cfg.r,
        lora_alpha=peft_cfg.lora_alpha,
        lora_dropout=peft_cfg.lora_dropout,
        target_modules=peft_cfg.target_modules,
        bias=peft_cfg.bias,
        task_type=peft_cfg.task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
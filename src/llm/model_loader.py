import torch
import logging
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

def fix_loftq_config(adapter_path):
    """
    Fix compatibility issue with loftq_config in adapter_config.json.
    Some PEFT versions fail if loftq_config is null.
    """
    try:
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                config_data = json.load(f)
            
            if config_data.get('loftq_config') is None:
                logger.info(f"Fixing loftq_config in {adapter_config_path}")
                config_data.pop('loftq_config', None)
                
                # Save to a temp file first then rename to be safe
                temp_config_path = adapter_config_path + ".temp"
                with open(temp_config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                os.rename(temp_config_path, adapter_config_path)
    except Exception as e:
        logger.warning(f"Failed to fix loftq_config: {e}")

def load_model(model_path, device='cuda', dtype=torch.float16):
    """
    Load a standard Hugging Face model.
    """
    logger.info(f"Loading model from {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise e

def load_random_model(base_model_path, device='cuda', dtype=torch.float16):
    """
    Load a model with random weights but same architecture as base model.
    """
    logger.info(f"Creating random model based on config from {base_model_path}")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.to(dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    return model, tokenizer

def load_adapter_model(base_model_path, adapter_path, device='cuda', dtype=torch.float16):
    """
    Load a model with LoRA adapter.
    """
    fix_loftq_config(adapter_path)
    
    logger.info(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    logger.info(f"Loading adapter from {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        # We don't necessarily need to merge for inference, but for consistency with B-A logic we might want to.
        # For simple inference, PeftModel wraps the base model fine.
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise e
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    return model, tokenizer

def create_ba_model(base_model_path, instruct_model_path, target_model_path, is_adapter=False, device='cuda', dtype=torch.float16):
    """
    Create a B-A (Before-After) model using task vector arithmetic.
    Formula: New_Model = Target - Instruct + Base
    
    Args:
        base_model_path: Path to the base model (e.g., Qwen-Base)
        instruct_model_path: Path to the instruct model (e.g., Qwen-Instruct)
        target_model_path: Path to the target finetuned model or adapter
        is_adapter: If True, target_model_path is treated as an adapter on top of Instruct.
    """
    logger.info("Creating B-A model...")
    
    # 1. Get Target State Dict
    if is_adapter:
        logger.info(f"Loading Target Adapter: {target_model_path} on Instruct")
        fix_loftq_config(target_model_path)
        # Load Instruct first
        target_base = AutoModelForCausalLM.from_pretrained(instruct_model_path, torch_dtype=dtype, trust_remote_code=True)
        # Load Adapter
        target_peft = PeftModel.from_pretrained(target_base, target_model_path)
        # Merge
        target_model = target_peft.merge_and_unload()
        target_state_dict = target_model.state_dict()
        del target_base, target_peft, target_model
    else:
        logger.info(f"Loading Target model: {target_model_path}")
        target_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=dtype, trust_remote_code=True)
        target_state_dict = target_model.state_dict()
        del target_model

    # 2. Get Base and Instruct State Dicts
    # Optimization: We can load these sequentially to save RAM
    
    logger.info(f"Loading Base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype, trust_remote_code=True)
    base_state_dict = base_model.state_dict()
    del base_model
    
    logger.info(f"Loading Instruct model: {instruct_model_path}")
    instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_path, torch_dtype=dtype, trust_remote_code=True)
    instruct_state_dict = instruct_model.state_dict()
    del instruct_model
    
    new_state_dict = {}
    
    logger.info("Calculating task vector...")
    for key in target_state_dict:
        if key in instruct_state_dict and key in base_state_dict:
            # Check shapes
            if (base_state_dict[key].shape != instruct_state_dict[key].shape or 
                instruct_state_dict[key].shape != target_state_dict[key].shape):
                new_state_dict[key] = target_state_dict[key]
                continue
                
            # Perform arithmetic: Target - Instruct + Base
            # We do this on CPU to save VRAM
            t_target = target_state_dict[key].cpu()
            t_instruct = instruct_state_dict[key].cpu()
            t_base = base_state_dict[key].cpu()
            
            new_state_dict[key] = t_target - t_instruct + t_base
        else:
            new_state_dict[key] = target_state_dict[key]
            
    # Clean up
    del base_state_dict, instruct_state_dict, target_state_dict
    torch.cuda.empty_cache()
    
    # Create new model instance (using Instruct config as template)
    logger.info("Instantiating new model with B-A weights...")
    model = AutoModelForCausalLM.from_pretrained(
        instruct_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device
    )
    
    # Load new state dict
    # Use strict=False to allow for minor mismatches (e.g. if vocab size changed slightly)
    model.load_state_dict(new_state_dict, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(instruct_model_path, trust_remote_code=True)
    
    return model, tokenizer

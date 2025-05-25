from pathlib import Path
from typing import Dict, List, Optional, Union
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors"):
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.

    """
    
    state_dict = _fetch_state_dict(
        lora_path,
        weight_name,
        True,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None)

    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    
    adapter_name = weight_name.split(".")[0]
    
    # Check if adapter already exists and delete it if it does
    if hasattr(transformer, 'peft_config') and adapter_name in transformer.peft_config:
        print(f"Adapter '{adapter_name}' already exists. Removing it before loading again.")
        # Use delete_adapters (plural) instead of delete_adapter
        transformer.delete_adapters([adapter_name])
    
    # Load the adapter with the original name
    transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=adapter_name)
    print(f"LoRA weights '{adapter_name}' loaded successfully.")
    
    return transformer
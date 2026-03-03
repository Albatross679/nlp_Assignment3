"""Part 3 model: LLM initialization for prompting (Gemma / CodeGemma)."""

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GemmaForCausalLM,
    GemmaTokenizer,
    GemmaTokenizerFast,
)

MODEL_REGISTRY = {
    "gemma": "google/gemma-1.1-2b-it",
    "codegemma": "google/codegemma-7b-it",
}


def initialize_model_and_tokenizer(model_name="gemma", quantize=False, device="cuda"):
    """Load a Gemma-family model and its tokenizer.

    Args:
        model_name: "gemma" or "codegemma".
        quantize: Use 4-bit NF4 quantization (useful for codegemma-7b).
        device: Target device.

    Returns:
        (tokenizer, model)
    """
    model_id = MODEL_REGISTRY[model_name]

    if model_name == "gemma":
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        model = GemmaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device)
    elif model_name == "codegemma":
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                quantization_config=nf4_config,
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
            ).to(device)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    return tokenizer, model

# __init__.py

def load_model_wrapper(model_id: str):
    if "llama" in model_id.lower():
        from .wrapper import LlamaModelWrapper
        return LlamaModelWrapper("/content/models/" + model_id.replace("/", "_"), token="your_token_here")
    elif "gemma" in model_id.lower():
        from .gemma_wrapper import GemmaModelWrapper
        return GemmaModelWrapper("/content/models/" + model_id.replace("/", "_"), token="your_token_here")
    else:
        raise NotImplementedError(f"No wrapper for model: {model_id}")

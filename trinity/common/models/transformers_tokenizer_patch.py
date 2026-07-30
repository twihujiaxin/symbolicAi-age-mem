"""
Run at Python startup (e.g. via .pth) to patch PreTrainedTokenizerBase so that
tokenizers without all_special_tokens_extended (e.g. Qwen2Tokenizer in newer
transformers) get a compatible attribute. Required for vLLM EngineCore subprocess.
"""
try:
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def _all_special_tokens_extended(self):
            return getattr(self, "all_special_tokens", [])
        PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended
except Exception:
    pass

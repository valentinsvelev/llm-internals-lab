"""Reference PyTorch implementations of the six architectures in the explorer.

    tab   mechanism                         model                 module
    MHA   multi-head attention              GPT-2 small           attention.GroupedQueryAttention (G = H)
    MQA   multi-query attention             Gemma 2B              attention.GroupedQueryAttention (G = 1)
    GQA   grouped-query attention           Qwen2.5-0.5B          attention.GroupedQueryAttention
    MLA   multi-head latent attention       DeepSeek-V2-Lite      mla.MultiHeadLatentAttention
    GDN   Gated DeltaNet + gated attention  Qwen3-Next-80B-A3B    gated_deltanet.GatedDeltaNet
    DSA   sparse MLA, lightning indexer     DeepSeek-V3.2         mla.MultiHeadLatentAttention + LightningIndexer
    QSA   block-sparse attention            Qwen3.8-Flash-Next    qsa.QwenSparseAttention (one layer)

`model.DecoderLM` assembles any of the first six from its config (`config.CONFIGS`), `decoding` has the
eight decoding methods, and `hf` loads Hugging Face checkpoints. Every model is tested against the
transformers implementation of the same architecture: with the same weights, the logits agree.

Written for reading, not speed: one module per idea, float32 math, no fused kernels.
"""
from .config import CONFIGS, QSA_FLASH_NEXT, ModelConfig, MoE, QSAConfig, YaRN

__all__ = ['CONFIGS', 'QSA_FLASH_NEXT', 'ModelConfig', 'MoE', 'QSAConfig', 'YaRN', 'Cache', 'DecoderLM']


def __getattr__(name):
    """The models need PyTorch; the configs do not, so they load on first use."""
    if name in ('Cache', 'DecoderLM'):
        from . import model
        return getattr(model, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

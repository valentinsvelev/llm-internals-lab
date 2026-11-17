"""One configuration type for all six reference models, and the published configs of each.

Every number below is copied from the model's config.json on Hugging Face (links in `source`).
`tiny()` shrinks a config for tests while keeping its structure (same mechanisms, same layer
pattern, same ratios where they matter), so a small random model exercises every code path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class YaRN:
    """YaRN context extension of RoPE (Peng et al., 2023), as configured by DeepSeek."""
    factor: float
    original_max_positions: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0


@dataclass(frozen=True)
class MoE:
    """A mixture-of-experts feed-forward layer."""
    experts: int
    top_k: int
    expert_dim: int
    shared: int = 0                 # number of shared experts (DeepSeek fuses them into one MLP)
    shared_dim: int = 0             # hidden size of the shared MLP
    router: str = 'softmax'         # 'softmax' | 'sigmoid' (DeepSeek-V3: with load-balancing bias and groups)
    normalize: bool = False         # renormalize the top-k weights to sum to 1
    scaling: float = 1.0            # routed_scaling_factor
    groups: int = 1                 # n_group: experts are split into groups ...
    top_groups: int = 1             # ... and only the best topk_group groups may be chosen
    shared_gate: bool = False       # Qwen3-Next: sigmoid gate on the shared expert
    first_dense: int = 0            # the first k layers use a dense MLP instead


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    hidden: int
    layers: int
    # softmax attention ('gqa' covers MHA, MQA and GQA; 'mla' is DeepSeek's latent attention)
    attention: str
    heads: int
    kv_heads: int
    head_dim: int
    qkv_bias: bool = False
    out_bias: bool = False
    qk_norm: bool = False           # RMSNorm on every query and key head (Qwen3-Next)
    output_gate: bool = False       # sigmoid gate on the attention output (Qwen3-Next)
    # positions
    learned_positions: bool = False  # GPT-2 adds a learned position table instead of RoPE
    max_positions: int = 2048
    rope_theta: float = 10000.0
    rope_dim: int | None = None     # rotated dims per head; None = all (Qwen3-Next rotates 64 of 256)
    rope_interleaved: bool = False  # pairs (x0, x1), (x2, x3) … instead of halves (DeepSeek MLA)
    yarn: YaRN | None = None
    # MLA
    q_rank: int | None = None       # query latent (DeepSeek-V3/V3.2); None = project queries directly
    kv_rank: int = 0
    nope_dim: int = 0               # content part of each query/key head
    rope_head_dim: int = 0          # rotary part, shared by all heads on the key side
    v_dim: int = 0
    # DeepSeek Sparse Attention: the lightning indexer
    index_heads: int = 0
    index_dim: int = 0
    index_topk: int = 0
    # Gated DeltaNet (Qwen3-Next): which layers are linear attention
    full_attention_interval: int = 0  # every k-th layer is softmax attention, the others Gated DeltaNet
    lin_qk_heads: int = 0
    lin_v_heads: int = 0
    lin_head_dim: int = 0
    conv_kernel: int = 4
    # normalization and embeddings
    norm: str = 'rms'               # 'rms' | 'layer'
    norm_eps: float = 1e-6
    norm_offset: bool = False       # weight stored as w with scale 1 + w (Gemma, Qwen3-Next)
    embed_scale: bool = False       # Gemma multiplies embeddings by sqrt(hidden)
    tied: bool = False
    # feed-forward
    ffn: str = 'glu'                # 'mlp' (GPT-2) | 'glu' (gate ⊙ up)
    act: str = 'silu'               # 'silu' | 'gelu_tanh'
    intermediate: int = 0
    mlp_bias: bool = False
    moe: MoE | None = None
    source: str = ''

    # ---- derived ----
    def layer_type(self, i: int) -> str:
        """'attention' or 'gdn' for layer i (0-based)."""
        if self.full_attention_interval and (i + 1) % self.full_attention_interval:
            return 'gdn'
        return 'attention'

    def uses_moe(self, i: int) -> bool:
        return self.moe is not None and i >= self.moe.first_dense

    @property
    def qk_head_dim(self) -> int:
        return self.nope_dim + self.rope_head_dim if self.attention == 'mla' else self.head_dim

    @property
    def rotary_dim(self) -> int:
        if self.attention == 'mla':
            return self.rope_head_dim
        return self.rope_dim or self.head_dim

    def cache_values_per_token(self, i: int) -> int:
        """Numbers a layer stores per token for generation (0 for Gated DeltaNet: its state is fixed)."""
        if self.layer_type(i) == 'gdn':
            return 0
        if self.attention == 'mla':
            return self.kv_rank + self.rope_head_dim + self.index_dim
        return 2 * self.kv_heads * self.head_dim

    def tiny(self, **overrides) -> ModelConfig:
        """A small version with the same mechanisms, for tests and quick experiments."""
        small = dict(vocab_size=97, hidden=64, max_positions=256)
        if self.attention == 'gqa':   # keep the sharing pattern: MHA 4/4, MQA 4/1, GQA 4/2
            kv = 4 if self.kv_heads == self.heads else 1 if self.kv_heads == 1 else 2
            small.update(heads=4, kv_heads=kv, head_dim=16)
            if self.rope_dim:
                small.update(rope_dim=8)
        else:
            small.update(heads=4, kv_heads=4, head_dim=16, kv_rank=32, nope_dim=16, rope_head_dim=8, v_dim=16,
                         q_rank=24 if self.q_rank else None)
            if self.yarn:
                small.update(yarn=replace(self.yarn, original_max_positions=64))
        if self.index_heads:
            small.update(index_heads=4, index_dim=16, index_topk=5)
        if self.full_attention_interval:
            small.update(lin_qk_heads=2, lin_v_heads=4, lin_head_dim=16)
        layers = 4 if self.full_attention_interval else 3
        small.update(layers=layers, intermediate=128)
        if self.moe:
            small.update(moe=replace(self.moe, experts=8, top_k=2, expert_dim=32,
                                     shared_dim=32 * max(1, self.moe.shared) if self.moe.shared_dim else 0,
                                     groups=4 if self.moe.groups > 1 else 1, top_groups=2 if self.moe.groups > 1 else 1,
                                     first_dense=min(self.moe.first_dense, 1)))
        small.update(overrides)
        return replace(self, name=self.name + ' (tiny)', **small)


# ---- the six models shown in the explorer ----

GPT2_SMALL = ModelConfig(
    name='GPT-2 small', vocab_size=50257, hidden=768, layers=12,
    attention='gqa', heads=12, kv_heads=12, head_dim=64, qkv_bias=True, out_bias=True,
    learned_positions=True, max_positions=1024,
    norm='layer', norm_eps=1e-5, tied=True,
    ffn='mlp', act='gelu_tanh', intermediate=3072, mlp_bias=True,
    source='https://huggingface.co/openai-community/gpt2/blob/main/config.json')

GEMMA_2B = ModelConfig(
    name='Gemma 2B', vocab_size=256000, hidden=2048, layers=18,
    attention='gqa', heads=8, kv_heads=1, head_dim=256,
    max_positions=8192, rope_theta=10000.0,
    norm_offset=True, embed_scale=True, tied=True,
    ffn='glu', act='gelu_tanh', intermediate=16384,
    source='https://huggingface.co/google/gemma-2b/blob/main/config.json')

QWEN2_5_0_5B = ModelConfig(
    name='Qwen2.5-0.5B', vocab_size=151936, hidden=896, layers=24,
    attention='gqa', heads=14, kv_heads=2, head_dim=64, qkv_bias=True,
    max_positions=32768, rope_theta=1000000.0, tied=True,
    ffn='glu', act='silu', intermediate=4864,
    source='https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json')

DEEPSEEK_V2_LITE = ModelConfig(
    name='DeepSeek-V2-Lite', vocab_size=102400, hidden=2048, layers=27,
    attention='mla', heads=16, kv_heads=16, head_dim=192,
    kv_rank=512, nope_dim=128, rope_head_dim=64, v_dim=128,
    max_positions=163840, rope_theta=10000.0, rope_interleaved=True,
    yarn=YaRN(factor=40, original_max_positions=4096, mscale=0.707, mscale_all_dim=0.707),
    ffn='glu', act='silu', intermediate=10944,
    moe=MoE(experts=64, top_k=6, expert_dim=1408, shared=2, shared_dim=2 * 1408, first_dense=1),
    source='https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json')

QWEN3_NEXT_80B = ModelConfig(
    name='Qwen3-Next-80B-A3B', vocab_size=151936, hidden=2048, layers=48,
    attention='gqa', heads=16, kv_heads=2, head_dim=256, qk_norm=True, output_gate=True,
    max_positions=262144, rope_theta=10000000.0, rope_dim=64,
    full_attention_interval=4, lin_qk_heads=16, lin_v_heads=32, lin_head_dim=128, conv_kernel=4,
    norm_offset=True,
    ffn='glu', act='silu', intermediate=5120,
    moe=MoE(experts=512, top_k=10, expert_dim=512, shared=1, shared_dim=512, normalize=True, shared_gate=True),
    source='https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json')

DEEPSEEK_V3_2 = ModelConfig(
    name='DeepSeek-V3.2', vocab_size=129280, hidden=7168, layers=61,
    attention='mla', heads=128, kv_heads=128, head_dim=192,
    q_rank=1536, kv_rank=512, nope_dim=128, rope_head_dim=64, v_dim=128,
    index_heads=64, index_dim=128, index_topk=2048,
    max_positions=163840, rope_theta=10000.0, rope_interleaved=True,
    yarn=YaRN(factor=40, original_max_positions=4096, mscale=1.0, mscale_all_dim=1.0),
    ffn='glu', act='silu', intermediate=18432,
    moe=MoE(experts=256, top_k=8, expert_dim=2048, shared=1, shared_dim=2048, router='sigmoid',
            normalize=True, scaling=2.5, groups=8, top_groups=4, first_dense=3),
    source='https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json')

# Keyed like the explorer's tabs.
CONFIGS = {'mha': GPT2_SMALL, 'mqa': GEMMA_2B, 'gqa': QWEN2_5_0_5B,
           'mla': DEEPSEEK_V2_LITE, 'gdn': QWEN3_NEXT_80B, 'dsa': DEEPSEEK_V3_2}


@dataclass(frozen=True)
class QSAConfig:
    """Qwen Sparse Attention, one attention layer of Qwen3.8-Flash-Next (published text config)."""
    hidden: int = 2560
    heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rope_dim: int = 64              # partial_rotary_factor 0.25
    rope_theta: float = 10000000.0
    index_heads: int = 4
    index_dim: int = 128
    block: int = 4                  # indexer_compress_ratio
    budget: int = 2048              # indexer_budget, in tokens
    norm_eps: float = 1e-6

    @property
    def top_blocks(self) -> int:
        return self.budget // self.block


QSA_FLASH_NEXT = QSAConfig()

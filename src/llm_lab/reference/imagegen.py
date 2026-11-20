"""How two language models draw (the Image out tab of the Multimodal page).

Both write an image the way they write text: one discrete token at a time, each token an entry of a
VQ codebook that stands for a small square of pixels; a VQ decoder turns the grid of codes into pixels.

    janus  Janus-Pro-7B   384 × 384 → a 24 × 24 grid of codes (16 pixels each), codebook 16,384.
                          Its own embeddings, aligner and head for image codes; to read images it uses a
                          separate SigLIP encoder. Classifier-free guidance runs every step twice.
    emu3   Emu3-Gen       one vocabulary for text and codes: 720 × 720 → 90 × 90 codes (8 pixels each),
                          codebook 32,768, an end-of-line token after every row: 8,190 tokens

Numbers from deepseek-community/Janus-Pro-7B and BAAI/Emu3-Gen; tests/test_imagegen_models.py checks
them against transformers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .vision import Decoder


@dataclass(frozen=True)
class ImageGenModel:
    key: str
    name: str
    source: str
    decoder: Decoder
    codebook: int
    downsample: int             # pixels per code, along each side
    code_dim: int               # size of a codebook entry


JANUS = ImageGenModel(
    'janus', 'Janus-Pro-7B', 'deepseek-community/Janus-Pro-7B',
    Decoder('DeepSeek-LLM 7B', layers=30, hidden=4096, heads=32, kv_heads=32, head_dim=128, intermediate=11008,
            vocab=102400, max_positions=16384),
    codebook=16384, downsample=16, code_dim=8)
EMU3 = ImageGenModel(
    'emu3', 'Emu3-Gen', 'BAAI/Emu3-Gen',
    Decoder('Emu3 (Llama-style)', layers=32, hidden=4096, heads=32, kv_heads=8, head_dim=128, intermediate=14336,
            vocab=184622, max_positions=9216),
    codebook=32768, downsample=8, code_dim=4)

IMAGEGEN_MODELS = {m.key: m for m in (JANUS, EMU3)}

JANUS_SIDE = 384
JANUS_CFG = 5.0                  # transformers' default guidance scale for Janus
EMU3_AREA = 720 * 720            # image_area in Emu3-Gen's config
EMU3_RATIOS = ('1:1', '4:3', '16:9', '9:16')

# (image tokenizer / VQ, image-specific layers, language model incl. LM head), counted from transformers.
# Janus's image-specific layers: the SigLIP encoder + aligner for reading, the code embeddings + aligner +
# head for writing. Emu3 has none: codes are ordinary entries of its vocabulary.
PARAMETERS = {
    'janus': (71_883_403, 425_524_224, 6_910_365_696),
    'emu3': (270_740_287, 0, 8_492_011_520),
}


def emu3_grid(ratio: str = '1:1', area: int = EMU3_AREA, factor: int = 8) -> tuple[int, int]:
    """Emu3Processor.calculate_generate_size: the code grid (rows, columns) for an aspect ratio."""
    w, h = map(int, ratio.split(':'))
    scale = (area / (w * h)) ** 0.5
    return int(round(h * scale / factor)), int(round(w * scale / factor))


@dataclass(frozen=True)
class GenPlan:
    grid: tuple[int, int]        # codes (rows, columns)
    pixels: tuple[int, int]      # image (height, width)
    codes: int
    steps: int                   # tokens generated: one forward pass each
    sequences: int               # sequences run side by side (2 with classifier-free guidance)
    kv_values: int               # keys and values cached for the image tokens, all sequences


def gen_plan(model: str, ratio: str = '1:1') -> GenPlan:
    m = IMAGEGEN_MODELS[model]
    per_token = m.decoder.kv_values_per_token()
    if model == 'janus':
        g = JANUS_SIDE // m.downsample
        return GenPlan((g, g), (JANUS_SIDE, JANUS_SIDE), g * g, g * g, 2, 2 * g * g * per_token)
    if model == 'emu3':
        rows, cols = emu3_grid(ratio)
        steps = rows * (cols + 1)            # an end-of-line token after every row
        # Emu3-Gen's example also runs classifier-free guidance, with a negative prompt as the second sequence.
        return GenPlan((rows, cols), (rows * m.downsample, cols * m.downsample), rows * cols, steps, 2, 2 * steps * per_token)
    raise KeyError(model)


def cfg_logits(cond, uncond, scale: float = JANUS_CFG):
    """Classifier-free guidance: push the logits away from what the model writes without the prompt."""
    return uncond + scale * (cond - uncond)


def janus_generation_layers(code_dim: int = 8, hidden: int = 4096, codebook: int = 16384, depth: int = 2):
    """Janus's layers for writing images, with transformers' names: `generation_embeddings` (a code → 8 numbers),
    `generation_aligner` (8 → 4,096: fc1, GELU, hidden_layers[0]) feeding the language model, and
    `generation_head` (4,096 → 4,096 → 16,384: proj_out, GELU, vision_head) reading its output."""
    from torch import nn

    class Aligner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(code_dim, hidden)
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth - 1)])

        def forward(self, x):
            x = self.fc1(x)
            for layer in self.hidden_layers:
                x = layer(nn.functional.gelu(x))
            return x

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_out = nn.Linear(hidden, hidden)
            self.vision_head = nn.Linear(hidden, codebook)

        def forward(self, h):
            return self.vision_head(nn.functional.gelu(self.proj_out(h)))

    class JanusGeneration(nn.Module):
        def __init__(self):
            super().__init__()
            self.generation_embeddings = nn.Embedding(codebook, code_dim)
            self.generation_aligner = Aligner()
            self.generation_head = Head()

        def embed(self, codes):                       # the next input of the language model
            return self.generation_aligner(self.generation_embeddings(codes))

        def logits(self, h):                          # scores over the 16,384 codes
            return self.generation_head(h)

    return JanusGeneration()


def hf_config(model: str):
    import transformers as tf
    m = IMAGEGEN_MODELS[model]
    d = m.decoder
    if model == 'janus':
        return tf.JanusConfig(
            text_config=dict(model_type='llama', hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers,
                             num_attention_heads=d.heads, num_key_value_heads=d.kv_heads, vocab_size=d.vocab,
                             max_position_embeddings=d.max_positions, rms_norm_eps=1e-6),
            vision_config=dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, image_size=JANUS_SIDE, patch_size=16,
                               projection_dim=d.hidden),
            vq_config=dict(embed_dim=m.code_dim, num_embeddings=m.codebook, num_patches=JANUS_SIDE // m.downsample,
                           projection_dim=d.hidden, image_token_embed_dim=d.hidden),
            tie_word_embeddings=False)
    if model == 'emu3':
        return tf.Emu3Config(
            text_config=dict(hidden_size=d.hidden, intermediate_size=d.intermediate, num_hidden_layers=d.layers, num_attention_heads=d.heads,
                             num_key_value_heads=d.kv_heads, vocab_size=d.vocab, max_position_embeddings=d.max_positions),
            vq_config=dict(codebook_size=m.codebook, embed_dim=m.code_dim), vocabulary_map={})
    raise KeyError(model)

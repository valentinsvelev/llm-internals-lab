"""The vision-language reference (llm_lab.reference.vision) against transformers.

The image processors decide how many tokens an image costs; the connectors, the 3D positions, Gemma's
image mask and Llama's gated cross-attention decide what the language model reads. Each is compared
with its transformers counterpart, with random weights where there are weights.
"""
from __future__ import annotations

import random
import warnings

import pytest

torch = pytest.importorskip('torch')
tf = pytest.importorskip('transformers')

from llm_lab.reference import vision as V  # noqa: E402

warnings.filterwarnings('ignore')
tf.logging.set_verbosity_error()

SIZES = [(336, 336), (480, 640), (768, 1024), (1080, 1920), (300, 1200), (2000, 500), (100, 100), (3024, 4032)]


def _random_(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.5 if p.ndim == 1 else 0.1))
    return module


def _image(h, w):
    from PIL import Image
    return Image.new('RGB', (w, h), (120, 160, 200))


# ------------------------------------------------------------------------------------------- resizing

def test_resize_rules_match_transformers_on_many_sizes():
    from transformers.models.mllama.image_processing_pil_mllama import (
        get_image_size_fit_to_canvas,
        get_optimal_tiled_canvas,
    )
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize
    rng = random.Random(0)
    sizes = SIZES + [(rng.randint(20, 6000), rng.randint(20, 6000)) for _ in range(2000)]
    for h, w in sizes:
        if max(h, w) / min(h, w) > 200:
            continue
        assert V.smart_resize(h, w) == tuple(smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216)), (h, w)
        canvas = V.tiled_canvas(h, w)
        assert canvas == tuple(int(x) for x in get_optimal_tiled_canvas(h, w, 4, 560)), (h, w)
        assert V.fit_to_canvas(h, w, *canvas) == tuple(int(x) for x in get_image_size_fit_to_canvas(h, w, *canvas, 560)), (h, w)


@pytest.mark.parametrize('h, w', SIZES)
def test_image_plans_match_the_image_processors(h, w):
    pytest.importorskip('PIL')
    img = _image(h, w)
    llava = tf.CLIPImageProcessorPil(size={'shortest_edge': 336}, crop_size={'height': 336, 'width': 336}, do_center_crop=True)
    assert tuple(llava(img, return_tensors='pt')['pixel_values'].shape[-2:]) == V.image_plan('llava', h, w).resized

    qwen = tf.Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2,
                                       size={'shortest_edge': 65536, 'longest_edge': 16777216})
    out = qwen(img, return_tensors='pt')
    plan = V.image_plan('qwen', h, w)
    assert out['image_grid_thw'].tolist() == [[1, *plan.patches]]
    assert out['pixel_values'].shape[0] == plan.patches[0] * plan.patches[1] == 4 * plan.tokens

    llama = tf.MllamaImageProcessorPil(size={'height': 560, 'width': 560}, max_image_tiles=4)
    out = llama([[img]], return_tensors='pt')
    plan = V.image_plan('llama', h, w)
    assert out['num_tiles'] == [[plan.tiles]]
    assert tuple(out['pixel_values'].shape) == (1, 1, 4, 3, 560, 560)       # always 4 tiles, the unused ones empty

    gemma = tf.Gemma3ImageProcessorPil(size={'height': 896, 'width': 896})
    assert tuple(gemma(img, return_tensors='pt')['pixel_values'].shape[-2:]) == V.image_plan('gemma', h, w).resized


def test_published_token_counts():
    assert V.image_plan('llava', 900, 1600).tokens == 576 == (336 // 14) ** 2           # the class token is dropped
    assert V.image_plan('gemma', 900, 1600).tokens == 256 == (896 // 14 // 4) ** 2      # 64 × 64 patches, 4 × 4 pooled
    assert V.image_plan('qwen', 1024, 1024).tokens == 1024                              # one token per 32 × 32 pixels
    assert V.image_plan('llama', 1080, 1920).tiles == 4 and V.image_plan('llama', 480, 640).tiles == 2
    assert V.image_plan('llama', 480, 640).cross_positions == 4 * 1601


def test_llama_cross_attention_cache_covers_all_four_tiles():
    """The cross-attention layer caches keys for max_num_tiles × (patches + 1) positions, used tiles or not."""
    from transformers.models.mllama.configuration_mllama import MllamaTextConfig, MllamaVisionConfig
    vc = MllamaVisionConfig(hidden_size=32, num_hidden_layers=2, num_global_layers=1, attention_heads=2, intermediate_size=64,
                            image_size=28, patch_size=14, max_num_tiles=4, intermediate_layers_indices=[0, 1], vision_output_dim=96,
                            supported_aspect_ratios=[list(a) for a in V.tile_arrangements()])
    tc = MllamaTextConfig(hidden_size=64, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
                          cross_attention_layers=[3], vocab_size=300, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg = tf.MllamaConfig(vision_config=vc, text_config=tc, image_token_index=299)
    cfg._attn_implementation = 'eager'
    model = tf.MllamaForConditionalGeneration(cfg).eval()
    xmask = torch.zeros(1, 5, 1, 4, dtype=torch.long)
    xmask[0, 1:, 0, :2] = 1                                                             # an image of 2 tiles
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[1, 299, 5, 6, 7]]), pixel_values=torch.randn(1, 1, 4, 3, 28, 28),
                    aspect_ratio_ids=torch.tensor([[2]]), aspect_ratio_mask=torch.tensor([[[1, 1, 0, 0]]]),
                    cross_attention_mask=xmask, use_cache=True)
    lengths = [layer.keys.shape[2] for layer in out.past_key_values.layers]
    assert lengths == [5, 5, 5, 4 * ((28 // 14) ** 2 + 1), 5]


# ----------------------------------------------------------------------------------- whole-model sizes

@pytest.mark.parametrize('key, published', [('llava', 7.06e9), ('qwen', 8.77e9), ('llama', 10.67e9), ('gemma', 4.30e9)])
def test_full_size_parameter_counts(key, published):
    with torch.device('meta'):
        model = tf.AutoModelForImageTextToText.from_config(V.hf_config(key))
    count = lambda m: sum(p.numel() for p in m.parameters())   # noqa: E731
    total, inner = count(model), model.model
    if key == 'qwen':
        connector = count(inner.visual.merger) + sum(count(m) for m in inner.visual.deepstack_merger_list)
        vision = count(inner.visual) - connector
    else:
        vision = count(inner.vision_model if key == 'llama' else inner.vision_tower)
        connector = count(inner.multi_modal_projector)
    assert (vision, connector, total - vision - connector) == V.PARAMETERS[key]
    assert abs(total - published) / published < 0.005, total


# ------------------------------------------------------------------------------------------ connectors

def test_llava_projector_matches_transformers():
    from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
    cfg = tf.LlavaConfig(vision_config=tf.CLIPVisionConfig(hidden_size=24, num_attention_heads=4),
                         text_config=tf.LlamaConfig(hidden_size=40, num_attention_heads=4, num_key_value_heads=4))
    theirs = _random_(LlavaMultiModalProjector(cfg)).eval()
    ours = V.llava_projector(24, 40)
    ours.load_state_dict(theirs.state_dict())
    x = torch.randn(2, 9, 24)
    torch.testing.assert_close(ours(x), theirs(x))


@pytest.mark.parametrize('postshuffle', [False, True])
def test_qwen_merger_matches_transformers(postshuffle):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger
    cfg = Qwen3VLVisionConfig(hidden_size=24, out_hidden_size=40, spatial_merge_size=2)
    theirs = _random_(Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=postshuffle)).eval()
    ours = V.qwen_merger(24, 40, postshuffle_norm=postshuffle)
    ours.load_state_dict(theirs.state_dict())
    x = torch.randn(4 * 6, 24)                                    # 6 blocks of 2 × 2 patches
    torch.testing.assert_close(ours(x), theirs(x))
    assert ours(x).shape == (6, 40)


def test_gemma_projector_matches_transformers():
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector
    cfg = tf.Gemma3Config(vision_config=dict(hidden_size=24, num_attention_heads=4, image_size=112, patch_size=14),
                          text_config=dict(hidden_size=40, num_attention_heads=4, num_key_value_heads=2, head_dim=10),
                          mm_tokens_per_image=4)
    theirs = _random_(Gemma3MultiModalProjector(cfg)).eval()
    ours = V.gemma_projector(24, 40, patches_per_side=8, tokens=4)
    ours.load_state_dict(theirs.state_dict())
    x = torch.randn(2, 64, 24)                                    # 8 × 8 patches → 2 × 2 tokens
    torch.testing.assert_close(ours(x), theirs(x))


def test_gated_cross_attention_matches_transformers():
    from transformers.models.mllama.configuration_mllama import MllamaTextConfig
    from transformers.models.mllama.modeling_mllama import (
        MllamaCrossAttentionDecoderLayer,
        _prepare_cross_attention_mask,
    )
    tc = MllamaTextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
                          vocab_size=100, pad_token_id=0)
    tc._attn_implementation = 'eager'
    theirs = _random_(MllamaCrossAttentionDecoderLayer(tc, layer_idx=0)).eval()
    ours = V.gated_cross_attention_block(64, 4, 2, 16, 96)
    ours.load_state_dict(theirs.state_dict())
    tiles, per_tile, T = 4, 5, 6
    tile_mask = torch.zeros(1, T, 1, tiles, dtype=torch.long)
    tile_mask[0, 2:, 0, :2] = 1                                   # tokens 0–1 come before the image; it has 2 tiles
    h, image = torch.randn(1, T, 64), torch.randn(1, tiles * per_tile, 64)
    mask4d, rows = _prepare_cross_attention_mask(tile_mask, per_tile, torch.float32)
    with torch.no_grad():
        expected = theirs(h, image, mask4d, None, rows)
        got = ours(h, image, tile_mask[:, :, 0].repeat_interleave(per_tile, dim=-1).bool())
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
    fresh = V.gated_cross_attention_block(64, 4, 2, 16, 96)
    torch.testing.assert_close(fresh(h, image, torch.ones(1, T, 20, dtype=torch.bool)), h)   # gates at 0: the identity


# ----------------------------------------------------------------------------------- positions and masks

def test_mrope_positions_match_transformers():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    cfg = tf.Qwen3VLConfig(vision_config=dict(depth=1, hidden_size=16, num_heads=2, out_hidden_size=32, intermediate_size=32,
                                              deepstack_visual_indexes=[]),
                           text_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                            head_dim=16, intermediate_size=64, vocab_size=160000))
    model = Qwen3VLModel(cfg)
    grids = [(1, 6, 10), (1, 4, 4)]                               # patch grids: 3 × 5 and 2 × 2 tokens
    img = cfg.image_token_id
    ids, types = [], []
    for n_text, grid in ((5, grids[0]), (3, grids[1]), (4, None)):
        ids += [7] * n_text
        types += [0] * n_text
        if grid:
            n = grid[0] * grid[1] * grid[2] // 4
            ids += [img] * n
            types += [1] * n
    expected, _ = model.get_rope_index(torch.tensor([ids]), torch.tensor([types]), image_grid_thw=torch.tensor(grids))
    got = V.mrope_positions([('text', 5), ('image', grids[0]), ('text', 3), ('image', grids[1]), ('text', 4)])
    assert expected[:, 0].tolist() == got
    assert got[0][5:20] == [5] * 15 and got[1][5:10] == [5] * 5 and got[2][5:10] == [5, 6, 7, 8, 9]
    assert got[0][20] == 5 + max(3, 5)                            # text resumes after the image's longer side


def test_mrope_frequency_layout_matches_transformers():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    cfg = tf.Qwen3VLConfig().text_config
    cfg.rope_parameters = {'rope_type': 'default', 'rope_theta': 5e6, 'mrope_section': [24, 20, 20], 'mrope_interleaved': True}
    cfg.head_dim = 128
    rope = Qwen3VLTextRotaryEmbedding(cfg)
    marked = torch.stack([torch.full((1, 1, 64), float(a)) for a in range(3)])   # pair value = which axis it came from
    axes = rope.recomposition_frequencies(marked)[0, 0, :64]
    assert ''.join('THW'[int(a)] for a in axes) == V.mrope_pair_axes()
    assert V.mrope_pair_axes().count('T') == 24 and V.mrope_pair_axes().count('H') == 20


def test_gemma_image_mask_matches_transformers():
    from transformers.models.gemma3.modeling_gemma3 import get_block_sequence_ids_for_mask
    types = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    blocks = get_block_sequence_ids_for_mask(torch.tensor([types]))[0].tolist()
    mask = V.gemma_attention_mask([t == 1 for t in types])
    n = len(types)
    expected = [[j <= i or (blocks[i] >= 0 and blocks[i] == blocks[j]) for j in range(n)] for i in range(n)]
    assert mask == expected
    assert mask[2][4] and not mask[2][7] and not mask[5][6]       # inside an image: both ways; never into the future otherwise


def test_cache_cost_per_image():
    mib = {k: V.image_cache_values(k, 768, 1024) * 2 / 2 ** 20 for k in V.VLMS}
    assert mib == {'llava': 288.0, 'qwen': 108.0, 'llama': 6404 * 32 / 1024 + 0.125, 'gemma': 34.0}

"""llm_lab.reference.{audio, video, speech, imagegen} against transformers: token counts, frame and code
layouts, the adapters and generation layers with random weights, and parameter counts at full size."""
import math
import types

import numpy as np
import pytest

torch = pytest.importorskip('torch')
tf = pytest.importorskip('transformers')
tf.logging.set_verbosity_error()

from llm_lab.reference import audio as A  # noqa: E402
from llm_lab.reference import imagegen as G  # noqa: E402
from llm_lab.reference import speech as SP  # noqa: E402
from llm_lab.reference import video as V  # noqa: E402

torch.manual_seed(0)
count = lambda m: sum(p.numel() for p in m.parameters())   # noqa: E731
DURATIONS = [0.3, 1, 2.5, 5, 7.3, 29.99, 30, 31, 45, 90, 600]


# --------------------------------------------------------------------------------------------------- audio

def test_qwen2_audio_tokens_follow_the_feature_extractor():
    fe = tf.WhisperFeatureExtractor(feature_size=A.MELS)
    for sec in DURATIONS:
        x = np.zeros(round(sec * A.SAMPLE_RATE), dtype=np.float32)
        out = fe(x, sampling_rate=A.SAMPLE_RATE, return_attention_mask=True, padding='max_length', truncation=True, return_tensors='np')
        frames = int(out['attention_mask'].sum())
        plan = A.audio_plan('qwen2audio', sec)
        assert out['input_features'].shape[-1] == A.CHUNK_FRAMES
        assert plan.mel_frames == frames, sec
        assert plan.tokens == (((frames - 1) // 2 + 1) - 2) // 2 + 1    # Qwen2AudioProcessor
        assert plan.dropped_seconds == pytest.approx(max(0.0, sec - 30), abs=1e-4)


def test_voxtral_pads_to_whole_30_second_chunks():
    fe = tf.WhisperFeatureExtractor(feature_size=A.MELS)
    for sec in DURATIONS:
        x = np.zeros(round(sec * A.SAMPLE_RATE), dtype=np.float32)
        out = fe(x, sampling_rate=A.SAMPLE_RATE, padding=True, truncation=False, pad_to_multiple_of=A.CHUNK_SAMPLES, return_tensors='np')
        chunks = out['input_features'].shape[-1] // A.CHUNK_FRAMES                  # as VoxtralProcessor splits them
        plan = A.audio_plan('voxtral', sec)
        assert (plan.chunks, plan.tokens) == (chunks, 375 * chunks), sec


def _tiny_audio(key):
    cfg = A.hf_config(key)
    enc = cfg.audio_config
    enc.num_hidden_layers = enc.encoder_layers = 1
    for name, val in (('d_model', 16), ('hidden_size', 16), ('encoder_ffn_dim', 32), ('intermediate_size', 64),
                      ('encoder_attention_heads', 2), ('num_attention_heads', 2)):
        if hasattr(enc, name):
            setattr(enc, name, val)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers, t.num_attention_heads, t.num_key_value_heads = 32, 64, 1, 4, 2
    t.head_dim, t.vocab_size = 8, 200
    if key == 'voxtral':
        enc.intermediate_size = 4 * 16                       # the projector reads 4 stacked encoder frames
    cls = tf.Qwen2AudioForConditionalGeneration if key == 'qwen2audio' else tf.VoxtralForConditionalGeneration
    return cls(cfg).eval()


def test_qwen2_audio_adapter_matches_transformers():
    model = _tiny_audio('qwen2audio')
    tower = model.model.audio_tower
    fe = tf.WhisperFeatureExtractor(feature_size=A.MELS)
    for sec in (2.5, 30):
        out = fe(np.random.randn(round(sec * A.SAMPLE_RATE)).astype(np.float32), sampling_rate=A.SAMPLE_RATE,
                 return_attention_mask=True, padding='max_length', truncation=True, return_tensors='pt')
        captured = {}
        hook = tower.layers[-1].register_forward_hook(lambda m, i, o, c=captured: c.update(h=o[0] if isinstance(o, tuple) else o))
        with torch.no_grad():
            hf = model.model.multi_modal_projector(tower(out['input_features']).last_hidden_state)
        hook.remove()
        ours = A.qwen2_audio_adapter(16, 32)
        ours.layer_norm.load_state_dict(tower.layer_norm.state_dict())
        ours.linear.load_state_dict(model.model.multi_modal_projector.linear.state_dict())
        with torch.no_grad():
            mine = ours(captured['h'])
        tokens = A.audio_plan('qwen2audio', sec).tokens
        assert int(tower._get_feat_extract_output_lengths(out['attention_mask'].sum(-1))[1]) == tokens    # rows the model keeps
        assert hf.shape[1] == A.ENCODER_POSITIONS // 2
        torch.testing.assert_close(mine, hf, atol=1e-5, rtol=1e-5)


def test_voxtral_adapter_matches_transformers():
    model = _tiny_audio('voxtral')
    h = torch.randn(2, A.ENCODER_POSITIONS, 16)                       # two 30-s chunks out of the encoder
    ours = A.voxtral_adapter(16, 32)
    ours.load_state_dict(model.model.multi_modal_projector.state_dict())
    with torch.no_grad():
        hf = model.model.multi_modal_projector(h.reshape(-1, model.config.audio_config.intermediate_size))
        mine = ours(h)
    assert mine.shape == (2 * A.VOXTRAL_TOKENS_PER_CHUNK, 32)
    torch.testing.assert_close(mine, hf)


@pytest.mark.parametrize('key, published', [('qwen2audio', 8.40e9), ('voxtral', 4.68e9)])
def test_audio_parameter_counts(key, published):
    cls = tf.Qwen2AudioForConditionalGeneration if key == 'qwen2audio' else tf.VoxtralForConditionalGeneration
    with torch.device('meta'):
        m = cls(A.hf_config(key))
    enc, conn = count(m.model.audio_tower), count(m.model.multi_modal_projector)
    assert (enc, conn, count(m) - enc - conn) == A.PARAMETERS[key]
    assert abs(count(m) - published) / published < 0.005


# --------------------------------------------------------------------------------------------------- video

@pytest.fixture(scope='module')
def qwen_video():
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    return Qwen3VLVideoProcessor()


@pytest.mark.parametrize('video_fps', [30, 25, 24, 60])
def test_qwen3vl_samples_two_frames_per_second(qwen_video, video_fps):
    from transformers.video_utils import VideoMetadata
    for sec in [1, 2.2, 3, 10, 61.7, 300, 1200]:
        total = round(sec * video_fps)
        md = VideoMetadata(total_num_frames=total, fps=video_fps, duration=total / video_fps)
        assert V.sample_frames(sec, video_fps) == qwen_video.sample_frames(md).tolist(), (sec, video_fps)


def test_qwen3vl_video_resize_and_timestamps(qwen_video):
    from transformers.image_utils import SizeDict
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.video_processing_utils import BaseVideoProcessor
    seen = {}
    real = BaseVideoProcessor.resize
    BaseVideoProcessor.resize = lambda self, image, size, **kw: seen.update(size=(size.height, size.width))
    try:
        for sec in [1, 3, 10, 60, 300, 1200]:
            for h, w in [(720, 1280), (1080, 1920), (480, 640), (1280, 720)]:
                plan = V.video_plan('qwen3vl', sec, h, w)
                qwen_video.resize(torch.zeros(1, 1, 3, h, w).expand(1, plan.frames, 3, h, w), size=SizeDict(shortest_edge=V.MIN_PIXELS, longest_edge=V.MAX_PIXELS),
                                  resample=None, factor=V.FACTOR, temporal_factor=V.TEMPORAL_PATCH, cap_pixels_per_frame=True)
                assert plan.frame_size == seen['size'], (sec, h, w)
                assert plan.tokens_per_group == (seen['size'][0] // 32) * (seen['size'][1] // 32)
                stamps = Qwen3VLProcessor._calculate_timestamps(None, V.sample_frames(sec), 30.0, V.TEMPORAL_PATCH)
                assert plan.timestamps == pytest.approx(stamps) and plan.groups == len(stamps)
    finally:
        BaseVideoProcessor.resize = real


def test_onevision_video_pooling_matches_transformers():
    cfg = V.hf_config('llava_ov')
    cfg.vision_config.num_hidden_layers, cfg.vision_config.hidden_size, cfg.vision_config.intermediate_size = 1, 16, 32
    cfg.vision_config.num_attention_heads = 2
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers, t.num_attention_heads, t.num_key_value_heads, t.vocab_size = 32, 64, 1, 4, 2, 200
    model = tf.LlavaOnevisionForConditionalGeneration(cfg).eval()
    feats = torch.randn(5, 27 * 27, 32)
    with torch.no_grad():
        torch.testing.assert_close(V.onevision_pool(feats), model.model.apply_pooling(feats))
    plan = V.video_plan('llava_ov', 600)
    assert plan.tokens == V.OV_FRAMES * math.ceil(27 / 2) ** 2 + 1 == 6273           # LlavaOnevisionProcessor


def test_video_parameter_counts():
    with torch.device('meta'):
        m = tf.LlavaOnevisionForConditionalGeneration(V.hf_config('llava_ov'))
    enc, conn = count(m.model.vision_tower), count(m.model.multi_modal_projector)
    assert (enc, conn, count(m) - enc - conn) == V.PARAMETERS['llava_ov']
    assert abs(count(m) - 8.03e9) / 8.03e9 < 0.005
    from llm_lab.reference import vision
    assert V.PARAMETERS['qwen3vl'] == vision.PARAMETERS['qwen']


# -------------------------------------------------------------------------------------------- speech out

def test_qwen3_omni_code2wav_makes_1920_samples_per_frame():
    cfg = SP.hf_config('qwen3omni').code2wav_config
    assert math.prod(cfg.upsample_rates + cfg.upsampling_ratios) == SP.SAMPLES_PER_FRAME == math.prod(SP.CODE2WAV_UPSAMPLE)
    assert SP.OUT_SAMPLE_RATE / SP.SAMPLES_PER_FRAME == SP.FRAME_RATE
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeCode2WavConfig,
    )
    tiny = Qwen3OmniMoeCode2WavConfig(codebook_size=32, codebook_dim=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                         num_key_value_heads=2, intermediate_size=32, decoder_dim=16, upsample_rates=list(cfg.upsample_rates),
                                         upsampling_ratios=list(cfg.upsampling_ratios), num_quantizers=SP.QWEN3_OMNI.codebooks,
                                         sliding_window=8)
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav
    model = Qwen3OmniMoeCode2Wav(tiny).eval()
    with torch.no_grad():
        short, long = (model(torch.randint(0, 32, (1, SP.QWEN3_OMNI.codebooks, t))).shape[-1] for t in (7, 20))
    assert long - short == 13 * SP.SAMPLES_PER_FRAME                  # 1,920 samples per frame (the edges trim 555 in all)
    assert SP.speech_plan('qwen3omni', 20 / SP.FRAME_RATE).samples == 20 * SP.SAMPLES_PER_FRAME


def test_mimi_frames_and_moshi_streams():
    mimi = tf.MimiModel(tf.MimiConfig()).eval()
    with torch.no_grad():
        codes = mimi.encode(torch.zeros(1, 1, 2 * SP.OUT_SAMPLE_RATE), num_quantizers=SP.MOSHI.codebooks).audio_codes
    assert tuple(codes.shape) == (1, SP.MOSHI.codebooks, SP.speech_plan('moshi', 2).frames)
    cfg = SP.hf_config('moshi')
    assert (cfg.num_codebooks, cfg.audio_encoder_config.codebook_size, cfg.depth_decoder_config.num_codebooks) == (8, 2048, 8)
    assert cfg.depth_decoder_config.max_position_embeddings == 1 + SP.MOSHI.codebooks       # its text token, then 8 codes
    assert SP.MOSHI_STREAMS == 17


def test_moshi_delays_match_transformers():
    cfg = SP.hf_config('moshi')
    cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.ffn_dim = 32, 1, 4, 4, 8, 64
    d = cfg.depth_decoder_config
    d.hidden_size, d.input_size, d.num_hidden_layers, d.num_attention_heads, d.num_key_value_heads, d.head_dim, d.ffn_dim = 32, 32, 1, 4, 4, 8, 64
    model = tf.MoshiForConditionalGeneration(cfg)
    codes = torch.arange(1, 1 + 8 * 5).reshape(1, 8, 5)
    shifted, _ = model.build_delay_pattern_mask(codes, bos_token_id=-1, pad_token_id=-2, max_length=6)
    ours = SP.delayed_grid(codes[0].tolist(), SP.moshi_delays())
    for row, delay in zip(range(8), SP.moshi_delays()):
        for step in range(5):
            if ours[row][step] is not None:
                assert shifted[0, row, step].item() == ours[row][step]
            else:
                assert shifted[0, row, step].item() in (-1, -2) and delay == 1


def test_qwen3_omni_audio_tokens_and_interleaving():
    from transformers.models.qwen3_omni_moe import processing_qwen3_omni_moe as P
    for frames in list(range(1, 400)) + [999, 1000, 1001, 3000, 6000]:
        assert SP.omni_audio_tokens(frames) == int(P._get_feat_extract_output_lengths(np.array(frames))), frames
    fake = types.SimpleNamespace(video_processor=types.SimpleNamespace(merge_size=2), audio_bos_token='<', audio_eos_token='>',
                                 video_token='v', audio_token='a')
    for grids, h, w, seconds in [(3, 4, 6, 3), (5, 2, 2, 5), (10, 8, 8, 9.7)]:
        audio = SP.omni_audio_tokens(round(seconds * 100))
        out = P.Qwen3OmniMoeProcessor.replace_video_with_audio_token(
            fake, {'video_grid_thw': [np.array([grids, h, w])], 'video_second_per_grid': [SP.OMNI_SECONDS_PER_GRID]}, 0,
            {'num_audio_tokens': [audio]}, 0, position_id_per_seconds=SP.POSITIONS_PER_SECOND)
        assert out == '<' + SP.omni_interleave(grids, h * w // 4, audio) + '>'


@pytest.mark.parametrize('key', ['qwen3omni', 'moshi'])
def test_speech_parameter_counts(key):
    with torch.device('meta'):
        m = (tf.Qwen3OmniMoeForConditionalGeneration if key == 'qwen3omni' else tf.MoshiForConditionalGeneration)(SP.hf_config(key))
    if key == 'qwen3omni':
        parts = (count(m.thinker), count(m.talker), count(m.code2wav))
        inner = m.thinker.model
        assert (count(inner.audio_tower), count(inner.visual), count(inner.language_model) + count(m.thinker.lm_head)) == tuple(
            SP.QWEN3_OMNI_THINKER.values())
        published = 35.3e9
    else:
        mimi, depth = count(m.audio_encoder), count(m.depth_decoder)
        parts = (mimi, count(m) - mimi - depth, depth)
        published = 7.69e9 + mimi                                     # the model card's count leaves out Mimi
    assert parts == SP.PARAMETERS[key]
    assert abs(count(m) - published) / published < 0.005


# -------------------------------------------------------------------------------------------- image out

def test_janus_generation_layers_match_transformers():
    cfg = G.hf_config('janus')
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers, t.num_attention_heads, t.num_key_value_heads, t.vocab_size = 32, 64, 1, 4, 4, 300
    t.head_dim = 8
    v = cfg.vision_config
    v.hidden_size, v.num_hidden_layers, v.num_attention_heads, v.projection_dim = 16, 1, 2, 32
    q = cfg.vq_config
    q.projection_dim, q.image_token_embed_dim, q.num_embeddings = 32, 32, 64
    model = tf.JanusForConditionalGeneration(cfg).eval()
    ours = G.janus_generation_layers(code_dim=8, hidden=32, codebook=64)
    for name in ('generation_embeddings', 'generation_aligner', 'generation_head'):
        getattr(ours, name).load_state_dict(getattr(model.model, name).state_dict())
    codes = torch.randint(0, 64, (2, 7))
    h = torch.randn(2, 7, 32)
    with torch.no_grad():
        torch.testing.assert_close(ours.embed(codes), model.prepare_embeddings_for_image_generation(codes))
        torch.testing.assert_close(ours.logits(h), model.model.generation_head(h))
    assert G.gen_plan('janus').codes == cfg.vq_config.num_patches ** 2 == 576


def test_classifier_free_guidance_matches_transformers():
    cond, uncond = torch.randn(1, 50), torch.randn(1, 50)
    proc = tf.ClassifierFreeGuidanceLogitsProcessor(G.JANUS_CFG)
    hf = proc(torch.zeros(1, 1, dtype=torch.long), torch.cat([cond, uncond]))
    torch.testing.assert_close(G.cfg_logits(cond, uncond), hf)


def test_emu3_grid_and_tokens_follow_its_processor():
    from transformers.models.emu3.processing_emu3 import Emu3Processor
    for ratio in G.EMU3_RATIOS + ('3:2', '2:1'):
        rows, cols = G.emu3_grid(ratio)
        assert (rows, cols) == Emu3Processor.calculate_generate_size(None, ratio, G.EMU3_AREA, 8)
        fake = types.SimpleNamespace(downsample_ratio=8, image_start_token='[', fake_token_around_image='|', image_token='x',
                                     eof_token='$', image_end_token=']')
        text = Emu3Processor.replace_image_token(fake, {'image_sizes': [[rows * 8, cols * 8]]}, 0)
        if ratio in G.EMU3_RATIOS:
            assert text.count('x') == G.gen_plan('emu3', ratio).steps == rows * (cols + 1)
    assert G.gen_plan('emu3').grid == (90, 90) and G.gen_plan('emu3').steps == 8190


@pytest.mark.parametrize('key, published', [('janus', 7.42e9), ('emu3', 8.49e9)])
def test_imagegen_parameter_counts(key, published):
    with torch.device('meta'):
        m = (tf.JanusForConditionalGeneration if key == 'janus' else tf.Emu3ForConditionalGeneration)(G.hf_config(key))
    vq = count(m.model.vqmodel)
    img = sum(count(getattr(m.model, k)) for k in ('vision_model', 'aligner', 'generation_embeddings', 'generation_aligner',
                                                    'generation_head')) if key == 'janus' else 0
    assert (vq, img, count(m) - vq - img) == G.PARAMETERS[key]
    language = count(m) - vq if key == 'emu3' else count(m)            # Emu3's published size leaves out its tokenizer
    assert abs(language - published) / published < 0.005

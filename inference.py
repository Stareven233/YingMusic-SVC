'''
YingMusic-SVC inference -- swapped two-stage vocoder pipeline (training-free):

    YingMusic-SVC DiT -> predicted Mel
      -> [stage 1] Pupu-Vocoder            -> intermediate waveform (in-memory)
      -> re-extract Mel (PC-NSF params) + explicit F0 from the source audio
         (RMVPE, pitch-controllable via --pitch-shift / --f0-scale)
      -> [stage 2] PC-NSF-HiFiGAN          -> final.flac

The original BigGAN-family (bigvgan) vocoding path has been removed; the
predicted Mel is no longer fed to the built-in vocoder directly.

最终输出为 FLAC，文件名自动附带元数据后缀（项目名、ckpt、变调等 CLI
参数信息），由 gen_output_suffix() 生成。

Quick start refs:
https://zread.ai/GiantAILab/YingMusic-SVC/2-quick-start
https://zread.ai/GiantAILab/YingMusic-SVC/3-model-download-and-setup

uv venv --python=3.10
./.venv/Scripts/python.exe -c 'from modelscope import snapshot_download; snapshot_download('giantailab/YingMusic-SVC', local_dir='./checkpoints')'
uv pip install torch~=2.4.0 torchaudio~=2.4.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r ./requirements.txt

./.venv/Scripts/python.exe inference.py --source vocal.wav --target timbre.wav
uv run ./inference.py --source 'D:/Document/ai-sings/テオ/【翻唱】将手（テオ） ／ Omoi【Kotone(天神子兔音)cover】_Vocals_vocals_noreverb.flac' --target 'D:/Code/projects/DDSP-SVC/data/megumin/train/audio/ちいさな冒険者 めぐみん Version - 高橋李依_vocals_noreverb_0007.wav'
'''

# ---------------------------------------------------------------------------
# Dependency guard: fail early with an actionable message instead of a raw
# traceback when the environment is not fully provisioned.
# ---------------------------------------------------------------------------
_CORE_DEPS = ['numpy', 'torch', 'torchaudio', 'librosa', 'yaml', 'soundfile']


def _check_core_deps():
    import importlib.util
    missing = [mod for mod in _CORE_DEPS if importlib.util.find_spec(mod) is None]
    if missing:
        print('=' * 72)
        print('MISSING PYTHON DEPENDENCIES:', ', '.join(missing))
        print('Please install them yourself (no automatic downloads), e.g.:')
        print(f'    uv pip install {" ".join(missing)}')
        print('Plus the new-vocoder extras: uv pip install julius safetensors')
        print('=' * 72)
        raise SystemExit(2)


_check_core_deps()

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm

warnings.simplefilter('ignore')

import librosa

from hf_utils import load_custom_model_from_hf
from mm4 import preprocess_voice_conversion
from modules.audio import mel_spectrogram
from modules.commons import *
from pc_nsf_hifigan import load_pc_nsf_hifigan
from pupu_vocoder import load_pupu_vocoder
from Remix.auger import echo_then_reverb_save

# ---------------------------------------------------------------------------
# 本地预训练资源（见项目说明）。按项目约定不做任何自动下载，
# 缺失项由 preflight_check() 逐条报告。
# ---------------------------------------------------------------------------
PROJECT_NAME = 'YingMusic'
DEFAULT_DIFFUSION_STEPS = 30  # 与下方 --diffusion-steps 默认值保持一致

# 声码器计算精度：fp32 保持原始精度；fp16/bf16 通过 autocast 把激活值降到半精度
# （权重仅 ~50MB，显存大头是随音频长度线性增长的激活值，半精度可再省约一半）
VOCODER_DTYPES = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}

# 从 ckpt 文件名提取训练步数：model-step=10000.ckpt -> 10000
CKPT_STEP_PATTERN = re.compile(r'step[=_-](\d+)', re.IGNORECASE)

LOCAL_RMVPE_PATH = Path('./pretrain/rmvpe/model.pt')
PUPU_VOCODER_DIR = Path('./pretrain/vocoder/Pupu-Vocoder/experiments/pupuvocoder/checkpoint/epoch-0051_step-2553605_loss-62.135194')
PC_NSF_HIFIGAN_DIR = Path('./pretrain/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02')


def gen_output_suffix(args: argparse.Namespace) -> str:
    '''由 CLI 参数拼装输出文件名的元数据后缀，形如：

        YingMusic@10ks_+2st_1.2f0_50step
            │        │   │    │     └─ 扩散步数（非默认时才写入）
            │        │   │    └─ 显式 F0 缩放（非 1.0 时才写入）
            │        │   └─ 变调（semitones，恒写入）
            │        │
            │        └─ ckpt 训练步数（文件名可解析出 step 时写入）
            └─ 项目名

    与默认值相同的参数不写入，避免文件名冗长。
    '''
    *_, ckpt_name = Path(args.checkpoint).parts       # .../YingMusic-SVC-full.pt
    ckpt_tag = Path(ckpt_name).stem.removeprefix(f'{PROJECT_NAME}-') or 'base'
    parts = [f'{PROJECT_NAME}@{ckpt_tag}']

    if (m := CKPT_STEP_PATTERN.search(ckpt_name)):
        parts.append(f'{int(m.group(1)) / 1000:g}ks')  # 10000 -> '10ks'
    parts.append(f'{args.pitch_shift:+g}key')           # 变调恒写入，如 '+2key'/'-1.5key'
    if args.f0_scale != 1.0:
        parts.append(f'{args.f0_scale:g}f0')
    if args.semi_tone_shift is not None:
        parts.append(f'auto{args.semi_tone_shift:+g}st')
    if args.length_adjust != 1.0:
        parts.append(f'len{args.length_adjust:g}')
    if args.diffusion_steps != DEFAULT_DIFFUSION_STEPS:
        parts.append(f'{args.diffusion_steps}step')
    return '_'.join(parts)


def preflight_check(args):
    '''Verify local weights and python dependencies; report what is missing.

    Per project policy nothing is downloaded automatically -- missing items
    are listed so the user can provision them manually.'''
    import importlib.util

    ok = True

    print('-' * 72)
    print('[preflight] checking python dependencies ...')
    # (numpy/torch/librosa/yaml/soundfile already guarded at import time.)
    required_deps = {
        'scipy': 'scipy', 'munch': 'munch', 'einops': 'einops',
        'transformers': 'transformers',
        'julius': 'julius', 'beartype': 'beartype',
        'rotary_embedding_torch': 'rotary_embedding_torch',
        'ml_collections': 'ml_collections', 'loralib': 'loralib',
        'accelerate': 'accelerate',
    }
    missing_deps = [d for d in required_deps if importlib.util.find_spec(d) is None]
    if missing_deps:
        ok = False
        print(f'  [MISSING] {", ".join(missing_deps)}')
        print(f'            install manually, e.g.: uv pip install {" ".join(missing_deps)}')
    else:
        print('  [OK] all required packages found')

    print('[preflight] checking model weights ...')
    weight_checks = [
        ('SVC checkpoint (YingMusic-SVC)', Path(args.checkpoint), True),
        ('SVC config', Path(args.config), True),
        ('rmvpe (local)', LOCAL_RMVPE_PATH, False),
        ('Pupu-Vocoder generator', PUPU_VOCODER_DIR / 'model.safetensors', True),
        ('PC-NSF-HiFiGAN config', PC_NSF_HIFIGAN_DIR / 'config.json', True),
        ('PC-NSF-HiFiGAN ckpt', PC_NSF_HIFIGAN_DIR / 'model.ckpt', False),
    ]
    for label, path, required in weight_checks:
        if path.is_file():
            print(f'  [OK] {label}: {path}')
        elif required:
            ok = False
            print(f'  [MISSING] {label}: {path}')
        else:
            print(f'  [WARN] {label}: {path} not found')

    print('[preflight] NOTE: the following are fetched through HuggingFace on '
          'first run unless already cached:')
    print('           - openai/whisper-small            (speech tokenizer)')
    print('           - funasr/campplus                 (style encoder)')
    if not LOCAL_RMVPE_PATH.is_file():
        print('           - lj1995/VoiceConversionWebUI rmvpe.pt (local copy missing)')
    print('           Set HF_ENDPOINT=https://hf-mirror.com if needed.')
    print('-' * 72)

    if not ok:
        print('Preflight FAILED. Provision the items above yourself '
              '(no automatic downloads), then re-run.')
        raise SystemExit(2)


def extract_explicit_f0(f0_fn, waves_16k, thred=0.03):
    '''Extract a continuous explicit F0 trajectory with RMVPE.

    Reuses the RMVPE model already loaded for SVC conditioning
    (modules/rmvpe.py): it runs on 16 kHz audio and yields one F0 value every
    10 ms (hop 160); unvoiced frames are marked as 0 and are linearly
    interpolated here so the NSF excitation stays smooth.
    '''
    f0 = f0_fn(waves_16k, thred=thred)
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)

    voiced = f0 > 0
    if voiced.any():
        idx = np.arange(f0.size)
        f0 = np.interp(idx, idx[voiced], f0[voiced])
    else:
        # fully unvoiced input: zero F0 makes the NSF source silent
        f0 = np.zeros_like(f0)

    return f0.astype(np.float32)


def resize_f0_to_frames(f0, n_frames):
    '''Linearly resample an F0 curve onto exactly `n_frames` points.'''
    if f0.size == n_frames:
        return f0
    x_old = np.linspace(0.0, 1.0, f0.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, n_frames, endpoint=True)
    return np.interp(x_new, x_old, f0).astype(np.float32)


def load_models_api(args, device=torch.device('cuda')):
    '''加载全部模型并打包为 bundle；推理期需要的派生配置一并放入。'''
    dit_checkpoint_path = Path(args.checkpoint)
    dit_config_path = Path(args.config)
    print(f'load model from {dit_checkpoint_path}')
    print(f'load config from {dit_config_path}')

    # f0 extractor (prefer the local rmvpe weights; fall back to HF hub)
    from modules.rmvpe import RMVPE
    if LOCAL_RMVPE_PATH.is_file():
        model_path = LOCAL_RMVPE_PATH
        print(f'load rmvpe from {model_path}')
    else:
        model_path = load_custom_model_from_hf('lj1995/VoiceConversionWebUI', 'rmvpe.pt', None)
    f0_extractor = RMVPE(model_path, is_half=False, device=device)
    f0_fn = f0_extractor.infer_from_audio

    config = yaml.safe_load(dit_config_path.read_text(encoding='utf-8'))
    model_params = recursive_munch(config['model_params'])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage='DiT')
    sr = config['preprocess_params']['sr']

    model, _, _, _ = load_checkpoint(
        model,
        None,
        str(dit_checkpoint_path),
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )

    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        'funasr/campplus', 'campplus_cn_common.bin', config_filename=None
    )

    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location='cpu'))
    campplus_model.eval()
    campplus_model.to(device)

    # ------------------------------------------------------------------
    # NOTE: the original bigvgan ('BigGAN') vocoder block was removed here.
    # Synthesis now goes through Pupu-Vocoder + PC-NSF-HiFiGAN (see
    # run_inference); the DiT-predicted Mel is never vocoded by bigvgan.
    # ------------------------------------------------------------------

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name

        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors='pt',
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    else:
        raise ValueError(f'Unknown speech tokenizer type: {speech_tokenizer_type}')

    # Generate mel spectrograms (SVC model convention: ln-compressed, fmin=0)
    mel_fn_args = {
        'n_fft': config['preprocess_params']['spect_params']['n_fft'],
        'win_size': config['preprocess_params']['spect_params']['win_length'],
        'hop_size': config['preprocess_params']['spect_params']['hop_length'],
        'num_mels': config['preprocess_params']['spect_params']['n_mels'],
        'sampling_rate': sr,
        'fmin': config['preprocess_params']['spect_params'].get('fmin', 0),
        'fmax': None if config['preprocess_params']['spect_params'].get('fmax', 'None') == 'None' else 8000,
        'center': False
    }

    def to_mel(x):
        return mel_spectrogram(x, **mel_fn_args)

    # ---- swapped-in vocoders ------------------------------------------
    # 分块 + autocast 半精度推理：峰值显存与音频总长解耦，12GB 卡不再爆显存
    vocoder_dtype = VOCODER_DTYPES[args.vocoder_dtype]
    print(f'[vocoders] dtype={args.vocoder_dtype}, chunk={args.vocoder_chunk} frames, '
          f'overlap={args.vocoder_overlap} frames')
    pupu_vocoder = load_pupu_vocoder(PUPU_VOCODER_DIR, device=device, dtype=vocoder_dtype,
                                     chunk_frames=args.vocoder_chunk,
                                     overlap_frames=args.vocoder_overlap)   # stage 1
    pcnsf_vocoder = load_pc_nsf_hifigan(PC_NSF_HIFIGAN_DIR, device=device, dtype=vocoder_dtype,
                                        chunk_frames=args.vocoder_chunk,
                                        overlap_frames=args.vocoder_overlap)  # stage 2

    return {
        'model': model,
        'semantic_fn': semantic_fn,
        'f0_fn': f0_fn,
        'campplus_model': campplus_model,
        'mel_fn': to_mel,
        'mel_fn_args': mel_fn_args,
        'pupu_vocoder': pupu_vocoder,
        'pcnsf_vocoder': pcnsf_vocoder,
        # 推理期不再重复读取 yaml：这里预取 length_regulator 的开关
        'use_style_residual':
            config['model_params']['length_regulator'].get('use_style_residual', False),
    }


@torch.no_grad()
def run_inference(args, bundle, device=torch.device('cuda')):
    '''Voice conversion up to the predicted Mel, then the swapped two-stage
    vocoder synthesis:

      predicted Mel chunks --(concat)--> Pupu-Vocoder --> intermediate waveform
      (in-memory, no disk round-trip) --> re-extracted Mel + explicit F0
      (pitch-scaled) --> PC-NSF-HiFiGAN --> final.flac
    '''
    model = bundle['model']
    semantic_fn = bundle['semantic_fn']
    f0_fn = bundle['f0_fn']
    campplus_model = bundle['campplus_model']
    mel_fn = bundle['mel_fn']
    mel_fn_args = bundle['mel_fn_args']
    pupu_vocoder = bundle['pupu_vocoder']
    pcnsf_vocoder = bundle['pcnsf_vocoder']
    use_style_residual = bundle['use_style_residual']

    fp16 = args.fp16
    f0_condition = args.f0_condition
    diffusion_steps = args.diffusion_steps

    source, target = Path(args.source), Path(args.target)
    print(f'[input] source: {source}')
    print(f'[input] target: {target}')

    # 先以模型自身采样率加载音频；管线内部采样率随后按是否带 F0 条件重定义
    model_sr = mel_fn_args['sampling_rate']
    source_audio = torch.tensor(librosa.load(source, sr=model_sr)[0]).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(librosa.load(target, sr=model_sr)[0][:model_sr * 25]).unsqueeze(0).float().to(device)

    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16

    time_vc_start = time.time()
    # Resample
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # <= 30 s 时 whisper 单次前向即可；长音频走重叠滑窗分块编码
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 秒重叠
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            S_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_chunk)
            else:
                S_alt_list.append(S_chunk[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.float())
    mel2 = mel_fn(ref_audio.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * args.length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        shifted_log_f0_alt = log_f0_alt.clone()
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)

        # automatic f0 adjust（注意：forch_pitch_shift 是上游 mm4 的既有拼写）
        shifted_f0_alt, pitch_shift = preprocess_voice_conversion(
            voiced_f0_ori=voiced_F0_ori,
            voiced_f0_alt=voiced_F0_alt,
            shifted_f0_alt=shifted_f0_alt,
            enable_adaptive=True,
            max_shift_semitones=24,
            forch_pitch_shift=args.semi_tone_shift,
        )
        print(f'automatic pitch shift {pitch_shift} semi tones')
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None
        pitch_shift = 0

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss, style_cond = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3,
        f0=shifted_f0_alt, style=style2, return_style_residual=True)
    prompt_condition, _, codes, commitment_loss, codebook_loss, style_prompt = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3,
        f0=F0_ori, style=style2, return_style_residual=True)

    max_source_window = max_context_window - mel2.size(2)
    # 将源条件 cond 分块逐段预测 Mel 并收集结果
    total_frames = cond.size(1)
    processed_frames = 0
    pred_mel_chunks = []
    # 进度条实时显示已处理帧数 / 总条件帧数
    pbar = tqdm(total=total_frames, desc='flow-matching', unit='frame', dynamic_ncols=True)
    while processed_frames < total_frames:
        # 显式确定本块终点：中间块取满窗口、末块负责收尾（可不足一窗）
        chunk_end = min(processed_frames + max_source_window, total_frames)
        chunk_cond = cond[:, processed_frames:chunk_end]
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        # use_style_residual
        if use_style_residual:
            chunk_style_cond = style_cond[:, processed_frames:chunk_end]
            cat_style_cond = torch.cat([style_prompt, chunk_style_cond], dim=1)
        else:
            cat_style_cond = None
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            # Voice Conversion (predicts Mel, not wave)
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps,
                inference_cfg_rate=args.inference_cfg_rate, style_r=cat_style_cond,
                pbar=None,
            )
            vc_target = vc_target[:, :, mel2.size(-1):]

        # 每个 chunk 会重复生成前 overlap 帧，后续块裁掉后再无缝拼接
        if processed_frames == 0:
            pred_mel_chunks.append(vc_target.float())
        else:
            pred_mel_chunks.append(vc_target.float()[:, :, overlap_frame_len:])
        # 关键修复：原实现按 `advance = 本块输出帧数 - overlap` 推进，
        # 当尾部剩余帧数收敛到恰好 overlap_frame_len 时 advance 恒为 0，
        # processed_frames 永远无法到达 total_frames -> 死循环
        # （表现为进度条停在约 99.9%（显示成 100%）但推理永不结束）。
        # 现改为显式跳到本块终点：中间块回退 overlap 帧以便与下一块无缝拼接，
        # 末块处理完直接退出；max() 兜底保证 processed_frames 严格单调前进。
        if chunk_end >= total_frames:
            pbar.update(total_frames - pbar.n)
            break
        processed_frames = max(chunk_end - overlap_frame_len, processed_frames + 1)
        pbar.update(processed_frames - pbar.n)
    pbar.close()

    pred_mel = torch.cat(pred_mel_chunks, dim=2)  # (1, num_mels, frames)
    print(f'predicted Mel: {pred_mel.shape[2]} frames @ {sr} Hz')

    time_vc_end = time.time()
    print(f'flow-matching stage RTF: {(time_vc_end - time_vc_start) / (pred_mel.size(2) * hop_length) * sr}')

    # DiT 主干推理结束：把缓存池里的碎片化显存归还驱动，给声码器腾出连续空间
    torch.cuda.empty_cache()

    exp_path = Path(args.output) / args.expname
    exp_path.mkdir(parents=True, exist_ok=True)  # 连同 --output 根目录一并创建
    src_stem, tgt_stem = source.stem, target.stem

    # ==================================================================
    # Stage 1: Pupu-Vocoder -- 预测 Mel -> 中间波形（仅驻留内存，不落盘）
    # ==================================================================
    temp_wave = pupu_vocoder.mel_to_wav(pred_mel.cpu())
    print(f'[stage 1] Pupu-Vocoder -> in-memory waveform ({temp_wave.shape[-1]} samples @ {pupu_vocoder.sample_rate} Hz)')

    # ==================================================================
    # Stage 2: 显式 F0 控制 + PC-NSF-HiFiGAN -> 最终 FLAC
    # ==================================================================
    # 2.1 用 PC-NSF-HiFiGAN 自身的谱参数从中间波形重提取 Mel
    #     （128 bins, hop 512, fmin 40, fmax 16000 @ 44.1 kHz）
    # 两级声码器当前均为 44.1 kHz，直接内存传递；若日后换用其它采样率的
    # Pupu checkpoint，这里在内存里兜底重采样，保证谱参数对齐
    if pupu_vocoder.sample_rate != pcnsf_vocoder.sample_rate:
        temp_wave = librosa.resample(temp_wave, orig_sr=pupu_vocoder.sample_rate, target_sr=pcnsf_vocoder.sample_rate)
        print(f'[stage 2] resampled intermediate waveform to {pcnsf_vocoder.sample_rate} Hz')
    temp_wave = torch.from_numpy(temp_wave).unsqueeze(0).float().to(device)
    new_mel = pcnsf_vocoder.wave_to_mel(temp_wave)  # (1, num_mels, frames)
    n_frames = new_mel.shape[2]

    # 2.2 从原始输入音频提取显式 F0 条件（复用已加载的 RMVPE，
    #     16 kHz 重采样、10 ms 网格），并施加变调（--pitch-shift / --f0-scale）
    f0_scale = args.f0_scale * (2.0 ** (args.pitch_shift / 12.0))
    explicit_f0 = extract_explicit_f0(f0_fn, converted_waves_16k[0])
    explicit_f0 = resize_f0_to_frames(explicit_f0, n_frames) * f0_scale
    print(f'[stage 2] explicit F0 scale x{f0_scale:.4f} (pitch shift {args.pitch_shift:+g} st, f0 scale {args.f0_scale:g})')

    # 2.3 用音高可控声码器做最终合成
    final_wave = pcnsf_vocoder.mel_to_wav(new_mel, explicit_f0)
    final_tensor = torch.clamp(torch.from_numpy(final_wave)[None, :].float(), -1.0, 1.0)

    time_synth_end = time.time()
    print(f'Overall RTF: {(time_synth_end - time_vc_start) / final_tensor.size(-1) * sr}')

    # 输出统一为 FLAC，文件名附带元数据后缀（项目名/ckpt/变调等）
    vc_stem = f'{src_stem}_{tgt_stem}_{gen_output_suffix(args)}'
    if uuid := getattr(args, 'uuid', None):
        vc_stem += f'_{uuid}'
    output_path = exp_path / f'{vc_stem}.flac'
    torchaudio.save(str(output_path), final_tensor.cpu(), sr)
    print(f'[stage 2] PC-NSF-HiFiGAN wrote {output_path}')
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YingMusic-SVC inference with Pupu-Vocoder -> PC-NSF-HiFiGAN '
                    'two-stage synthesis and explicit F0 (pitch) control.')
    parser.add_argument('--source', type=str, help='Source vocal audio (wav/flac/mp3 ...)')
    parser.add_argument('--target', type=str, help='Reference audio providing the target timbre')
    parser.add_argument('--diffusion-steps', type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/YingMusic-SVC-full.pt',
                        help='Path to the SVC checkpoint file')
    parser.add_argument('--expname', type=str, default='swap_vocoder')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--fp16', type=str, default='True')
    parser.add_argument('--accompany', type=str, default=None,
                        help='Optional accompaniment track for remixing (echo/reverb)')
    parser.add_argument('--config', type=str, default='./configs/YingMusic-SVC.yml')

    # --- 显式音高控制（作用于最终 PC-NSF-HiFiGAN 阶段） ---
    parser.add_argument('--pitch-shift', type=float, default=0.0, dest='pitch_shift', help='Pitch shift in semitones applied to the explicit F0 '
                             'condition of PC-NSF-HiFiGAN (e.g. 2.0 = +2 semitones up)')
    parser.add_argument('--f0-scale', type=float, default=1.0, dest='f0_scale',
                        help='Direct multiplicative scale on the explicit F0 '
                             '(e.g. 1.2 ~= up 3.2 semitones, 0.8 ~= down 3.9 semitones); '
                             'applied on top of --pitch-shift')

    parser.add_argument('--semi-tone-shift', type=float, default=None, dest='semi_tone_shift',
                        help="Forced semi-tone shift for the SVC model's internal adaptive F0 alignment; None keeps automatic sandhi")
    parser.add_argument('--length-adjust', type=float, default=1.0, dest='length_adjust')
    parser.add_argument('--inference-cfg-rate', type=float, default=0.7, dest='inference_cfg_rate')
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--skip-check', action='store_true', dest='skip_check',
                        help='Skip the pre-flight dependency/weight check')

    # --- 声码器显存优化（分块 + autocast 半精度） ---
    parser.add_argument('--vocoder-dtype', type=str, default='bf16', dest='vocoder_dtype',
                        choices=['fp32', 'bf16', 'fp16'],
                        help='两个声码器的计算精度：fp16/bf16 用 autocast 把激活值降到半精度，'
                             '显存约减半（权重始终 fp32 驻留）；如听到伪影可退回 fp32')
    parser.add_argument('--vocoder-chunk', type=int, default=2048, dest='vocoder_chunk',
                        help='声码器分块推理的 mel 帧数（hop512@44.1kHz 下每帧约 11.6ms，'
                             '512 帧约 6s 音频；<=0 表示不分块、整段前向）')
    parser.add_argument('--vocoder-overlap', type=int, default=64, dest='vocoder_overlap',
                        help='分块间重叠的 mel 帧数（波形交叉淡化拼接区，需大于模型感受野）')
    args = parser.parse_args()

    args.cuda = torch.device(f'cuda:{args.cuda}')
    args.fp16 = str2bool(args.fp16)
    if args.fp16:
        print('Start fp16 to accelerate inference！')

    args.f0_condition = True
    if not args.skip_check:
        preflight_check(args)

    models = load_models_api(args, device=args.cuda)
    vc = run_inference(args, models, device=args.cuda)
    if args.accompany:
        # 回声+混响 remix，产物写入 outputs/<expname>/accompany/
        acc_dir = vc.parent / 'accompany'
        acc_dir.mkdir(parents=True, exist_ok=True)
        echo_then_reverb_save(str(vc), str(acc_dir / vc.name), args.accompany)

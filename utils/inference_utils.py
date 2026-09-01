from __future__ import annotations

import argparse
import importlib.util
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_NAME = 'YingMusic'
DEFAULT_DIFFUSION_STEPS = 30
LOCAL_RMVPE_PATH = Path('./pretrain/rmvpe/model.pt')

_CORE_DEPS = (
    'numpy', 'torch', 'torchaudio', 'librosa', 'yaml', 'soundfile', 'tqdm'
)
# 从 ckpt 文件名提取训练步数：model-step=10000.ckpt -> 10000
_CKPT_STEP_PATTERN = re.compile(r'step[=_-](\d+)', re.IGNORECASE)

COMMON_REQUIRED_DEPS = {
    'scipy': 'scipy',
    'munch': 'munch',
    'einops': 'einops',
    'transformers': 'transformers',
    'huggingface_hub': 'huggingface-hub',
    'beartype': 'beartype',
    'rotary_embedding_torch': 'rotary-embedding-torch',
    'ml_collections': 'ml-collections',
    'loralib': 'loralib',
    'accelerate': 'accelerate',
}


@dataclass(frozen=True)
class InferenceContext:
    '''声码器无关的推理中间结果，供不同合成后端复用。'''

    source: Path
    model: Any
    f0_fn: Any
    converted_waves_16k: Any
    mel2: Any
    style2: Any
    cond: Any
    prompt_condition: Any
    style_cond: Any
    style_prompt: Any
    use_style_residual: bool
    sr: int
    hop_length: int
    overlap_frame_len: int
    max_source_window: int
    started_at: float


def check_core_deps() -> None:
    '''在导入重量级模块前报告基础依赖缺失，避免原始 traceback。'''
    missing = [module for module in _CORE_DEPS if importlib.util.find_spec(module) is None]
    if not missing:
        return

    print('=' * 72)
    print('MISSING PYTHON DEPENDENCIES:', ', '.join(missing))
    print('Please install them yourself (no automatic downloads), e.g.:')
    print(f'    uv pip install {" ".join(missing)}')
    print('=' * 72)
    raise SystemExit(2)


def gen_output_suffix(
    args: argparse.Namespace,
    *,
    vocoder_suffix: str | None = None,
    include_f0_scale: bool = False,
) -> str:
    '''由 CLI 参数拼装输出文件名的元数据后缀，形如：

        YingMusic@+2key_10ks_3ref_1.2f0_50step
            │         │     │    │      │     └─ 扩散步数（非默认时才写入）
            │         │     │    │      └─ 显式 F0 缩放（非 1.0 时才写入）
            │         │     │    └─ 参考片段数（多于 1 段时才写入）
            │         │     └─ ckpt 训练步数（文件名可解析出 step 时写入）
            │         └─ 变调（semitones）
            └─ 项目名

    与默认值相同的参数不写入，避免文件名冗长。
    '''
    ckpt_name = Path(args.checkpoint).name
    parts = [f'{PROJECT_NAME}@{args.pitch_shift:+g}key']

    if match := _CKPT_STEP_PATTERN.search(ckpt_name):
        parts.append(f'{int(match.group(1)) / 1000:g}ks')
    if (ref_count := len(args.target)) > 1:
        parts.append(f'{ref_count}ref')
    if include_f0_scale and args.f0_scale != 1.0:
        parts.append(f'{args.f0_scale:g}f0')
    if args.semi_tone_shift is not None:
        parts.append(f'auto{args.semi_tone_shift:+g}st')
    if args.length_adjust != 1.0:
        parts.append(f'len{args.length_adjust:g}')
    if args.diffusion_steps != DEFAULT_DIFFUSION_STEPS:
        parts.append(f'{args.diffusion_steps}step')
    if vocoder_suffix:
        parts.append(vocoder_suffix)
    return '_'.join(parts)


def preflight_check(
    args: argparse.Namespace,
    *,
    required_deps: Mapping[str, str],
    weight_checks: Sequence[tuple[str, Path, bool]],
) -> None:
    '''统一检查 Python 包与本地权重；只报告问题，不自动安装或下载。'''
    ok = True

    print('-' * 72)
    print('[preflight] checking python dependencies ...')
    missing_modules = [
        module for module in required_deps
        if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        ok = False
        packages = [required_deps[module] for module in missing_modules]
        print(f'  [MISSING] {", ".join(missing_modules)}')
        print(f'            install manually, e.g.: uv pip install {" ".join(packages)}')
    else:
        print('  [OK] all required packages found')

    print('[preflight] checking model weights ...')
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


def load_common_models(args: argparse.Namespace, device: Any) -> dict[str, Any]:
    '''加载两条推理管线共用的 RMVPE、DiT、CAMPPlus、Whisper 与 Mel 配置。'''
    import torch
    import yaml
    from transformers import AutoFeatureExtractor, WhisperModel

    from hf_utils import load_custom_model_from_hf
    from modules.audio import mel_spectrogram
    from modules.campplus.DTDNN import CAMPPlus
    from modules.commons import build_model, load_checkpoint, recursive_munch
    from modules.rmvpe import RMVPE

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    print(f'load model from {checkpoint_path}')
    print(f'load config from {config_path}')

    if LOCAL_RMVPE_PATH.is_file():
        rmvpe_path = LOCAL_RMVPE_PATH
        print(f'load rmvpe from {rmvpe_path}')
    else:
        rmvpe_path = load_custom_model_from_hf(
            'lj1995/VoiceConversionWebUI', 'rmvpe.pt', None
        )
    f0_extractor = RMVPE(rmvpe_path, is_half=False, device=device)

    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    model_params = recursive_munch(config['model_params'])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage='DiT')
    model, _, _, _ = load_checkpoint(
        model,
        None,
        str(checkpoint_path),
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    campplus_path = load_custom_model_from_hf(
        'funasr/campplus', 'campplus_cn_common.bin', config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_path, map_location='cpu'))
    campplus_model.eval().to(device)

    if model_params.speech_tokenizer.type != 'whisper':
        raise ValueError(
            f'Unknown speech tokenizer type: {model_params.speech_tokenizer.type}'
        )
    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(
        whisper_name, torch_dtype=torch.float16
    ).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k):
        inputs = whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors='pt',
            return_attention_mask=True,
        )
        input_features = whisper_model._mask_input_features(
            inputs.input_features,
            attention_mask=inputs.attention_mask,
        ).to(device)
        with torch.no_grad():
            outputs = whisper_model.encoder(
                input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        semantics = outputs.last_hidden_state.to(torch.float32)
        return semantics[:, :waves_16k.size(-1) // 320 + 1]

    spect_params = config['preprocess_params']['spect_params']
    mel_fn_args = {
        'n_fft': spect_params['n_fft'],
        'win_size': spect_params['win_length'],
        'hop_size': spect_params['hop_length'],
        'num_mels': spect_params['n_mels'],
        'sampling_rate': config['preprocess_params']['sr'],
        'fmin': spect_params.get('fmin', 0),
        'fmax': None if spect_params.get('fmax', 'None') == 'None' else 8000,
        'center': False,
    }

    def mel_fn(audio):
        return mel_spectrogram(audio, **mel_fn_args)

    return {
        'model': model,
        'model_params': model_params,
        'config': config,
        'semantic_fn': semantic_fn,
        'f0_fn': f0_extractor.infer_from_audio,
        'campplus_model': campplus_model,
        'mel_fn': mel_fn,
        'mel_fn_args': mel_fn_args,
        'use_style_residual':
            config['model_params']['length_regulator'].get('use_style_residual', False),
    }


def _encode_source_semantics(semantic_fn: Any, waves_16k: Any) -> Any:
    '''Whisper 超过 30 秒时使用与原入口一致的 5 秒重叠滑窗。'''
    import torch

    if waves_16k.size(-1) <= 16000 * 30:
        return semantic_fn(waves_16k)

    overlap_seconds = 5
    chunks = []
    buffer = None
    traversed = 0
    while traversed < waves_16k.size(-1):
        if buffer is None:
            chunk = waves_16k[:, traversed:traversed + 16000 * 30]
        else:
            chunk = torch.cat(
                [
                    buffer,
                    waves_16k[
                        :, traversed:traversed + 16000 * (30 - overlap_seconds)
                    ],
                ],
                dim=-1,
            )
        semantics = semantic_fn(chunk)
        chunks.append(semantics if traversed == 0 else semantics[:, 50 * overlap_seconds:])
        buffer = chunk[:, -16000 * overlap_seconds:]
        traversed += (
            30 * 16000
            if traversed == 0
            else chunk.size(-1) - 16000 * overlap_seconds
        )
    return torch.cat(chunks, dim=1)


def prepare_inference_context(
    args: argparse.Namespace,
    bundle: Mapping[str, Any],
    device: Any,
    *,
    internal_pitch_shift: float = 0.0,
) -> InferenceContext:
    '''完成声码器之前的全部共享推理步骤，并返回分块合成所需上下文。'''
    import librosa
    import torch
    import torchaudio

    from mm4 import preprocess_voice_conversion

    model = bundle['model']
    semantic_fn = bundle['semantic_fn']
    f0_fn = bundle['f0_fn']
    campplus_model = bundle['campplus_model']
    mel_fn = bundle['mel_fn']
    model_sr = bundle['mel_fn_args']['sampling_rate']

    source = Path(args.source)
    targets = [Path(target) for target in args.target]
    print(f'[input] source: {source}')
    for index, target in enumerate(targets):
        print(f'[input] target[{index}]: {target}')

    source_audio = torch.tensor(
        librosa.load(source, sr=model_sr)[0]
    ).unsqueeze(0).float().to(device)
    ref_parts = [
        torch.tensor(librosa.load(target, sr=model_sr)[0][:model_sr * 25])
        for target in targets
    ]
    ref_audio = torch.cat(ref_parts)[:model_sr * 25].unsqueeze(0).float().to(device)
    print(f'[input] {len(ref_parts)} ref clip(s), '
          f'{ref_audio.size(-1) / model_sr:.2f}s used')

    sr = 44100 if args.f0_condition else 22050
    hop_length = 512 if args.f0_condition else 256
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    started_at = time.time()

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    source_semantics = _encode_source_semantics(semantic_fn, converted_waves_16k)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    ref_semantics = semantic_fn(ref_waves_16k)

    source_mel = mel_fn(source_audio.float())
    ref_mel = mel_fn(ref_audio.float())
    source_lengths = torch.LongTensor([
        int(source_mel.size(2) * args.length_adjust)
    ]).to(source_mel.device)
    ref_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

    ref_features = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    ref_features -= ref_features.mean(dim=0, keepdim=True)
    style = campplus_model(ref_features.unsqueeze(0))

    if args.f0_condition:
        ref_f0 = torch.from_numpy(
            f0_fn(ref_waves_16k[0], thred=0.03)
        ).to(device)[None]
        source_f0 = torch.from_numpy(
            f0_fn(converted_waves_16k[0], thred=0.03)
        ).to(device)[None]
        shifted_source_f0, automatic_shift = preprocess_voice_conversion(
            voiced_f0_ori=ref_f0[ref_f0 > 1],
            voiced_f0_alt=source_f0[source_f0 > 1],
            shifted_f0_alt=torch.exp(torch.log(source_f0 + 1e-5)),
            enable_adaptive=True,
            max_shift_semitones=24,
            forch_pitch_shift=args.semi_tone_shift,
        )
        print(f'automatic pitch shift {automatic_shift} semi tones')
        if internal_pitch_shift != 0.0:
            pitch_scale = 2.0 ** (internal_pitch_shift / 12.0)
            shifted_source_f0 *= pitch_scale
            print(f'manual pitch shift {internal_pitch_shift:+g} semi tones '
                  f'(internal F0 x{pitch_scale:.4f})')
    else:
        ref_f0 = None
        shifted_source_f0 = None

    cond, _, _, _, _, style_cond = model.length_regulator(
        source_semantics,
        ylens=source_lengths,
        n_quantizers=3,
        f0=shifted_source_f0,
        style=style,
        return_style_residual=True,
    )
    prompt_condition, _, _, _, _, style_prompt = model.length_regulator(
        ref_semantics,
        ylens=ref_lengths,
        n_quantizers=3,
        f0=ref_f0,
        style=style,
        return_style_residual=True,
    )

    return InferenceContext(
        source=source,
        model=model,
        f0_fn=f0_fn,
        converted_waves_16k=converted_waves_16k,
        mel2=ref_mel,
        style2=style,
        cond=cond,
        prompt_condition=prompt_condition,
        style_cond=style_cond,
        style_prompt=style_prompt,
        use_style_residual=bundle['use_style_residual'],
        sr=sr,
        hop_length=hop_length,
        overlap_frame_len=overlap_frame_len,
        max_source_window=max_context_window - ref_mel.size(2),
        started_at=started_at,
    )


def infer_mel_chunk(
    context: InferenceContext,
    args: argparse.Namespace,
    device: Any,
    start: int,
    end: int,
) -> Any:
    '''为指定条件帧区间执行一次 flow-matching，返回去除 prompt 的 Mel。'''
    import torch

    chunk_condition = context.cond[:, start:end]
    condition = torch.cat([context.prompt_condition, chunk_condition], dim=1)
    if context.use_style_residual:
        chunk_style = context.style_cond[:, start:end]
        style = torch.cat([context.style_prompt, chunk_style], dim=1)
    else:
        style = None

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if args.fp16 else torch.float32,
    ):
        mel = context.model.cfm.inference(
            condition,
            torch.LongTensor([condition.size(1)]).to(context.mel2.device),
            context.mel2,
            context.style2,
            None,
            args.diffusion_steps,
            inference_cfg_rate=args.inference_cfg_rate,
            style_r=style,
            pbar=None,
        )
    return mel[:, :, context.mel2.size(-1):]


def build_output_path(
    args: argparse.Namespace,
    source: Path,
    *,
    vocoder_suffix: str | None = None,
    include_f0_scale: bool = False,
) -> Path:
    '''创建输出目录并根据统一元数据规则生成 FLAC 路径。'''
    output_dir = source.parent if args.output is None else Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = gen_output_suffix(
        args,
        vocoder_suffix=vocoder_suffix,
        include_f0_scale=include_f0_scale,
    )
    stem = f'{source.stem}_{stem}'
    if uuid := getattr(args, 'uuid', None):
        stem += f'_{uuid}'
    return output_dir / f'{stem}.flac'


def add_common_inference_arguments(
    parser: argparse.ArgumentParser,
    *,
    pitch_shift_help: str,
) -> None:
    '''注册两份入口一致的 CLI 参数。'''
    parser.add_argument('--source', type=str, help='Source vocal audio (wav/flac/mp3 ...)')
    parser.add_argument(
        '--target',
        type=str,
        nargs='+',
        help='Reference audio(s) providing the target timbre; '
             '可传多段，按顺序拼接后截断至 25 s',
    )
    parser.add_argument('--diffusion-steps', type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints/YingMusic-SVC-full.pt',
        help='Path to the SVC checkpoint file',
    )
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--fp16', type=str, default='True')
    parser.add_argument(
        '--accompany',
        type=str,
        default=None,
        help='Optional accompaniment track for remixing (echo/reverb)',
    )
    parser.add_argument('--config', type=str, default='./configs/YingMusic-SVC.yml')
    parser.add_argument(
        '--pitch-shift',
        type=float,
        default=0.0,
        dest='pitch_shift',
        help=pitch_shift_help,
    )
    parser.add_argument(
        '--semi-tone-shift',
        type=float,
        default=None,
        dest='semi_tone_shift',
        help="Forced semi-tone shift for the SVC model's internal adaptive F0 alignment; None keeps automatic sandhi",
    )
    parser.add_argument('--length-adjust', type=float, default=1.0, dest='length_adjust')
    parser.add_argument(
        '--inference-cfg-rate',
        type=float,
        default=0.7,
        dest='inference_cfg_rate',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory; 默认 None 表示直接写到源音频所在目录',
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        dest='skip_check',
        help='Skip the pre-flight dependency/weight check',
    )


def finalize_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    '''规范化设备与布尔参数，并注入模型固定使用的 F0 条件开关。'''
    import torch

    if isinstance(args.fp16, bool):
        fp16 = args.fp16
    elif args.fp16.lower() in ('yes', 'true', 't', 'y', '1'):
        fp16 = True
    elif args.fp16.lower() in ('no', 'false', 'f', 'n', '0'):
        fp16 = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    args.cuda = torch.device(f'cuda:{args.cuda}')
    args.fp16 = fp16
    args.f0_condition = True
    if args.fp16:
        print('Start fp16 to accelerate inference！')
    return args


def remix_accompaniment(output_path: Path, accompaniment: str | None) -> None:
    '''可选地生成回声与混响伴奏版本。'''
    if not accompaniment:
        return

    from Remix.auger import echo_then_reverb_save

    accompany_dir = output_path.parent / 'accompany'
    accompany_dir.mkdir(parents=True, exist_ok=True)
    echo_then_reverb_save(
        str(output_path),
        str(accompany_dir / output_path.name),
        accompaniment,
    )

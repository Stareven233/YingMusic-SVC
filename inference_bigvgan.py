'''
YingMusic-SVC inference -- original BigVGAN synthesis path:

    YingMusic-SVC DiT -> predicted Mel chunks
      -> BigVGAN (local weights) -> waveform crossfade -> final.flac

BigVGAN weights are loaded only from ./pretrain/vocoder/BigVGAN. The shared
model preparation and CLI behavior live in utils/inference_utils.py.

./.venv/Scripts/python.exe inference_bigvgan.py --source vocal.wav --target timbre.wav
./.venv/Scripts/python.exe inference_bigvgan.py --source vocal.wav --target timbre1.wav timbre2.wav
'''

# 基础依赖检查必须早于第三方导入，因此这里有意拆分 import block。
# ruff: noqa: I001

from utils.inference_utils import check_core_deps

check_core_deps()

import argparse
import time
import warnings
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from utils.inference_utils import (
    COMMON_REQUIRED_DEPS,
    LOCAL_RMVPE_PATH,
    add_common_inference_arguments,
    build_output_path,
    finalize_inference_args,
    infer_mel_chunk,
    load_common_models,
    preflight_check as _shared_preflight_check,
    prepare_inference_context,
    remix_accompaniment,
)

warnings.simplefilter('ignore')

# BigVGAN.from_pretrained() 从本地目录读取 config.json 与 bigvgan_generator.pt。
BIGVGAN_DIR = Path('./pretrain/vocoder/bigvgan_v2_44khz_128band_512x')


def preflight_check(args: argparse.Namespace) -> None:
    '''补充 BigVGAN 本地权重后调用共享预检。'''
    weight_checks = [
        ('SVC checkpoint (YingMusic-SVC)', Path(args.checkpoint), True),
        ('SVC config', Path(args.config), True),
        ('rmvpe (local)', LOCAL_RMVPE_PATH, False),
        ('BigVGAN config', BIGVGAN_DIR / 'config.json', True),
        ('BigVGAN generator', BIGVGAN_DIR / 'bigvgan_generator.pt', True),
    ]
    _shared_preflight_check(
        args,
        required_deps=COMMON_REQUIRED_DEPS,
        weight_checks=weight_checks,
    )


def crossfade(
    previous_tail: torch.Tensor,
    current_wave: torch.Tensor,
    overlap: int,
) -> torch.Tensor:
    '''对相邻 BigVGAN 波形块执行与原实现一致的等功率交叉淡化。'''
    overlap = min(overlap, previous_tail.numel(), current_wave.numel())
    if overlap <= 0:
        return current_wave

    phase = torch.linspace(0, torch.pi / 2, overlap, dtype=current_wave.dtype)
    fade_out = torch.cos(phase).square()
    fade_in = torch.cos(torch.pi / 2 - phase).square()
    current_wave = current_wave.clone()
    current_wave[:overlap] = (
        current_wave[:overlap] * fade_in
        + previous_tail[-overlap:] * fade_out
    )
    return current_wave


def load_models_api(args, device=None):
    '''加载共享模型后，附加本地 BigVGAN。'''
    if device is None:
        device = torch.device('cuda')
    bundle = load_common_models(args, device)
    model_params = bundle['model_params']
    config = bundle['config']
    if model_params.vocoder.type != 'bigvgan':
        raise ValueError(
            'BigVGAN inference requires model_params.vocoder.type=bigvgan, '
            f'got {model_params.vocoder.type!r}'
        )

    from modules.bigvgan import bigvgan

    print(f'load BigVGAN from {BIGVGAN_DIR}')
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        str(BIGVGAN_DIR),
        use_cuda_kernel=False,
        local_files_only=True,
    )

    spect_params = config['preprocess_params']['spect_params']
    expected_spec = {
        'sampling_rate': config['preprocess_params']['sr'],
        'num_mels': spect_params['n_mels'],
        'hop_size': spect_params['hop_length'],
    }
    mismatches = {
        key: (expected, bigvgan_model.h.get(key))
        for key, expected in expected_spec.items()
        if bigvgan_model.h.get(key) != expected
    }
    if mismatches:
        details = ', '.join(
            f'{key}: expected {expected}, got {actual}'
            for key, (expected, actual) in mismatches.items()
        )
        raise ValueError(
            f'BigVGAN checkpoint is incompatible with the SVC Mel config: {details}'
        )

    bigvgan_model.remove_weight_norm()
    bundle['vocoder_fn'] = bigvgan_model.eval().to(device)
    return bundle


@torch.no_grad()
def run_inference(args, bundle, device=None):
    '''逐块预测 Mel、调用 BigVGAN，并在波形域拼接重叠区。'''
    if device is None:
        device = torch.device('cuda')
    context = prepare_inference_context(
        args,
        bundle,
        device,
        internal_pitch_shift=args.pitch_shift,
    )
    vocoder_fn = bundle['vocoder_fn']
    overlap_wave_len = context.overlap_frame_len * context.hop_length
    total_frames = context.cond.size(1)
    processed_frames = 0
    generated_wave_chunks = []

    pbar = tqdm(
        total=total_frames,
        desc='flow-matching + BigVGAN',
        unit='frame',
        dynamic_ncols=True,
    )
    while processed_frames < total_frames:
        chunk_end = min(
            processed_frames + context.max_source_window,
            total_frames,
        )
        is_last_chunk = chunk_end >= total_frames
        predicted_mel = infer_mel_chunk(
            context,
            args,
            device,
            processed_frames,
            chunk_end,
        )
        # 与原始入口一致，BigVGAN 保持 fp32 且不启用自定义 CUDA kernel。
        current_wave = vocoder_fn(predicted_mel.float()).squeeze(1)[0].float().cpu()

        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(current_wave)
            else:
                generated_wave_chunks.append(current_wave[:-overlap_wave_len])
                previous_tail = current_wave[-overlap_wave_len:]
        elif is_last_chunk:
            generated_wave_chunks.append(
                crossfade(previous_tail, current_wave, overlap_wave_len)
            )
        else:
            generated_wave_chunks.append(
                crossfade(
                    previous_tail,
                    current_wave[:-overlap_wave_len],
                    overlap_wave_len,
                )
            )
            previous_tail = current_wave[-overlap_wave_len:]

        # 保留 16 帧重叠，同时保证特殊尾长下游标不会停滞。
        if is_last_chunk:
            pbar.update(total_frames - pbar.n)
            break
        processed_frames = max(
            chunk_end - context.overlap_frame_len,
            processed_frames + 1,
        )
        pbar.update(processed_frames - pbar.n)
    pbar.close()

    final_tensor = torch.clamp(
        torch.cat(generated_wave_chunks).unsqueeze(0),
        -1.0,
        1.0,
    )
    print(f'BigVGAN output: {final_tensor.size(-1)} samples @ {context.sr} Hz')
    print(
        'Overall RTF: '
        f'{(time.time() - context.started_at) / final_tensor.size(-1) * context.sr}'
    )

    output_path = build_output_path(
        args,
        context.source,
        vocoder_suffix='bigvgan',
    )
    torchaudio.save(str(output_path), final_tensor, context.sr)
    print(f'[BigVGAN] wrote {output_path}')
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='YingMusic-SVC inference with the original direct BigVGAN synthesis path.'
    )
    add_common_inference_arguments(
        parser,
        pitch_shift_help='Additional pitch shift in semitones applied to the SVC model internal F0 condition',
    )
    return parser


if __name__ == '__main__':
    args = finalize_inference_args(build_parser().parse_args())
    if not args.skip_check:
        preflight_check(args)

    models = load_models_api(args, device=args.cuda)
    output = run_inference(args, models, device=args.cuda)
    remix_accompaniment(output, args.accompany)

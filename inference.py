'''
YingMusic-SVC inference -- swapped two-stage vocoder pipeline (training-free):

    YingMusic-SVC DiT -> predicted Mel
      -> [stage 1] Pupu-Vocoder            -> intermediate waveform (in-memory)
      -> re-extract Mel (PC-NSF params) + explicit F0 from the source audio
         (RMVPE, pitch-controllable via --pitch-shift / --f0-scale)
      -> [stage 2] PC-NSF-HiFiGAN          -> final.flac

The original BigGAN-family (bigvgan) vocoding path has been removed; the
predicted Mel is no longer fed to the built-in vocoder directly.

最终输出为 FLAC，文件名自动附带元数据后缀（项目名、ckpt、变调等 CLI参数信息），由 gen_output_suffix() 生成。

Quick start refs:
https://zread.ai/GiantAILab/YingMusic-SVC/2-quick-start
https://zread.ai/GiantAILab/YingMusic-SVC/3-model-download-and-setup

uv venv --python=3.10
uv pip install torch~=2.4.0 torchaudio~=2.4.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r ./requirements.txt
./.venv/Scripts/python.exe -c 'from modelscope import snapshot_download; snapshot_download('giantailab/YingMusic-SVC', local_dir='./checkpoints')'

./.venv/Scripts/python.exe inference.py --source vocal.wav --target timbre.wav
./.venv/Scripts/python.exe inference.py --source vocal.wav --target timbre1.wav timbre2.wav  # 多段参考拼接
uv run ./inference.py --source 'D:/Document/ai-sings/テオ/【翻唱】将手（テオ） ／ Omoi【Kotone(天神子兔音)cover】_Vocals_vocals_noreverb.flac' --target 'D:/Code/projects/DDSP-SVC/data/megumin/train/audio/ちいさな冒険者 めぐみん Version - 高橋李依_vocals_noreverb_0007.wav' 'D:/Code/projects/DDSP-SVC/data/megumin/train/audio/101匹目の羊 -めぐみん Ver.- - 高橋李依_vocals_noreverb_0006.wav' 'D:/Code/projects/DDSP-SVC/data/megumin/train/audio/おうちに帰りたい -めぐみん ver.- - 高橋李依_vocals_noreverb_0016.wav'
'''

# 基础依赖检查必须早于第三方导入，因此这里有意拆分 import block。
# ruff: noqa: I001

from utils.inference_utils import check_core_deps

check_core_deps()

import argparse
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
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

# 两级声码器专属资源与计算参数留在入口脚本，避免污染共享模型流程。
PUPU_VOCODER_DIR = Path(
    './pretrain/vocoder/Pupu-Vocoder/experiments/pupuvocoder/checkpoint/'
    'epoch-0051_step-2553605_loss-62.135194'
)
PC_NSF_HIFIGAN_DIR = Path(
    './pretrain/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02'
)
# 声码器计算精度：fp32 保持原始精度；fp16/bf16 通过 autocast 把激活值降到半精度
# （权重仅 ~50MB，显存大头是随音频长度线性增长的激活值，半精度可再省约一半）
VOCODER_DTYPES = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def preflight_check(args: argparse.Namespace) -> None:
    '''补充两级声码器依赖与权重后调用共享预检。'''
    required_deps = {
        **COMMON_REQUIRED_DEPS,
        'julius': 'julius',
        'safetensors': 'safetensors',
    }
    weight_checks = [
        ('SVC checkpoint (YingMusic-SVC)', Path(args.checkpoint), True),
        ('SVC config', Path(args.config), True),
        ('rmvpe (local)', LOCAL_RMVPE_PATH, False),
        ('Pupu-Vocoder generator', PUPU_VOCODER_DIR / 'model.safetensors', True),
        ('PC-NSF-HiFiGAN config', PC_NSF_HIFIGAN_DIR / 'config.json', True),
        ('PC-NSF-HiFiGAN ckpt', PC_NSF_HIFIGAN_DIR / 'model.ckpt', False),
    ]
    _shared_preflight_check(
        args,
        required_deps=required_deps,
        weight_checks=weight_checks,
    )


def extract_explicit_f0(f0_fn, waves_16k, thred=0.03):
    '''提取连续显式 F0；无声区使用相邻有声帧线性插值。'''
    f0 = np.asarray(f0_fn(waves_16k, thred=thred), dtype=np.float64).reshape(-1)
    voiced = f0 > 0
    if voiced.any():
        indices = np.arange(f0.size)
        f0 = np.interp(indices, indices[voiced], f0[voiced])
    else:
        f0 = np.zeros_like(f0)
    return f0.astype(np.float32)


def resize_f0_to_frames(f0, frame_count):
    '''将 F0 轨迹线性重采样到声码器 Mel 的帧数。'''
    if f0.size == frame_count:
        return f0
    old_positions = np.linspace(0.0, 1.0, f0.size, endpoint=True)
    new_positions = np.linspace(0.0, 1.0, frame_count, endpoint=True)
    return np.interp(new_positions, old_positions, f0).astype(np.float32)


def load_models_api(args, device=None):
    '''加载共享模型后，附加 Pupu 与 PC-NSF-HiFiGAN 两级声码器。'''
    if device is None:
        device = torch.device('cuda')

    # 延迟导入声码器模块，确保 preflight 能先报告专属依赖缺失。
    from modules.pc_nsf_hifigan import load_pc_nsf_hifigan
    from modules.pupu_vocoder import load_pupu_vocoder

    bundle = load_common_models(args, device)
    vocoder_dtype = VOCODER_DTYPES[args.vocoder_dtype]
    print(f'[vocoders] dtype={args.vocoder_dtype}, chunk={args.vocoder_chunk} frames, '
          f'overlap={args.vocoder_overlap} frames')

    bundle.update({
        'pupu_vocoder': load_pupu_vocoder(
            PUPU_VOCODER_DIR,
            device=device,
            dtype=vocoder_dtype,
            chunk_frames=args.vocoder_chunk,
            overlap_frames=args.vocoder_overlap,
        ),
        'pcnsf_vocoder': load_pc_nsf_hifigan(
            PC_NSF_HIFIGAN_DIR,
            device=device,
            dtype=vocoder_dtype,
            chunk_frames=args.vocoder_chunk,
            overlap_frames=args.vocoder_overlap,
        ),
    })
    return bundle


@torch.no_grad()
def run_inference(args, bundle, device=None):
    '''预测整段 Mel，再经 Pupu 与 PC-NSF-HiFiGAN 两级声码器合成。'''
    if device is None:
        device = torch.device('cuda')
    context = prepare_inference_context(args, bundle, device)
    pupu_vocoder = bundle['pupu_vocoder']
    pcnsf_vocoder = bundle['pcnsf_vocoder']

    total_frames = context.cond.size(1)
    processed_frames = 0
    pred_mel_chunks = []
    pbar = tqdm(total=total_frames, desc='flow-matching', unit='frame', dynamic_ncols=True)
    while processed_frames < total_frames:
        chunk_end = min(
            processed_frames + context.max_source_window,
            total_frames,
        )
        predicted_mel = infer_mel_chunk(
            context,
            args,
            device,
            processed_frames,
            chunk_end,
        ).float()
        if processed_frames == 0:
            pred_mel_chunks.append(predicted_mel)
        else:
            pred_mel_chunks.append(
                predicted_mel[:, :, context.overlap_frame_len:]
            )

        # 末块直接退出；中间块回退 overlap 帧，且保证游标严格单调递增。
        if chunk_end >= total_frames:
            pbar.update(total_frames - pbar.n)
            break
        processed_frames = max(
            chunk_end - context.overlap_frame_len,
            processed_frames + 1,
        )
        pbar.update(processed_frames - pbar.n)
    pbar.close()

    pred_mel = torch.cat(pred_mel_chunks, dim=2)
    print(f'predicted Mel: {pred_mel.shape[2]} frames @ {context.sr} Hz')
    flow_finished_at = time.time()
    print(
        'flow-matching stage RTF: '
        f'{(flow_finished_at - context.started_at) / (pred_mel.size(2) * context.hop_length) * context.sr}'
    )
    torch.cuda.empty_cache()

    temp_wave = pupu_vocoder.mel_to_wav(pred_mel.cpu())
    print(f'[stage 1] Pupu-Vocoder -> in-memory waveform '
          f'({temp_wave.shape[-1]} samples @ {pupu_vocoder.sample_rate} Hz)')

    if pupu_vocoder.sample_rate != pcnsf_vocoder.sample_rate:
        temp_wave = librosa.resample(
            temp_wave,
            orig_sr=pupu_vocoder.sample_rate,
            target_sr=pcnsf_vocoder.sample_rate,
        )
        print(f'[stage 2] resampled intermediate waveform to '
              f'{pcnsf_vocoder.sample_rate} Hz')

    temp_wave_tensor = torch.from_numpy(temp_wave).unsqueeze(0).float().to(device)
    new_mel = pcnsf_vocoder.wave_to_mel(temp_wave_tensor)
    f0_scale = args.f0_scale * (2.0 ** (args.pitch_shift / 12.0))
    explicit_f0 = extract_explicit_f0(
        context.f0_fn,
        context.converted_waves_16k[0],
    )
    explicit_f0 = resize_f0_to_frames(explicit_f0, new_mel.shape[2]) * f0_scale
    print(f'[stage 2] explicit F0 scale x{f0_scale:.4f} '
          f'(pitch shift {args.pitch_shift:+g} st, f0 scale {args.f0_scale:g})')

    final_wave = pcnsf_vocoder.mel_to_wav(new_mel, explicit_f0)
    final_tensor = torch.clamp(
        torch.from_numpy(final_wave)[None, :].float(),
        -1.0,
        1.0,
    )
    print(
        'Overall RTF: '
        f'{(time.time() - context.started_at) / final_tensor.size(-1) * context.sr}'
    )

    output_path = build_output_path(
        args,
        context.source,
        vocoder_suffix='pupu',
        include_f0_scale=True,
    )
    torchaudio.save(str(output_path), final_tensor.cpu(), context.sr)
    print(f'[stage 2] PC-NSF-HiFiGAN wrote {output_path}')
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='YingMusic-SVC inference with Pupu-Vocoder -> PC-NSF-HiFiGAN '
                    'two-stage synthesis and explicit F0 (pitch) control.'
    )
    add_common_inference_arguments(
        parser,
        pitch_shift_help='Pitch shift in semitones applied to the explicit F0 '
                         'condition of PC-NSF-HiFiGAN (e.g. 2.0 = +2 semitones up)',
    )
    parser.add_argument(
        '--f0-scale',
        type=float,
        default=1.0,
        dest='f0_scale',
        help='Direct multiplicative scale on the explicit F0 '
                             '(e.g. 1.2 ~= up 3.2 semitones, 0.8 ~= down 3.9 semitones); '
                             'applied on top of --pitch-shift',
    )
    parser.add_argument(
        '--vocoder-dtype',
        type=str,
        default='bf16',
        dest='vocoder_dtype',
        choices=['fp32', 'bf16', 'fp16'],
        help='两个声码器的计算精度；如听到伪影可退回 fp32',
    )
    parser.add_argument(
        '--vocoder-chunk',
        type=int,
        default=2048,
        dest='vocoder_chunk',
        help='声码器分块推理的 Mel 帧数（hop512@44.1kHz 下每帧约 11.6ms，512 帧约 6s 音频）；<=0 表示整段前向',
    )
    parser.add_argument(
        '--vocoder-overlap',
        type=int,
        default=64,
        dest='vocoder_overlap',
        help='声码器分块间用于波形交叉淡化的 Mel 帧数',
    )
    return parser


if __name__ == '__main__':
    args = finalize_inference_args(build_parser().parse_args())
    if not args.skip_check:
        preflight_check(args)

    models = load_models_api(args, device=args.cuda)
    output = run_inference(args, models, device=args.cuda)
    remix_accompaniment(output, args.accompany)

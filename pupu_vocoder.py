"""
Loader for the locally bundled Pupu-Vocoder (AFGen) checkpoint:

    pretrain/vocoder/Pupu-Vocoder/
        experiments/pupuvocoder/checkpoint/epoch-0051_step-.../model.safetensors

The Pupu-Vocoder source tree ships its own top-level packages (`modules`,
`utils`, `models`, `dataset`) which collide with this project's packages of
the same name. We therefore import the generator inside a temporary import
context that swaps the conflicting entries of `sys.modules`, and restore the
project's packages afterwards.

The checkpoint directory matches the *base* PupuVocoder configuration
(upsample_initial_channel=512), i.e. `egs/pupuvocoder/exp_config_pupuvocoder.json`
merged over `egs/exp_config_pupuvocoder_base.json` / `config/afgen.json` /
`config/base.json` -- mirrored in `_build_pupu_cfg()` below so no external
config parsing is needed at inference time.
"""

import contextlib
import importlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

PUPO_ROOT = Path(__file__).resolve().parent / "pretrain" / "vocoder" / "Pupu-Vocoder"
PUPO_CKPT_DIR = (
    PUPO_ROOT / "experiments" / "pupuvocoder" / "checkpoint"
    / "epoch-0051_step-2553605_loss-62.135194"
)

# Top-level package names provided by both the project and the pupu repo.
_CONFLICT_TOP_LEVEL = ("modules", "utils", "models", "dataset", "bins")


@contextlib.contextmanager
def _isolated_pupu_imports():
    """Temporarily resolve top-level imports against the pupu repo root."""
    saved_modules = {}
    sys.path.insert(0, str(PUPO_ROOT))
    try:
        for name in list(sys.modules):
            if name.split(".")[0] in _CONFLICT_TOP_LEVEL:
                saved_modules[name] = sys.modules.pop(name)
        yield
    finally:
        sys.path.remove(str(PUPO_ROOT))
        # Drop anything freshly imported from the pupu tree...
        for name in list(sys.modules):
            if name.split(".")[0] in _CONFLICT_TOP_LEVEL and name not in saved_modules:
                del sys.modules[name]
        # ...and put the project's own packages back.
        sys.modules.update(saved_modules)


class _AttrNS(dict):
    """Minimal attribute-access dict used to mimic the pupu cfg namespace."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _build_pupu_cfg(sample_rate=44100, n_mel=128, n_fft=2048, win_size=2048,
                    hop_size=512, fmin=0, fmax=22050):
    """Mirror of egs/pupuvocoder/exp_config_pupuvocoder.json (base variant)."""
    return _AttrNS(
        preprocess=_AttrNS(
            n_mel=n_mel,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_size=win_size,
            hop_size=hop_size,
            fmin=fmin,
            fmax=fmax,
        ),
        model=_AttrNS(
            generator="pupuvocoder",
            pupuvocoder=_AttrNS(
                resblock="1",
                upsample_rates=[8, 8, 2, 2, 2],
                upsample_kernel_sizes=[16, 16, 4, 4, 4],
                upsample_initial_channel=512,
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            ),
        ),
    )


def load_safetensors(path):
    """Load safetensors weights; fall back to a manual parser when the
    `safetensors` package is not installed."""
    try:
        from safetensors.torch import load_file
        return load_file(str(path))
    except ImportError:
        pass

    _DTYPES = {
        'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16,
        'BF16': torch.bfloat16, 'I64': torch.int64, 'I32': torch.int32,
        'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool,
    }
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        blob = f.read()
    tensors = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dtype = _DTYPES[meta["dtype"]]
        t = torch.frombuffer(bytearray(blob[start:end]), dtype=dtype)
        tensors[key] = t.reshape(meta["shape"]).clone()
    return tensors


class PupuVocoderWrapper:
    """mel (batch, n_mel, frames) -> waveform numpy array @ 44.1 kHz.

    显存策略：
    - 分块推理：Pupu-Vocoder 全部由局部算子构成（Conv1d、零插值 + FIR、
      snake 激活），没有全局注意力/FFT，块间重叠超过感受野后用波形域
      线性交叉淡化拼接即可无缝衔接；峰值显存因此只与 chunk 大小相关，
      与音频总长解耦（这是 12GB 卡上长音频爆显存的根治手段）。
    - 半精度：权重仅约 13M 参数（~50MB fp32），显存大头是激活值；用
      autocast 把卷积激活降到 fp16/bf16 可再省约一半。注意不能用
      ``generator.half()``：源码 ResampleUpsampler 里 ``torch.zeros(...)``
      未指定 dtype（恒为 fp32），直接半化权重会导致卷积输入 dtype 不匹配；
      且源码内部已用 ``autocast(enabled=False)`` 保护 FIR 滤波，说明
      autocast 正是其训练时的混合精度路径，按此方式推理最安全。
    """

    def __init__(self, generator, cfg, device, dtype=torch.float32,
                 chunk_frames=512, overlap_frames=64):
        self.generator = generator
        self.cfg = cfg
        self.device = device
        self.dtype = dtype                # 计算精度（权重保持 fp32，靠 autocast 降激活精度）
        self.chunk_frames = chunk_frames  # 分块推理的 mel 帧数；<=0 表示整段前向
        self.overlap_frames = overlap_frames  # 块间重叠帧数（交叉淡化区）
        self.sample_rate = cfg.preprocess.sample_rate

    def _amp_ctx(self):
        """fp16/bf16 时返回 CUDA autocast 上下文；CPU 上强制回退 fp32。"""
        if self.device.type == 'cuda' and self.dtype != torch.float32:
            return torch.autocast('cuda', dtype=self.dtype)
        return contextlib.nullcontext()

    @torch.no_grad()
    def mel_to_wav(self, mel, chunk_frames=None, overlap_frames=None):
        """mel: (1, n_mel, frames) tensor (ln-compressed, HiFiGAN convention).
        Returns 1-D numpy waveform.

        chunk_frames / overlap_frames 不传时使用加载时的默认配置。
        """
        chunk_frames = self.chunk_frames if chunk_frames is None else chunk_frames
        overlap_frames = self.overlap_frames if overlap_frames is None else overlap_frames

        mel = mel.float().to(self.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        total_frames = mel.shape[-1]

        # 短音频或显式关闭分块：整段一次前向
        if chunk_frames is None or chunk_frames <= 0 or total_frames <= chunk_frames:
            with self._amp_ctx():
                wav = self.generator(mel)
            return wav.squeeze().float().cpu().numpy()

        # 总上采样倍率 = hop_size，输出样本数与输入帧数严格成正比（全等长卷积）
        upsample = math.prod(self.cfg.model.pupuvocoder.upsample_rates)  # 8*8*2*2*2 = 256
        ov_samples = overlap_frames * upsample
        out = None
        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            # 末段剩余不足一个重叠区时并入当前块，避免产生过短末块
            if 0 < total_frames - end < overlap_frames:
                end = total_frames
            with self._amp_ctx():
                wav = self.generator(mel[:, :, start:end])
            wav = wav.squeeze().float().cpu().numpy()
            if out is None:
                out = wav
            else:
                # 与上一块的尾部做线性交叉淡化（两版信号在重叠区内高度一致，
                # 线性淡化即可无缝；重叠帧数远超模型感受野，边界不可闻）
                n = min(out.size, wav.size, ov_samples)
                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[-n:] = out[-n:] * (1.0 - ramp) + wav[:n] * ramp
                out = np.concatenate([out, wav[n:]])
            if end >= total_frames:
                break
            # 回退 overlap 帧推进，保证下一块与已拼接区域有共同覆盖且严格前进
            start = max(end - overlap_frames, start + 1)
        return out


def load_pupu_vocoder(checkpoint_dir=None, device=torch.device("cuda"),
                      dtype=torch.float32, chunk_frames=512, overlap_frames=64):
    """
    Build PupuVocoder from the local AFGen source tree and load the released
    generator weights (model.safetensors). Returns a PupuVocoderWrapper.
    """
    checkpoint_dir = Path(checkpoint_dir or PUPO_CKPT_DIR)
    ckpt_file = checkpoint_dir / "model.safetensors"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Pupu-Vocoder weights not found: {ckpt_file}")

    with _isolated_pupu_imports():
        module = importlib.import_module("models.vocoders.gan.generator.pupuvocoder")
        generator_cls = module.PupuVocoder

    cfg = _build_pupu_cfg()
    generator = generator_cls(cfg)
    state_dict = load_safetensors(ckpt_file)
    result = generator.load_state_dict(state_dict, strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    if missing or unexpected:
        raise RuntimeError(
            f"Pupu-Vocoder weight mismatch. missing={missing[:8]} "
            f"unexpected={unexpected[:8]}")
    generator.eval()
    generator = generator.to(device)  # 权重始终以 fp32 驻留，精度切换由 wrapper 的 autocast 负责

    print(f'[Pupu-Vocoder] loaded generator from {ckpt_file} '
          f'(compute dtype={dtype}, chunk={chunk_frames} frames, overlap={overlap_frames} frames)')
    return PupuVocoderWrapper(generator, cfg, device, dtype=dtype, chunk_frames=chunk_frames, overlap_frames=overlap_frames)

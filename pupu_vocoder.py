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
import os
import struct
import sys
from pathlib import Path

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
    saved_path = list(sys.path)
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
    """mel (batch, n_mel, frames) -> waveform numpy array @ 44.1 kHz."""

    def __init__(self, generator, cfg, device):
        self.generator = generator
        self.cfg = cfg
        self.device = device
        self.sample_rate = cfg.preprocess.sample_rate

    @torch.no_grad()
    def mel_to_wav(self, mel):
        """mel: (1, n_mel, frames) tensor (ln-compressed, HiFiGAN convention).
        Returns 1-D numpy waveform."""
        mel = mel.float().to(self.device)
        wav = self.generator(mel)
        return wav.squeeze().cpu().numpy()


def load_pupu_vocoder(checkpoint_dir=None, device=torch.device("cuda")):
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
        generator_cls = getattr(module, "PupuVocoder")

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
    generator = generator.to(device)

    print(f'[Pupu-Vocoder] loaded generator from {ckpt_file}')
    return PupuVocoderWrapper(generator, cfg, device)

"""
PC-NSF-HiFiGAN: pitch-controllable NSF-HiFiGAN vocoder.

Self-contained loader / generator adapted from openvpi/DiffSinger
(`modules/nsf_hifigan/models.py`, Apache-2.0). Compatible with the released
openvpi vocoder checkpoints, e.g.:

    pretrain/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/
        config.json      # architecture + mel/spectrogram parameters
        model.ckpt       # {'generator': state_dict}

Mel convention expected by the generator (matches DiffSinger training):
natural-log compressed mel spectrogram (`ln(clamp(mag, 1e-5))`) with
n_fft=2048, win_size=2048, hop_size=512, num_mels=128, fmin=40, fmax=16000,
sampling_rate=44100 (values are read from the checkpoint's config.json).

The F0 condition is a per-mel-frame fundamental frequency trajectory in Hz,
shape (batch, n_frames).
"""

import contextlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class AttrDict(dict):
    """Dictionary with attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(torch.nn.Module):
    """Definition of sine generator."""

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        """f0: (batchsize, length, dim), dim = fundamental + overtones."""
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1))
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    @torch.no_grad()
    def forward(self, f0, upp):
        """input f0: tensor(batchsize=1, length, dim=1); unvoiced steps = 0."""
        f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf."""

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.mini_nsf = h.mini_nsf
        # Older releases ship configs without `noise_sigma`; default to no noise.
        self.noise_sigma = h.get('noise_sigma', None)

        if h.mini_nsf:
            self.source_sr = h.sampling_rate / int(np.prod(h.upsample_rates[2:]))
            self.upp = int(np.prod(h.upsample_rates[:2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(np.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(
                sampling_rate=h.sampling_rate,
                harmonic_num=8
            )
            self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        ch = h.upsample_initial_channel
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2)))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
            if not h.mini_nsf:
                if i + 1 < len(h.upsample_rates):
                    stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                    self.noise_convs.append(Conv1d(
                        1, ch, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
                else:
                    self.noise_convs.append(Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = Conv1d(1, ch, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def fastsinegen(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        sines = torch.sin(2 * np.pi * rad)
        return sines

    def forward(self, x, f0, har_source=None):
        """
        x:  (batch, num_mels, frames)  natural-log compressed mel
        f0: (batch, frames)            fundamental frequency in Hz
        har_source: 可选的预生成源激励，(batch, 1, frames * upp)。分块推理时由外部
            在全长 F0 上一次性生成后切片传入：fastsinegen 的相位是跨帧全局累积
            （cumsum）的，逐块重算会丢失相位连续性导致接缝处谐波失真。
        returns: (batch, 1, frames * hop_size)
        """
        if har_source is None:
            if self.mini_nsf:
                har_source = self.fastsinegen(f0)
            else:
                har_source = self.m_source(f0.unsqueeze(-1), self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if self.noise_sigma is not None and self.noise_sigma > 0:
            x += self.noise_sigma * torch.randn_like(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if not self.mini_nsf:
                x_source = self.noise_convs[i](har_source)
                x = x + x_source
            elif i == 1:
                x_source = self.source_conv(har_source)
                x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# ----------------------------------------------------------------------------
# Mel utilities (natural-log compression, HiFiGAN-style reflect padding --
# identical math to the project's modules.audio.mel_spectrogram, but bound to
# the vocoder's own config.json parameters).
# ----------------------------------------------------------------------------

_MEL_BASIS = {}
_HANN_WINDOWS = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def extract_mel_for_pcnsf(y, mel_fn_args):
    """y: (batch, samples) waveform in [-1, 1] -> (batch, num_mels, frames)."""
    global _MEL_BASIS, _HANN_WINDOWS
    n_fft = mel_fn_args["n_fft"]
    num_mels = mel_fn_args["num_mels"]
    sampling_rate = mel_fn_args["sampling_rate"]
    hop_size = mel_fn_args["hop_size"]
    win_size = mel_fn_args["win_size"]
    fmin = mel_fn_args.get("fmin", 0)
    fmax = mel_fn_args.get("fmax")

    key = f"{fmin}_{fmax}_{y.device}"
    if key not in _MEL_BASIS:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _MEL_BASIS[key] = torch.from_numpy(mel).float().to(y.device)
        _HANN_WINDOWS[y.device] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=_HANN_WINDOWS[y.device], center=mel_fn_args.get("center", False),
        pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(_MEL_BASIS[key], spec)
    mel_spec = dynamic_range_compression_torch(mel_spec)
    return mel_spec


class PCNSFHiFiGAN:
    """Thin inference wrapper: mel + explicit F0 -> waveform.

    显存策略与 pupu_vocoder.PupuVocoderWrapper 一致：分块推理（峰值显存与音频
    总长解耦）+ autocast 半精度（权重仅 ~50MB，激活值才是大头；不直接
    ``.half()`` 权重以避免零初始化缓冲区的 dtype 不匹配问题）。差异点：
    NSF 的源激励含全局相位累积，必须先在全长 F0 上生成一次再按块切片。
    """

    def __init__(self, generator, h, device, dtype=torch.float32,
                 chunk_frames=512, overlap_frames=64):
        self.generator = generator
        self.h = h
        self.device = device
        self.sample_rate = h.sampling_rate
        self.hop_length = int(h.get('hop_size', h.sampling_rate // 86))
        self.mel_fn_args = {
            "n_fft": h.n_fft,
            "win_size": h.win_size,
            "hop_size": h.hop_size,
            "num_mels": h.num_mels,
            "sampling_rate": h.sampling_rate,
            "fmin": h.fmin,
            "fmax": h.fmax,
            "center": False,
        }
        self.dtype = dtype                # 计算精度（权重保持 fp32，靠 autocast 降激活精度）
        self.chunk_frames = chunk_frames  # 分块推理的 mel 帧数；<=0 表示整段前向
        self.overlap_frames = overlap_frames  # 块间重叠帧数（交叉淡化区）

    def _amp_ctx(self):
        """fp16/bf16 时返回 CUDA autocast 上下文；CPU 上强制回退 fp32。"""
        if self.device.type == 'cuda' and self.dtype != torch.float32:
            return torch.autocast('cuda', dtype=self.dtype)
        return contextlib.nullcontext()


    @torch.no_grad()
    def wave_to_mel(self, wave, device=None):
        """wave: 1-D numpy array or (1, samples) tensor -> (1, num_mels, frames)."""
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave).float()
        wave = wave.to(device or self.device)
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        return extract_mel_for_pcnsf(wave.float(), self.mel_fn_args)

    @torch.no_grad()
    def mel_to_wav(self, mel, f0, chunk_frames=None, overlap_frames=None):
        """
        mel: (1, num_mels, frames) natural-log compressed mel (on any device)
        f0:  (frames,) numpy array or tensor of F0 in Hz, aligned to mel frames
        returns: 1-D numpy waveform at self.sample_rate

        chunk_frames / overlap_frames 不传时使用加载时的默认配置；短音频或
        ``chunk_frames <= 0`` 时整段前向。
        """
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float()
        mel = mel.float().to(self.device)
        # 源激励在 fp32 下计算（autocast 会自动按需转换给各卷积层），
        # 避免 fp16 下相位累积的精度损失
        f0 = f0.float().to(self.device).reshape(1, -1)
        assert f0.shape[1] == mel.shape[2], (
            f"F0 frames ({f0.shape[1]}) != mel frames ({mel.shape[2]})")

        chunk_frames = self.chunk_frames if chunk_frames is None else chunk_frames
        overlap_frames = self.overlap_frames if overlap_frames is None else overlap_frames
        total_frames = mel.shape[2]

        # 短音频或显式关闭分块：整段一次前向
        if chunk_frames is None or chunk_frames <= 0 or total_frames <= chunk_frames:
            with self._amp_ctx():
                wav = self.generator(mel, f0)
            return wav.squeeze().float().cpu().numpy()

        g = self.generator
        upp = g.upp  # 源激励相对 mel 帧的上采样倍率（mini NSF 与常规 NSF 一致）
        # 谐波源激励一次性全长生成：(1, 1, frames*upp)，几十 MB 级别，开销可忽略；
        # 保证跨块的绝对相位连续，接缝处才不会出现谐波突变
        with torch.no_grad():
            if g.mini_nsf:
                har_source = g.fastsinegen(f0)
            else:
                har_source = g.m_source(f0.unsqueeze(-1), g.upp).transpose(1, 2)

        out = None
        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            # 末段剩余不足一个重叠区时并入当前块，避免产生过短末块
            if 0 < total_frames - end < overlap_frames:
                end = total_frames
            with self._amp_ctx():
                wav = g(mel[:, :, start:end], f0[:, start:end],
                        har_source=har_source[:, :, start * upp:end * upp])
            wav = wav.squeeze().float().cpu().numpy()
            if out is None:
                out = wav
            else:
                # 与上一块尾部做线性交叉淡化（重叠远超感受野，边界不可闻）
                n = min(out.size, wav.size, overlap_frames * self.hop_length)
                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[-n:] = out[-n:] * (1.0 - ramp) + wav[:n] * ramp
                out = np.concatenate([out, wav[n:]])
            if end >= total_frames:
                break
            # 回退 overlap 帧推进，保证下一块与已拼接区域有共同覆盖且严格前进
            start = max(end - overlap_frames, start + 1)
        return out


def find_checkpoint_file(vocoder_dir):
    """Pick the generator checkpoint: prefer `model.ckpt`, skip training dumps."""
    candidates = []
    for fname in sorted(os.listdir(vocoder_dir)):
        if not fname.endswith('.ckpt'):
            continue
        if fname == 'model.ckpt':
            return os.path.join(vocoder_dir, fname)
        if fname.startswith(('model_', 'model_full')):
            continue  # optimizer/discriminator dumps
        candidates.append(os.path.join(vocoder_dir, fname))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No *.ckpt generator checkpoint found in {vocoder_dir}")


def load_pc_nsf_hifigan(vocoder_dir, device=torch.device("cuda"),
                        dtype=torch.float32, chunk_frames=512, overlap_frames=64):
    """
    Build the PC-NSF-HiFiGAN generator from an openvpi release directory
    (config.json + model.ckpt) and return a PCNSFHiFiGAN wrapper.
    """
    config_file = os.path.join(vocoder_dir, 'config.json')
    with open(config_file, "r", encoding="utf-8") as f:
        h = AttrDict(json.load(f))

    ckpt_file = find_checkpoint_file(vocoder_dir)
    print(f'[PC-NSF-HiFiGAN] loading generator from {ckpt_file}')
    cp_dict = torch.load(ckpt_file, map_location='cpu')
    state_dict = cp_dict.get('generator', cp_dict)
    generator = Generator(h)
    generator.load_state_dict(state_dict)
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    generator = generator.to(device)  # 权重始终以 fp32 驻留，精度切换由 wrapper 的 autocast 负责
    print(f'[PC-NSF-HiFiGAN] compute dtype={dtype}, chunk={chunk_frames} frames, '
          f'overlap={overlap_frames} frames')
    return PCNSFHiFiGAN(generator, h, device, dtype=dtype,
                        chunk_frames=chunk_frames, overlap_frames=overlap_frames)

<div align="center">

# YingMusic-SVC · 双声码器推理管线

**免训练改造：Pupu-Vocoder + PC-NSF-HiFiGAN 两级合成 × 显式音高控制**

[简体中文](README.md) | [English（上游原版）](README_EN.md)

[![Paper](https://img.shields.io/badge/Paper-YingMusic--SVC-blue)](https://arxiv.org/abs/2512.04793)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-YingMusic--SVC-yellow)](https://huggingface.co/GiantAILab/YingMusic-SVC)
[![Demo Page](https://img.shields.io/badge/🎧%20Demo%20Page-YingMusic--SVC-brightgreen)](https://giantailab.github.io/YingMusic-SVC)

</div>

---

> 本仓库 fork 自 [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC)，在**不重新训练任何模型**的前提下重构了推理管线：
> 替换声码器、新增显式 F0（音高）控制，提升合成音质的同时支持变调。
>
> **唯一推理入口是 [`inference.py`](inference.py)**。上游的论文介绍、基准测试等内容见 [README_EN.md](README_EN.md)。

## 🎯 这个 fork 改了什么

| | 上游原版 | 本 fork |
|---|---|---|
| 声码器 | BigGAN（bigvgan） | **Pupu-Vocoder → PC-NSF-HiFiGAN** 两级合成 |
| 音高控制 | 仅模型内部隐式 F0 对齐 | 额外提供**显式 F0 轨迹**，支持半音/倍率变调 |
| 输出格式 | wav | flac（44.1 kHz），文件名自带参数元数据 |
| 显存占用 | — | 默认参数实测约 **5 GB** |

### 合成管线

```text
YingMusic-SVC (CFM) ──预测 Mel──▶ Pupu-Vocoder ──中间波形（内存直传，不落盘）──┐
                                                                              ▼
                                    按 PC-NSF-HiFiGAN 自身参数重提取 Mel（44.1kHz / hop512 / 128bin）
                                                                              +
原始输入音频 ──RMVPE（16kHz / 10ms 网格）──▶ 显式 F0 ──变调（--pitch-shift / --f0-scale）──┐
                                                                              ▼
                                                        PC-NSF-HiFiGAN ──▶ 最终 FLAC
```

**设计要点**

- 🎚️ **PC-NSF-HiFiGAN**：openvpi DiffSinger 社区声码器，采用 MiniNSF 结构，具备保共振峰的音高迁移能力，高频细节与真实感显著更好；
- 🎛️ **显式音高控制**：变调直接作用于送入声码器的 F0 条件，完全不改动模型权重——`--pitch-shift`（半音）与 `--f0-scale`（倍率）可叠加使用；
- 💧 **中间波形内存直传**：两级声码器均为 44.1 kHz，无需落盘临时文件；若日后换用其它采样率的 Pupu checkpoint，会自动在内存中重采样兜底；
- 🧠 **RMVPE 本地化**：优先加载本地 `pretrain/rmvpe/model.pt`；
- 🩺 **Preflight 自检**：启动前检查 Python 依赖与全部预训练权重，缺失则打印清单后退出（**绝不自动下载大文件**）；`--skip-check` 可跳过；
- 💾 **显存优化**：两个声码器均支持分块推理 + bf16/fp16 autocast 半精度（激活值减半，权重始终 fp32 驻留）。

新增模块：[`modules/pupu_vocoder.py`](modules/pupu_vocoder.py)（隔离导入 AFGen / Pupu-Vocoder 源码并加载 safetensors）、[`modules/pc_nsf_hifigan.py`](modules/pc_nsf_hifigan.py)（自包含 vendored openvpi 实现，checkpoint 键名 293/293 全匹配）。

## 🛠️ 安装

Python 3.10 + CUDA 环境，使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境与依赖：

```bash
# 创建虚拟环境
uv venv --python=3.10

# 先从 PyTorch 官方索引安装 CUDA 12.4 版本，更高版本应该也行吧，偷懒直接按上游的来了
uv pip install torch~=2.4.0 torchaudio~=2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 再安装其余依赖
uv pip install -r ./requirements.txt

# 拉取 SVC 主模型到 ./checkpoints
./.venv/Scripts/python.exe -c "from modelscope import snapshot_download; snapshot_download('giantailab/YingMusic-SVC', local_dir='./checkpoints')"
```

## ⬇️ 预训练权重

按下表下载并放置（目录不存在请自行创建）：

```text
YingMusic-SVC/
├── checkpoints/
│   └── YingMusic-SVC-full.pt
└── pretrain/
    ├── rmvpe/
    │   └── model.pt
    └── vocoder/
        ├── Pupu-Vocoder/experiments/pupuvocoder/checkpoint/
        │   └── epoch-0051_step-2553605_loss-62.135194/
        │       └── model.safetensors
        └── pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/
            ├── config.json
            └── model.ckpt
```

| 组件 | 下载来源 | 放置位置 |
|---|---|---|
| **YingMusic-SVC-full**（主模型） | [🤗 GiantAILab/YingMusic-SVC](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/YingMusic-SVC-full.pt)，直接用安装一节的 ModelScope 命令拉取 | `checkpoints/YingMusic-SVC-full.pt` |
| **Pupu-Vocoder**（一级声码器） | [🤗 spellbrush/AliasingFreeNeuralAudioSynthesis · pupuvocoder 目录](https://huggingface.co/spellbrush/AliasingFreeNeuralAudioSynthesis/tree/main/pupuvocoder)，下载其中的 safetensors 权重 | `pretrain/vocoder/Pupu-Vocoder/experiments/pupuvocoder/checkpoint/epoch-0051_step-2553605_loss-62.135194/model.safetensors` |
| **PC-NSF-HiFiGAN**（二级声码器） | [openvpi/vocoders · release 2025.02](https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02)，解压取 `config.json` 与 `model.ckpt` | `pretrain/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/` |
| **RMVPE**（F0 提取） | [🤗 Pur1zumu/RIFT-SVC-modules · rmvpe 目录](https://huggingface.co/Pur1zumu/RIFT-SVC-modules/tree/main/rmvpe)，下载 `model.pt` | `pretrain/rmvpe/model.pt` |

> 💡 除上表外，运行时若无本地缓存还会从 HuggingFace 拉取 `openai/whisper-small`（语义 tokenizer）与 `funasr/campplus`（音色编码器），请确保网络可达或提前缓存。
>
> 💡 不确定自己缺什么？随便跑一条下面的推理命令，preflight 会把缺失项一次性列全。

## 🚀 快速开始

最小示例（升 2 个半音）：

```bash
python inference.py --source vocal.wav --target timbre.wav --pitch-shift 2
```

默认配置（实测约 5 GB 显存）：

```bash
python inference.py \
  --source vocal.wav --target timbre.wav \
  --cuda 0 --fp16 true --config "./configs/YingMusic-SVC.yml" \
  --pitch-shift 1 --f0-scale 1 \
  --length-adjust 1 --inference-cfg-rate 0.7 --diffusion-steps 30 \
  --vocoder-dtype bf16 --vocoder-chunk 2048 --vocoder-overlap 64
```

**输出说明**

- 结果统一为 **FLAC**，写到源音频所在目录（可用 `--output` 指定其它目录）；
- 文件名形如 `vocals_YingMusic@+2key_10ks_2ref_1.2f0.flac`：`@+2key` 为变调量（恒写入），其余后缀仅在对应参数非默认时追加（ckpt 步数 / 参考段数 / F0 倍率 / 扩散步数）；
- 其他玩法：传入 `--accompany 伴奏.wav`，会在输出目录旁生成 `accompany/` 子目录，存放加回声+混响后的伴唱 remix 版本。（未测试，我都直接用MSST）

## 🎛️ 参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | （必填） | 源人声音频，支持 wav/flac/mp3 等 |
| `--target` | （必填） | 目标音色参考音频，**可传多段**，按顺序拼接后截断至 25 s |
| `--pitch-shift` | `0.0` | 显式 F0 变调，单位半音（如 `2.0` = 升 2 半音，`-12` = 降八度） |
| `--f0-scale` | `1.0` | 显式 F0 直接倍率缩放（如 `1.2 ≈ +3.2 半音`），叠加在 pitch-shift 之后 |
| `--semi-tone-shift` | `None` | 强制指定 SVC 模型内部自适应 F0 对齐的半音数；留空保持自动 |
| `--diffusion-steps` | `30` | CFM 扩散步数 |
| `--checkpoint` | `./checkpoints/YingMusic-SVC-full.pt` | SVC 模型权重路径 |
| `--config` | `./configs/YingMusic-SVC.yml` | 模型配置路径 |
| `--cuda` | `0` | GPU 编号 |
| `--fp16` | `True` | 主干网络 fp16 推理加速 |
| `--length-adjust` | `1.0` | 输出时长调节倍率 |
| `--inference-cfg-rate` | `0.7` | CFG 引导强度 |
| `--accompany` | `None` | 伴奏轨路径，启用回声+混响 remix |
| `--output` | `None` | 输出目录；留空写到源音频所在目录 |
| `--vocoder-dtype` | `bf16` | 声码器计算精度 `fp32`/`bf16`/`fp16`；如听到伪影可退回 `fp32` |
| `--vocoder-chunk` | `2048` | 分块推理的 mel 帧数（hop512@44.1kHz 下每帧约 11.6 ms）；`<=0` 表示整段前向 |
| `--vocoder-overlap` | `64` | 分块间重叠帧数（波形交叉淡化拼接区，需大于模型感受野） |
| `--skip-check` | 关闭 | 跳过 preflight 依赖/权重自检 |

## 🖥️ 图形界面（可选）：leaf-flow

不想敲命令行？来试试 [leaf-flow](https://github.com/Stareven233/leaf-flow) ——一个声明式 UI 框架，搭配仓库根目录的 `yingmusic-svc.yaml`，运行时自动生成对应的图形界面，本质上是对 `inference.py` 命令行参数的可视化封装，详见其仓库主页。

## 🙏 致谢

- 上游项目与模型：[GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC)（[arXiv:2512.04793](https://arxiv.org/abs/2512.04793)）
- 架构基础：[Seed-VC](https://github.com/Plachtaa/seed-vc)
- 双声码器推理管线思路：
  - [Da1sypetals/yingmusic-svc-mlx](https://github.com/Da1sypetals/yingmusic-svc-mlx)
  - [blog.petals.top 相关博文](https://blog.petals.top/learn/2026-08-11-ai-singer-and-problem-in-yingmusic/)
- 声码器与工具链：
  - [AFGen / Pupu-Vocoder](https://huggingface.co/spellbrush/AliasingFreeNeuralAudioSynthesis)（Spellbrush）
  - [PC-NSF-HiFiGAN](https://github.com/openvpi/vocoders)（openvpi DiffSinger 社区声码器项目）
  - [RMVPE](https://huggingface.co/Pur1zumu/RIFT-SVC-modules/tree/main/rmvpe)

<details>
<summary>📚 学术引用</summary>

学术场景使用请引用上游论文：

```bibtex
@article{chen2025yingmusicsvc,
  title={YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases},
  author={Chen, Gongyu and Zhang, Xiaoyu and Weng, Zhenqiang and Zheng, Junjie and Shen, Da and Ding, Chaofan and Zhang, Wei-Qiang and Chen, Zihao},
  journal={arXiv preprint arXiv:2512.04793},
  year={2025}
}
```

</details>

## 📝 许可证

MIT License（同上游）

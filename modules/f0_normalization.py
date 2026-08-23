import os
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import torch


def f0_normalize_and_synthesize(
    audio_batch: torch.Tensor,
    sample_rate: int = 44100,
    hop_length: int = 512,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
    frame_period: float = None,
    normalize_method: str = "mean_replace"  # 用浊音帧平均值替换所有浊音帧
) -> torch.Tensor:
    """
    使用WORLD vocoder提取音频基频，对浊音帧进行归一化处理，然后合成新音频
    
    Args:
        audio_batch (torch.Tensor): 输入音频，形状为 (B, T)，其中B是批次大小，T是时间步数
        sample_rate (int): 采样率，默认44100
        hop_length (int): 跳跃长度，默认512
        f0_min (float): 基频最小值，默认50Hz
        f0_max (float): 基频最大值，默认1100Hz
        frame_period (float): 帧周期（毫秒），如果为None则根据hop_length计算
        normalize_method (str): 处理方法，支持"mean_replace"（用浊音帧平均值替换）
        
    Returns:
        torch.Tensor: 处理后的音频，形状为 (B, T')，其中T'可能与输入略有不同
    """
    
    if frame_period is None:
        frame_period = 1000 * hop_length / sample_rate
    
    # 确保输入是numpy数组格式，WORLD vocoder需要double类型
    if isinstance(audio_batch, torch.Tensor):
        audio_np = audio_batch.cpu().numpy()
    else:
        audio_np = audio_batch
    
    # 存储处理后的音频
    processed_audios = []
    
    # 处理批次中的每个音频样本
    for i in range(audio_np.shape[0]):
        audio = audio_np[i].astype(np.double)
        
        # 使用WORLD vocoder提取声学特征
        # 1. 提取基频 (F0)
        f0, t = pw.dio(
            audio, 
            sample_rate, 
            f0_floor=f0_min,
            f0_ceil=f0_max,
            frame_period=frame_period
        )
        
        # 使用StoneMask改进基频估计
        f0 = pw.stonemask(audio, f0, t, sample_rate)
        
        # 2. 提取频谱包络 (Spectral Envelope)
        sp = pw.cheaptrick(audio, f0, t, sample_rate)
        
        # 3. 提取非周期性参数 (Aperiodicity)
        ap = pw.d4c(audio, f0, t, sample_rate)
        
        # 对浊音帧的基频进行处理（用平均值替换）
        f0_processed = normalize_f0(f0, method=normalize_method)

        # 使用处理后的基频合成新音频
        synthesized_audio = pw.synthesize(
            f0_processed, 
            sp, 
            ap, 
            sample_rate, 
            frame_period
        )
        
        processed_audios.append(synthesized_audio)
    
    # 将所有处理后的音频对齐到相同长度
    max_length = max(len(audio) for audio in processed_audios)
    aligned_audios = []
    
    for audio in processed_audios:
        if len(audio) < max_length:
            # 零填充到最大长度
            padded_audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
        else:
            # 截断到最大长度
            padded_audio = audio[:max_length]
        aligned_audios.append(padded_audio)
    
    # 转换回torch tensor
    result = torch.from_numpy(np.stack(aligned_audios)).float()
    
    return result


def normalize_f0(f0: np.ndarray, method: str = "mean_replace") -> np.ndarray:
    """
    对基频进行处理，用浊音帧的平均值替换所有浊音帧的原f0
    
    Args:
        f0 (np.ndarray): 原始基频序列
        method (str): 处理方法，目前只支持"mean_replace"
            
    Returns:
        np.ndarray: 处理后的基频序列
    """
    f0_processed = f0.copy()
    
    # 找到浊音帧（基频大于0的帧）
    voiced_mask = f0 > 0
    
    if not np.any(voiced_mask):
        # 如果没有浊音帧，返回原始f0
        return f0_processed
    
    voiced_f0 = f0[voiced_mask]
    
    if method == "mean_replace":
        # 计算浊音帧的平均值
        mean_f0 = np.mean(voiced_f0)
        # 用平均值替换所有浊音帧
        f0_processed[voiced_mask] = mean_f0
        
    else:
        raise ValueError(f"Unsupported method: {method}. Only 'mean_replace' is supported.")
    
    return f0_processed


def extract_f0_features(
    audio_batch: torch.Tensor,
    sample_rate: int = 44100,
    hop_length: int = 512,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
    frame_period: float = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    仅提取WORLD vocoder特征，不进行合成
    
    Args:
        audio_batch (torch.Tensor): 输入音频，形状为 (B, T)
        sample_rate (int): 采样率
        hop_length (int): 跳跃长度
        f0_min (float): 基频最小值
        f0_max (float): 基频最大值
        frame_period (float): 帧周期（毫秒）
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (f0, spectral_envelope, aperiodicity, time_axis)
    """
    
    if frame_period is None:
        frame_period = 1000 * hop_length / sample_rate
    
    if isinstance(audio_batch, torch.Tensor):
        audio_np = audio_batch.cpu().numpy()
    else:
        audio_np = audio_batch
    
    batch_f0 = []
    batch_sp = []
    batch_ap = []
    batch_t = []
    
    for i in range(audio_np.shape[0]):
        audio = audio_np[i].astype(np.double)
        
        # 提取WORLD特征
        f0, t = pw.dio(
            audio, 
            sample_rate, 
            f0_floor=f0_min,
            f0_ceil=f0_max,
            frame_period=frame_period
        )
        f0 = pw.stonemask(audio, f0, t, sample_rate)
        sp = pw.cheaptrick(audio, f0, t, sample_rate)
        ap = pw.d4c(audio, f0, t, sample_rate)
        
        batch_f0.append(f0)
        batch_sp.append(sp)
        batch_ap.append(ap)
        batch_t.append(t)
    
    return np.array(batch_f0), np.array(batch_sp), np.array(batch_ap), np.array(batch_t)


# 示例使用函数
def demo_usage():
    """
    演示如何使用f0_normalize_and_synthesize函数
    """
    # 创建示例音频数据 (批次大小=2, 时间步数=22050*2)
    sample_rate = 44100
    duration = 2.0  # 2秒
    batch_size = 2
    
    # 生成示例音频（正弦波）
    t = torch.linspace(0, duration, int(sample_rate * duration))
    freq1 = 440  # A4音符
    freq2 = 523  # C5音符
    
    audio1 = torch.sin(2 * np.pi * freq1 * t)
    audio2 = torch.sin(2 * np.pi * freq2 * t)
    audio_batch = torch.stack([audio1, audio2])  # 形状: (2, T)
    
    print(f"输入音频形状: {audio_batch.shape}")
    
    # 处理音频
    processed_audio = f0_normalize_and_synthesize(
        audio_batch=audio_batch,
        sample_rate=sample_rate,
        normalize_method="mean_replace"
    )
    
    print(f"输出音频形状: {processed_audio.shape}")
    
    return processed_audio


def demo_save_audio(
    txt_file_path: str = "/user-fs/chenzihao/chengongyu/svc/seed-vc/test/dataset/debug.txt",
    output_dir: str | Path = "/user-fs/chenzihao/chengongyu/svc/seed-vc/highfreq_f0_pred/strange/f0normal",
    sample_rate: int = 22050,
    normalize_method: str = "mean_replace",
    max_duration: float = 60.0
) -> list[str]:
    """
    从txt文件读取音频路径，进行F0归一化处理并保存
    
    Args:
        txt_file_path (str): 包含音频路径的txt文件，每行一个路径
        output_dir (Union[str, Path]): 输出目录
        sample_rate (int): 目标采样率
        normalize_method (str): 归一化方法
        max_duration (float): 最大音频时长（秒），超过会被截断
        
    Returns:
        List[str]: 保存的文件路径列表
    """
    
    # 读取音频路径
    if not os.path.exists(txt_file_path):
        print(f"警告: 文件 {txt_file_path} 不存在，使用示例数据")
        return demo_with_synthetic_audio(output_dir, sample_rate, normalize_method)
    
    audio_paths = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()
            if path and os.path.exists(path):
                audio_paths.append(path)
            elif path:
                print(f"警告: 音频文件不存在: {path}")
    
    if not audio_paths:
        print("没有找到有效的音频文件，使用示例数据")
        return demo_with_synthetic_audio(output_dir, sample_rate, normalize_method)
    
    print(f"找到 {len(audio_paths)} 个有效音频文件")
    
    # 加载音频数据
    audio_tensors = []
    filenames = []
    
    for path in audio_paths:
        try:
            # 使用soundfile加载音频
            audio_data, orig_sr = sf.read(path)
            
            # 如果是立体声，取第一个通道
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # 重采样到目标采样率（简单的线性插值）
            if orig_sr != sample_rate:
                # 计算重采样比例
                resample_ratio = sample_rate / orig_sr
                new_length = int(len(audio_data) * resample_ratio)
                # 简单线性插值重采样
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
            
            # 限制最大时长
            max_samples = int(max_duration * sample_rate)
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            # 转换为tensor
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            audio_tensors.append(audio_tensor)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(path))[0]
            filenames.append(f"{base_name}_f0norm")
            
        except Exception as e:
            print(f"加载音频文件 {path} 时出错: {e}")
            continue
    
    if not audio_tensors:
        print("没有成功加载任何音频文件")
        return []
    
    # 将所有音频填充到相同长度
    max_length = max(len(tensor) for tensor in audio_tensors)
    padded_tensors = []
    
    for tensor in audio_tensors:
        if len(tensor) < max_length:
            padded = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)))
        else:
            padded = tensor[:max_length]
        padded_tensors.append(padded)
    
    # 创建batch
    audio_batch = torch.stack(padded_tensors)
    print(f"创建音频batch，形状: {audio_batch.shape}")
    
    # 进行F0归一化处理
    print("开始F0归一化处理...")
    processed_batch = f0_normalize_and_synthesize(
        audio_batch=audio_batch,
        sample_rate=sample_rate,
        normalize_method=normalize_method
    )
    print(f"处理完成，输出形状: {processed_batch.shape}")
    
    # 保存处理后的音频
    saved_paths = save_batch_audio(
        audio_batch=processed_batch,
        filenames=filenames,
        output_dir=output_dir,
        sample_rate=sample_rate
    )
    
    return saved_paths


def demo_with_synthetic_audio(
    output_dir: str | Path = "output/f0_normalized",
    sample_rate: int = 44100,
    normalize_method: str = "mean_replace"
) -> list[str]:
    """
    使用合成音频进行演示
    """
    print("使用合成音频进行F0归一化演示...")
    
    # 生成示例音频
    duration = 3.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # 创建不同频率的正弦波
    freqs = [220, 330, 440, 523]  # A3, E4, A4, C5
    audio_tensors = []
    filenames = []
    
    for i, freq in enumerate(freqs):
        # 添加一些频率调制使其更像真实语音
        freq_mod = freq * (1 + 0.1 * torch.sin(2 * np.pi * 2 * t))  # 2Hz调制
        audio = 0.5 * torch.sin(2 * np.pi * freq_mod * t)
        audio_tensors.append(audio)
        filenames.append(f"synthetic_{freq}hz")
    
    # 创建batch
    audio_batch = torch.stack(audio_tensors)
    print(f"合成音频batch形状: {audio_batch.shape}")
    
    # 进行F0归一化处理
    processed_batch = f0_normalize_and_synthesize(
        audio_batch=audio_batch,
        sample_rate=sample_rate,
        normalize_method=normalize_method
    )
    
    # 保存处理后的音频
    saved_paths = save_batch_audio(
        audio_batch=processed_batch,
        filenames=filenames,
        output_dir=output_dir,
        sample_rate=sample_rate
    )
    
    return saved_paths


def save_batch_audio(
    audio_batch: torch.Tensor,
    filenames: list[str],
    output_dir: str | Path = "output/f0_normalized",
    sample_rate: int = 44100,
    audio_format: str = "wav",
    create_dir: bool = True
) -> list[str]:
    """
    将batch音频保存到指定文件夹
    
    Args:
        audio_batch (torch.Tensor): 音频批次，形状为 (B, T)
        output_dir (Union[str, Path]): 输出文件夹路径
        filenames (List[str]): 文件名列表，长度应与batch_size相同
        sample_rate (int): 采样率，默认44100
        audio_format (str): 音频格式，支持 "wav", "flac", "mp3" 等
        create_dir (bool): 如果输出目录不存在是否创建，默认True
        
    Returns:
        List[str]: 保存的文件路径列表
        
    Raises:
        ValueError: 当filenames长度与batch_size不匹配时
        OSError: 当无法创建输出目录时
    """
    
    # 转换为Path对象
    output_dir = Path(output_dir)
    
    # 创建输出目录
    if create_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.exists():
        raise OSError(f"输出目录不存在: {output_dir}")
    
    # 检查输入参数
    if isinstance(audio_batch, torch.Tensor):
        audio_np = audio_batch.cpu().numpy()
    else:
        audio_np = np.array(audio_batch)
    
    batch_size = audio_np.shape[0]
    
    if len(filenames) != batch_size:
        raise ValueError(f"文件名数量({len(filenames)})与batch大小({batch_size})不匹配")
    
    # 保存每个音频文件
    saved_paths = []
    
    for i in range(batch_size):
        # 获取当前音频数据
        audio_data = audio_np[i]
        
        # 处理文件名和扩展名
        filename = filenames[i]
        if not filename.endswith(f".{audio_format}"):
            filename = f"{filename}.{audio_format}"
        
        # 完整的文件路径
        file_path = output_dir / filename
        
        # 保存音频文件
        try:
            # 确保音频数据在合理范围内
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            sf.write(
                file_path, 
                audio_data, 
                sample_rate, 
                format=audio_format.upper()
            )
            
            saved_paths.append(str(file_path))
            print(f"已保存: {file_path}")
            
        except Exception as e:
            print(f"保存文件 {file_path} 时出错: {e}")
            continue
    
    return saved_paths



if __name__ == "__main__":
    # 运行演示
    result = demo_usage()
    print("F0处理完成！")
    
    # 运行音频保存演示
    paths = demo_save_audio()
    print("音频保存演示完成！")

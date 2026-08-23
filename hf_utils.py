'''
uv run hf_utils.py --repo-id spellbrush/AliasingFreeNeuralAudioSynthesis --target-folder pupuvocoder --local-dir ./pretrain
'''

import argparse
import os

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="./checkpoints")
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="./checkpoints")

    return model_path, config_path


def download():
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="下载HuggingFace仓库中的指定文件夹")
    parser.add_argument("--repo-id", type=str, required=True, 
                        help="HuggingFace仓库ID，格式为'用户名/仓库名'，例如'spellbrush/AliasingFreeNeuralAudioSynthesis'")
    parser.add_argument("--target-folder", type=str, required=True, 
                        help="要下载的仓库内文件夹路径，例如根目录的'model'、子文件夹'examples/audio'，下载根目录则留空")
    parser.add_argument("--local-dir", type=str, default="./huggingface_downloads", 
                        help="文件本地保存路径，默认当前目录下的huggingface_downloads文件夹")
    parser.add_argument("--token", type=str, default=None, 
                        help="HuggingFace访问令牌，私有仓库需要填写，公共仓库可留空")
    parser.add_argument("--repo-type", type=str, default="model", choices=["model", "dataset", "space"], 
                        help="仓库类型，可选model/dataset/space，默认model")
    parser.add_argument("--max-workers", type=int, default=8, 
                        help="多线程下载的线程数，默认8，可根据网络情况调整")
    parser.add_argument("--no-recursive", action="store_true", 
                        help="指定后仅下载目标文件夹下的直接文件，不下载子文件夹内容")
    
    # 解析命令行参数
    args = parser.parse_args()
    # 处理文件夹路径末尾的斜杠，避免匹配异常
    target_folder = args.target_folder.rstrip("/")

    try:
        # 1. 验证目标文件夹是否存在
        print(f"正在检查仓库 {args.repo_id} 中是否存在文件夹 {target_folder or '根目录'}...")
        all_repo_files = list_repo_files(
            repo_id=args.repo_id, 
            token=args.token, 
            repo_type=args.repo_type
        )
        # 筛选目标文件夹下的文件（根目录下载时匹配所有文件）
        if target_folder:
            target_files = [f for f in all_repo_files if f.startswith(target_folder + "/")]
        else:
            target_files = all_repo_files
        
        if not target_files:
            raise ValueError(f"未找到路径为 '{target_folder}' 的文件夹，请检查路径是否正确\n仓库根目录文件列表前10个：{all_repo_files[:10]}")

        # 2. 配置下载匹配规则
        if args.no_recursive:
            # 仅下载当前文件夹的直接文件
            allow_patterns = [f"{target_folder}/*"] if target_folder else ["*"]
        else:
            # 递归下载所有子文件夹内容
            allow_patterns = [f"{target_folder}/**/*"] if target_folder else ["**/*"]

        # 3. 执行下载
        print(f"开始下载，{'仅下载直接文件' if args.no_recursive else '递归下载所有子文件夹'}...")
        downloaded_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
            allow_patterns=allow_patterns,
            token=args.token,
            max_workers=args.max_workers
        )

        # 4. 输出结果
        if target_folder:
            print(f"\n✅ 下载完成！目标文件夹保存在：{downloaded_path}/{target_folder}")
        else:
            print(f"\n✅ 下载完成！文件保存在：{downloaded_path}")

    except Exception as e:
        print(f"\n❌ 操作失败：{e}")
        print("请检查：1. 仓库ID和文件夹路径是否正确 2. 网络是否正常 3. 私有仓库是否填写了正确的Token")

if __name__ == "__main__":
    download()

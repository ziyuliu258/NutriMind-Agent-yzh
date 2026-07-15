#!/usr/bin/env python3
"""
AutoDL YOLO26 训练脚本 — NutriMind-Agent Phase 1.2

针对 RTX 3080 Ti (12GB VRAM) / 90GB RAM / 12 核 CPU 深度优化。

核心优化:
  - batch=32 启动，OOM 时自动降级到 16/8/4
  - cache='ram' 利用 90GB 大内存缓存全部图片（极限 I/O 加速）
  - workers=8 充分利用 12 核 CPU
  - amp=True 自动混合精度（节省显存 + 加速）
  - patience=10 早停（10 轮无提升自动结束，省算力费）
  - yolo26s.pt 预训练权重（轻量，适合密集食物检测）

用法:
  # 基本训练
  python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml

  # 自定义参数
  python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml \
      --model yolo26s --epochs 100 --batch 32

  # 后台运行（防止 SSH 断开）
  nohup python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml > train.log 2>&1 &

  # 使用 screen
  screen -S train
  python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml
  # Ctrl+A, D 分离; screen -r train 恢复

环境依赖:
  pip install ultralytics>=8.3.0
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional


# ==================================================================
# 默认配置（针对 RTX 3080 Ti + 90GB RAM 优化）
# ==================================================================

class Config:
    """训练超参数默认值。"""
    MODEL_NAME = "yolo26s"          # YOLO26 Small（轻量，适合密集食物检测）
    EPOCHS = 100                     # 训练轮数
    IMG_SIZE = 640                   # 输入图像尺寸
    BATCH = 32                       # 初始 batch size（OOM 时自动降级）
    WORKERS = 8                      # 数据加载线程数（12 核 CPU 的 2/3）
    PATIENCE = 10                    # 早停轮数
    CACHE = "ram"                    # 缓存策略（90GB 内存充足）
    AMP = True                       # 自动混合精度
    DEVICE = 0                       # GPU 设备编号
    OUTPUT_NAME = "yolo26_food_best.pt"  # 最终输出模型名
    MODEL_OUTPUT_DIR = None          # 模型输出目录（运行时计算）


# 备用 batch size 列表（从大到小尝试）
BATCH_SIZE_FALLBACKS = [32, 24, 16, 12, 8, 4, 2]


# ==================================================================
# 环境检查
# ==================================================================

def check_environment() -> dict:
    """检查训练环境是否就绪。

    Returns:
        环境信息字典
    """
    info = {
        "ultralytics_available": False,
        "cuda_available": False,
        "gpu_name": "N/A",
        "vram_gb": 0,
        "ram_gb": 0,
        "python_version": sys.version,
    }

    print("=" * 60)
    print("🔍 环境检查")
    print("=" * 60)

    # 1. ultralytics
    try:
        import ultralytics
        info["ultralytics_available"] = True
        print(f"✅ ultralytics {ultralytics.__version__}")
    except ImportError:
        print("❌ ultralytics 未安装！")
        print("   请运行: pip install ultralytics>=8.3.0")
        return info

    # 2. CUDA
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"✅ CUDA 可用")
            print(f"   GPU: {info['gpu_name']}")
            print(f"   VRAM: {info['vram_gb']:.1f} GB")
        else:
            print("⚠️ CUDA 不可用，将使用 CPU 训练（极慢，仅用于测试）")
    except ImportError:
        print("⚠️ torch 未安装（ultralytics 会自动安装）")

    # 3. 系统内存
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
        print(f"✅ 系统内存: {info['ram_gb']:.1f} GB")
    except ImportError:
        print("⚠️ psutil 未安装，无法检测内存")
        # 尝试用 /proc/meminfo 读取（Linux）
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        info["ram_gb"] = int(line.split()[1]) / (1024**2)
                        print(f"✅ 系统内存: {info['ram_gb']:.1f} GB (from /proc/meminfo)")
                        break
        except Exception:
            pass

    # 4. 数据盘
    autodl_tmp = Path("/root/autodl-tmp")
    if autodl_tmp.exists():
        try:
            usage = shutil.disk_usage(autodl_tmp)
            print(f"✅ 数据盘: {autodl_tmp}")
            print(f"   可用空间: {usage.free / (1024**3):.1f} GB")
        except Exception:
            pass

    print("=" * 60)
    return info


# ==================================================================
# 训练主逻辑
# ==================================================================

def run_training(
    data_yaml: str,
    model_name: str = Config.MODEL_NAME,
    epochs: int = Config.EPOCHS,
    img_size: int = Config.IMG_SIZE,
    batch: int = Config.BATCH,
    workers: int = Config.WORKERS,
    patience: int = Config.PATIENCE,
    cache: str = Config.CACHE,
    amp: bool = Config.AMP,
    device: int = Config.DEVICE,
    model_output_dir: Optional[Path] = None,
) -> dict:
    """执行 YOLO26 训练。

    包含自动 batch size 降级机制：
    如果 CUDA OOM，依次尝试更小的 batch size。

    Args:
        data_yaml: data.yaml 配置文件路径
        model_name: 预训练模型名（如 yolo26s）
        epochs: 训练轮数
        img_size: 输入图像尺寸
        batch: 初始 batch size
        workers: 数据加载线程数
        patience: 早停轮数
        cache: 缓存策略（'ram' 或 False）
        amp: 自动混合精度
        device: GPU 设备编号
        model_output_dir: 模型输出目录

    Returns:
        训练结果字典（best.pt 路径 + 关键指标）
    """
    from ultralytics import YOLO

    # 验证 data.yaml
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ data.yaml 不存在: {data_yaml}")
        sys.exit(1)

    # 模型输出目录
    if model_output_dir is None:
        model_output_dir = (
            Path(__file__).resolve().parent / "data" / "models"
        )
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # 预训练权重
    pretrained = f"{model_name}.pt"
    print(f"\n📥 预训练权重: {pretrained}")

    # 打印配置摘要
    print()
    print("=" * 60)
    print("⚙️ 训练配置")
    print("=" * 60)
    print(f"   模型: {model_name}")
    print(f"   预训练权重: {pretrained}")
    print(f"   数据配置: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {img_size}")
    print(f"   Batch Size: {batch} (初始，OOM时自动降级)")
    print(f"   Workers: {workers}")
    print(f"   Cache: {cache}")
    print(f"   AMP: {amp}")
    print(f"   Early Stopping: patience={patience}")
    print(f"   模型输出: {model_output_dir / Config.OUTPUT_NAME}")
    print("=" * 60)

    # 自动 batch size 降级训练
    batch_sizes_to_try = [batch] + [b for b in BATCH_SIZE_FALLBACKS if b < batch]

    for attempt, bs in enumerate(batch_sizes_to_try):
        try:
            print(f"\n{'='*60}")
            if attempt > 0:
                print(f"🔄 降级重试: batch={bs} (前一次 batch={batch_sizes_to_try[attempt-1]} OOM)")
            print(f"🚀 开始训练 (batch={bs})")
            print(f"{'='*60}")
            start_time = time.time()

            # 加载模型
            model = YOLO(pretrained)

            # 执行训练
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=bs,
                workers=workers,
                patience=patience,
                cache=cache,
                amp=amp,
                device=device,
                exist_ok=True,        # 覆盖已存在的输出目录
                pretrained=True,
                verbose=True,

                # 数据增强（轻量增强，避免过拟合）
                hsv_h=0.015,          # HSV 色调扰动
                hsv_s=0.7,            # HSV 饱和度扰动
                hsv_v=0.4,            # HSV 明度扰动
                degrees=5.0,          # 旋转角度
                translate=0.1,        # 平移比例
                scale=0.3,            # 缩放比例
                shear=2.0,            # 剪切角度
                perspective=0.0,      # 透视变换
                flipud=0.0,           # 上下翻转概率
                fliplr=0.5,           # 左右翻转概率
                mosaic=0.5,           # mosaic 增强概率（密集检测友好）
                mixup=0.1,            # mixup 增强概率
            )

            # 训练成功
            elapsed = time.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)

            print(f"\n{'='*60}")
            print(f"✅ 训练完成！")
            print(f"{'='*60}")
            print(f"   总耗时: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print(f"   最终 batch size: {bs}")

            # 提取指标
            metrics = extract_metrics(results)
            print_metrics(metrics)

            # 复制最佳权重到项目模型目录
            best_source = _find_best_pt(results)
            best_dest = model_output_dir / Config.OUTPUT_NAME
            if best_source and best_source.exists():
                shutil.copy2(best_source, best_dest)
                print(f"\n📦 模型已同步到项目目录:")
                print(f"   {best_dest}")
                print(f"   文件大小: {best_dest.stat().st_size / (1024**2):.1f} MB")

            metrics["output_model"] = str(best_dest)
            metrics["batch_used"] = bs
            metrics["training_time_seconds"] = round(elapsed, 1)

            return metrics

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "OOM" in error_msg:
                print(f"\n⚠️ CUDA OOM (batch={bs}): 显存不足")
                if attempt < len(batch_sizes_to_try) - 1:
                    next_bs = batch_sizes_to_try[attempt + 1]
                    print(f"   → 自动降级至 batch={next_bs}，正在重试...")
                    # 清理 GPU 缓存
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    time.sleep(3)
                    continue

            # 非 OOM 错误，直接抛出
            print(f"\n❌ 训练失败: {error_msg}")
            raise

    # 所有 batch size 都失败了
    print(f"\n❌ 所有 batch size 均失败（{batch_sizes_to_try}），请尝试:")
    print(f"   1. 减小 img_size（当前 {img_size}）")
    print(f"   2. 关闭 cache='ram' 释放系统内存")
    print(f"   3. 使用更小的模型（yolo26n）")
    sys.exit(1)


# ==================================================================
# 指标提取与展示
# ==================================================================

def extract_metrics(results) -> dict:
    """从 ultralytics 训练结果中提取关键指标。

    Args:
        results: YOLO.train() 返回值

    Returns:
        指标字典
    """
    metrics = {}

    try:
        if hasattr(results, "results_dict"):
            rd = results.results_dict
            metrics["mAP50"] = float(rd.get("metrics/mAP50(B)", 0))
            metrics["mAP50-95"] = float(rd.get("metrics/mAP50-95(B)", 0))
            metrics["precision"] = float(rd.get("metrics/precision(B)", 0))
            metrics["recall"] = float(rd.get("metrics/recall(B)", 0))
    except Exception:
        # 兼容旧版 ultralytics
        pass

    # 从 saved_dir 读取 results.csv
    if hasattr(results, "save_dir"):
        csv_path = Path(results.save_dir) / "results.csv"
        if csv_path.exists():
            metrics["results_csv"] = str(csv_path)

    metrics["save_dir"] = (
        str(results.save_dir) if hasattr(results, "save_dir") else "N/A"
    )

    return metrics


def _find_best_pt(results) -> Optional[Path]:
    """查找训练产出的 best.pt 文件。

    Args:
        results: YOLO.train() 返回值

    Returns:
        best.pt 路径或 None
    """
    if hasattr(results, "save_dir"):
        best_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_path.exists():
            return best_path

    # 回退：在 runs/ 下查找最新的 best.pt
    runs_dir = Path.cwd() / "runs" / "detect"
    if runs_dir.exists():
        train_dirs = sorted(
            runs_dir.glob("train*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in train_dirs:
            best_path = d / "weights" / "best.pt"
            if best_path.exists():
                return best_path

    return None


def print_metrics(metrics: dict) -> None:
    """格式化打印训练指标。"""
    print(f"\n{'='*60}")
    print("📊 最终训练指标")
    print("=" * 60)

    keys_labels = [
        ("mAP50", "mAP@50"),
        ("mAP50-95", "mAP@50-95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]

    for key, label in keys_labels:
        if key in metrics and metrics[key] is not None:
            print(f"   {label:12s}: {metrics[key]:.4f}")

    if "training_time_seconds" in metrics:
        t = metrics["training_time_seconds"]
        h, m = divmod(t, 3600)
        m, s = divmod(m, 60)
        print(f"   {'训练耗时':12s}: {int(h)}h {int(m)}m {int(s)}s")

    if "batch_used" in metrics:
        print(f"   {'Batch Size':12s}: {metrics['batch_used']}")

    if "save_dir" in metrics:
        print(f"   {'输出目录':12s}: {metrics['save_dir']}")

    print("=" * 60)


# ==================================================================
# CLI
# ==================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NutriMind-Agent YOLO26 训练脚本 (AutoDL 优化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 prepare_data.py 生成的 data.yaml 开始训练
  python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml

  # 自定义参数
  python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml \\
      --model yolo26s --epochs 100 --batch 32 --workers 8

  # 后台运行（推荐！防止 SSH 断开）
  nohup python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml > train.log 2>&1 &

  # 跟踪日志
  tail -f train.log

  # 使用 screen
  screen -S train_yolo26
  python train_yolo26.py --data /root/autodl-tmp/food_data/data.yaml
  # 按 Ctrl+A 再按 D 分离会话
  # 恢复: screen -r train_yolo26

  # 仅检查环境
  python train_yolo26.py --check-env
        """,
    )

    parser.add_argument(
        "--data", type=str, required=False,
        help="data.yaml 配置文件路径"
    )
    parser.add_argument(
        "--model", type=str, default=Config.MODEL_NAME,
        help=f"预训练模型名称（默认: {Config.MODEL_NAME}）"
    )
    parser.add_argument(
        "--epochs", type=int, default=Config.EPOCHS,
        help=f"训练轮数（默认: {Config.EPOCHS}）"
    )
    parser.add_argument(
        "--img-size", type=int, default=Config.IMG_SIZE,
        help=f"输入图像尺寸（默认: {Config.IMG_SIZE}）"
    )
    parser.add_argument(
        "--batch", type=int, default=Config.BATCH,
        help=f"初始 batch size（默认: {Config.BATCH}，OOM 时自动降级）"
    )
    parser.add_argument(
        "--workers", type=int, default=Config.WORKERS,
        help=f"数据加载线程数（默认: {Config.WORKERS}）"
    )
    parser.add_argument(
        "--patience", type=int, default=Config.PATIENCE,
        help=f"早停轮数（默认: {Config.PATIENCE}，0 禁用）"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="禁用图片缓存（显存不足时使用）"
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="禁用自动混合精度"
    )
    parser.add_argument(
        "--device", type=int, default=Config.DEVICE,
        help=f"GPU 设备编号（默认: {Config.DEVICE}）"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="模型输出目录（默认: backend/data/models/）"
    )
    parser.add_argument(
        "--check-env", action="store_true",
        help="仅检查环境，不执行训练"
    )

    args = parser.parse_args()

    # 模式：仅检查环境
    if args.check_env:
        check_environment()
        return

    # 必须提供 data.yaml
    if args.data is None:
        parser.print_help()
        print("\n❌ 请指定 --data 参数（data.yaml 路径）")
        print("   提示：先用 prepare_data.py 准备数据")
        sys.exit(1)

    # 检查环境
    env_info = check_environment()
    if not env_info["ultralytics_available"]:
        sys.exit(1)

    # 确认运行
    print()
    print("⚠️  即将开始训练，注意：")
    print(f"   - 预计耗时: ~{args.epochs * 2 // 60} - {args.epochs * 4 // 60} 小时（取决于数据量）")
    print(f"   - 建议在 screen 或 nohup 中运行以防 SSH 断开")
    print(f"   - 训练日志将同时输出到控制台")
    print()

    # 执行训练
    metrics = run_training(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        cache=False if args.no_cache else Config.CACHE,
        amp=False if args.no_amp else Config.AMP,
        device=args.device,
        model_output_dir=args.output_dir,
    )

    print(f"\n🎉 训练流程全部完成！")
    print(f"   最终模型: {metrics.get('output_model', 'N/A')}")


if __name__ == "__main__":
    main()

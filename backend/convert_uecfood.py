#!/usr/bin/env python3
"""
UECFOOD256 → YOLOv11 格式转换脚本

输入: UECFOOD256.zip
输出:
  /root/autodl-tmp/UECFOOD256/
    data.yaml
    images/train/  |  labels/train/
    images/val/    |  labels/val/
    images/test/   |  labels/test/

用法:
  python convert_uecfood.py --zip /root/autodl-tmp/UECFOOD256.zip --out /root/autodl-tmp/UECFOOD256/
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

# ==================================================================
# 配置
# ==================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# test = 1 - train - val = 0.15


def parse_category_txt(zip_path: str) -> Dict[int, str]:
    """从 zip 中读取 category.txt，返回 {class_dir_number: food_name}"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        content = zf.read('UECFOOD256/category.txt').decode('utf-8')
    mapping = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('id'):
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            class_id = int(parts[0])
            name = parts[1].replace(' ', '_').replace('/', '-').lower()
            mapping[class_id] = name
    print(f"[OK] 读取到 {len(mapping)} 个食物类别")
    return mapping


def convert_bb_info(zip_path: str, class_dirs: List[str], class_mapping: Dict[int, str],
                     out_dir: Path) -> List[Tuple[str, str, int]]:
    """将 bb_info.txt 转为 YOLO 归一化格式并写入磁盘。

    返回 [(split, image_filename, class_id), ...] 用于后续 data.yaml
    """
    samples = []  # (split, rel_image_path, class_id)

    random.seed(42)

    for class_dir in class_dirs:
        class_num = int(class_dir)
        class_name = class_mapping.get(class_num, f"class_{class_num}")
        class_id = class_num - 1  # YOLO class IDs are 0-indexed

        prefix = f'UECFOOD256/{class_dir}/'
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List images in this class directory
            all_images = sorted([
                name for name in zf.namelist()
                if name.startswith(prefix) and name.lower().endswith('.jpg')
            ])

            # Read bb_info.txt
            bb_path = prefix + 'bb_info.txt'
            if bb_path not in zf.namelist():
                continue
            bb_content = zf.read(bb_path).decode('utf-8')

        # Parse bb_info.txt
        bboxes = {}  # image_num -> list of (x1,y1,x2,y2)
        for line in bb_content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('img'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                img_num = parts[0]
                x1, y1, x2, y2 = map(float, parts[1:5])
                if img_num not in bboxes:
                    bboxes[img_num] = []
                bboxes[img_num].append((x1, y1, x2, y2))

        # Build image filename -> image arcname mapping
        img_map = {}
        for arcname in all_images:
            fname = Path(arcname).stem  # e.g. "1"
            img_map[fname] = arcname

        # Shuffle and split
        random.shuffle(all_images)
        train_end = int(len(all_images) * TRAIN_RATIO)
        val_end = train_end + int(len(all_images) * VAL_RATIO)

        train_imgs = all_images[:train_end]
        val_imgs = all_images[train_end:val_end]
        test_imgs = all_images[val_end:]

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for split, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
                for arcname in img_list:
                    fname = Path(arcname).stem  # e.g. "1"
                    out_img_name = f'{class_name}__{Path(arcname).name}'

                    # Extract image
                    dst_img = out_dir / 'images' / split / out_img_name
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(arcname) as src:
                        dst_img.write_bytes(src.read())

                    # Write YOLO label
                    dst_label = out_dir / 'labels' / split / f'{dst_img.stem}.txt'
                    dst_label.parent.mkdir(parents=True, exist_ok=True)

                    if fname in bboxes:
                        # Get image dimensions from the first bbox context
                        # Actually we need to read image dimensions
                        try:
                            from PIL import Image
                            with zf.open(arcname) as src:
                                img = Image.open(src)
                                img_w, img_h = img.size
                        except Exception:
                            # fallback: try to read the already-extracted image
                            try:
                                from PIL import Image
                                img = Image.open(dst_img)
                                img_w, img_h = img.size
                            except Exception:
                                img_w, img_h = 640, 480  # worst-case fallback

                        lines = []
                        for x1, y1, x2, y2 in bboxes[fname]:
                            # Convert to YOLO format (cx, cy, w, h) normalized
                            cx = ((x1 + x2) / 2) / img_w
                            cy = ((y1 + y2) / 2) / img_h
                            w = (x2 - x1) / img_w
                            h = (y2 - y1) / img_h
                            # Clamp
                            cx = max(0, min(1, cx))
                            cy = max(0, min(1, cy))
                            w = max(0, min(1, w))
                            h = max(0, min(1, h))
                            lines.append(f'{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
                        dst_label.write_text('\n'.join(lines))
                    else:
                        # No bbox for this image (shouldn't happen but be safe)
                        dst_label.write_text('')

                    samples.append((split, out_img_name, class_id))

    return samples


def generate_data_yaml(out_dir: Path, class_names: List[str]) -> Path:
    """生成 data.yaml"""
    yaml_path = out_dir / 'data.yaml'
    content = f"""path: {out_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {json.dumps(class_names)}
"""
    yaml_path.write_text(content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='UECFOOD256 → YOLOv11 格式转换')
    parser.add_argument('--zip', required=True, help='UECFOOD256.zip 路径')
    parser.add_argument('--out', required=True, help='输出目录')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取类别映射
    print('=' * 60)
    print('[1/4] 读取类别映射...')
    class_mapping = parse_category_txt(args.zip)

    # 2. 获取所有分类目录
    with zipfile.ZipFile(args.zip, 'r') as zf:
        class_dirs = sorted(set(
            name.split('/')[1]
            for name in zf.namelist()
            if name.startswith('UECFOOD256/')
            and len(name.split('/')) >= 2
            and name.split('/')[1].isdigit()
        ))
    print(f'[OK] {len(class_dirs)} 个类别目录')

    # 3. 转换
    print('[2/4] 转换标注格式 (bb_info.txt → YOLO)...')
    samples = convert_bb_info(args.zip, class_dirs, class_mapping, out_dir)

    # 4. 统计
    print('[3/4] 统计...')
    splits = {'train': 0, 'val': 0, 'test': 0}
    for split, _, _ in samples:
        splits[split] += 1
    print(f'  train: {splits["train"]}  val: {splits["val"]}  test: {splits["test"]}')
    print(f'  总计: {sum(splits.values())} 张图片')

    # 5. 生成 data.yaml
    print('[4/4] 生成 data.yaml...')
    class_names = [class_mapping.get(i + 1, f'class_{i+1}') for i in range(len(class_dirs))]
    yaml_path = generate_data_yaml(out_dir, class_names)
    print(f'[OK] {yaml_path}')
    print('=' * 60)
    print('[DONE] 转换完成!')


if __name__ == '__main__':
    main()

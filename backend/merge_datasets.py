#!/usr/bin/env python3
"""
合并 UECFOOD-256 + School Lunch 数据集，支持增量微调。

流程：
1. 保持 UECFOOD-256 的 0-255 类不变
2. School Lunch 中重叠的类映射到 UECFOOD-256 class_id
3. School Lunch 中新增的类追加到 256+
4. 输出合并后的 data.yaml + 符号链接到 UECFOOD-256 的图片

输出目录结构:
  merged/
    data.yaml          # 合并后的配置 (nc: 263)
    images/train/      # UECFOOD-256 + school_lunch 训练图
    images/val/        # 同上
    labels/train/
    labels/val/

用法:
  python merge_datasets.py \
    --uecfood /root/autodl-tmp/UECFOOD256 \
    --schoollunch /root/autodl-tmp/school_lunch.zip \
    --out /root/autodl-tmp/merged_food/
"""

import argparse
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==================================================================
# 学校午餐 21 类 → UECFOOD-256 class_id 映射
# ==================================================================
# 键: school_lunch class_id (0-20)
# 值: UECFOOD-256 class_id (0-255) 或 None（新类别）

SCHOOL_LUNCH_CLASSES = [
    "Milk",              # 0  → 新
    "Drinkable yogurt",  # 1  → 新
    "Rice",              # 2  → UEC rice (0)
    "Mixed rice",        # 3  → UEC mixed_rice (94)
    "Bread",             # 4  → 新
    "White bread",       # 5  → 新
    "Udon",              # 6  → UEC udon_noodle (20)
    "Fish",              # 7  → 新
    "Meat",              # 8  → 新
    "Salad",             # 9  → UEC green_salad (83)
    "Cherry tomatoes",   # 10 → 新
    "Soups",             # 11 → UEC miso_soup (36)
    "Curry",             # 12 → UEC beef_curry (5)
    "Spicy chili-flavored tofu",  # 13 → UEC spicy_chili-flavored_tofu (68)
    "Bibimbap",          # 14 → UEC bibimbap (12)
    "Fried noodles",     # 15 → UEC fried_noodle (26)
    "Spaghetti",         # 16 → UEC spaghetti (27)
    "Citrus",            # 17 → 新
    "Apple",             # 18 → 新
    "Cup desserts",      # 19 → 新
    "Other foods",       # 20 → 新
]

# 手动指定重叠映射（UECFOOD-256 class_id 根据实际 data.yaml 确定）
# 这些映射来自 UECFOOD-256 category.txt 中的 class_id（目录编号 - 1）
UECFOOD_MAPPING = {
    "Rice":                         0,      # UEC: rice
    "Mixed rice":                   94,     # UEC: mixed_rice
    "Udon":                         20,     # UEC: udon_noodle
    "Salad":                        83,     # UEC: green_salad
    "Soups":                        36,     # UEC: miso_soup
    "Curry":                        5,      # UEC: beef_curry
    "Spicy chili-flavored tofu":   68,     # UEC: spicy_chili-flavored_tofu
    "Bibimbap":                     12,     # UEC: bibimbap
    "Fried noodles":                26,     # UEC: fried_noodle
    "Spaghetti":                    27,     # UEC: spaghetti
}


def read_uecfood_names(uecfood_dir: Path) -> Dict[int, str]:
    """读取 UECFOOD-256 data.yaml，返回 {class_id: name}。"""
    data_yaml = uecfood_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"[FAIL] 找不到 {data_yaml}")
        sys.exit(1)

    # 简单解析 YAML
    nc = 0
    names = []
    with open(data_yaml) as f:
        for line in f:
            line = line.strip()
            if line.startswith("nc:"):
                nc = int(line.split(":")[1].strip())
            elif line.startswith("names:"):
                # names: ["rice", "eels_on_rice", ...]
                names_str = line.split(":", 1)[1].strip()
                names = json.loads(names_str)

    if len(names) != nc:
        print(f"[WARN] nc={nc} 但 names 有 {len(names)} 个")
    print(f"[OK] UECFOOD-256: {len(names)} 类")
    return {i: name for i, name in enumerate(names)}


def build_mapping(uecfood_names: Dict[int, str]) -> Dict[int, Optional[int]]:
    """构建 school_lunch → 合并后 class_id 的映射。"""
    # 反向索引：UEC name → class_id
    uec_name_to_id = {name: cid for cid, name in uecfood_names.items()}

    mapping: Dict[int, Optional[int]] = {}
    new_classes: List[str] = []

    for sl_id, sl_name in enumerate(SCHOOL_LUNCH_CLASSES):
        # 尝试按 UECFOOD_MAPPING 手动映射
        if sl_name in UECFOOD_MAPPING:
            uec_id = UECFOOD_MAPPING[sl_name]
            print(f"  [MAP] school_lunch[{sl_id}] '{sl_name}' → UEC[{uec_id}] '{uecfood_names.get(uec_id, '?')}' ")
            mapping[sl_id] = uec_id
            continue

        # 尝试按名称匹配
        sl_name_lower = sl_name.lower().replace(' ', '_')
        if sl_name_lower in uec_name_to_id:
            uec_id = uec_name_to_id[sl_name_lower]
            print(f"  [MATCH] school_lunch[{sl_id}] '{sl_name}' → UEC[{uec_id}]")
            mapping[sl_id] = uec_id
            continue

        # 新类别
        new_id = len(uecfood_names) + len(new_classes)
        mapping[sl_id] = new_id
        new_classes.append(sl_name)
        print(f"  [NEW] school_lunch[{sl_id}] '{sl_name}' → 新增 class_id={new_id}")

    print(f"\n[OK] 重叠 {len(mapping) - len(new_classes)} 类, 新增 {len(new_classes)} 类")
    print(f"[OK] 合并后总计: {len(uecfood_names) + len(new_classes)} 类")
    return mapping


def convert_school_lunch(
    zip_path: str,
    out_dir: Path,
    mapping: Dict[int, Optional[int]],
    uecfood_names: Dict[int, str],
) -> Tuple[int, int]:
    """转换 school_lunch 数据集并符号链接到合并目录。

    Returns:
        (train_images, val_images)
    """
    zf = zipfile.ZipFile(zip_path, 'r')

    # 读取所有图片列表
    all_images = sorted([
        name for name in zf.namelist()
        if name.startswith('school_lunch/Images/') and name.lower().endswith('.jpg')
    ])

    # 80/20 split
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    total_train = total_val = 0

    for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
        for arcname in img_list:
            base = Path(arcname).stem  # e.g. "010008000"
            annot_path = f"school_lunch/Annotations/{base}.txt"

            if annot_path not in zf.namelist():
                continue

            # 读取标注
            annot_content = zf.read(annot_path).decode('utf-8').strip()
            if not annot_content:
                continue

            # 读取图片尺寸
            try:
                from PIL import Image
                with zf.open(arcname) as src:
                    img = Image.open(src)
                    img_w, img_h = img.size
            except Exception:
                img_w, img_h = 640, 480

            # 转换标注
            yolo_lines = []
            for line in annot_content.split('\n'):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                sl_class_id = int(parts[0])
                x1, y1, x2, y2 = map(float, parts[1:5])

                # 映射到合并后的 class_id
                merged_class_id = mapping.get(sl_class_id)
                if merged_class_id is None:
                    continue

                # 像素坐标 → YOLO 归一化
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                yolo_lines.append(f'{merged_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

            if not yolo_lines:
                continue

            # 输出文件名：加 sl_ 前缀避免与 UEC 图片重名
            out_name = f'sl_{base}.jpg'

            # 提取图片
            dst_img = out_dir / 'images' / split / out_name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                with zf.open(arcname) as src:
                    dst_img.write_bytes(src.read())

            # 写 YOLO 标签
            dst_label = out_dir / 'labels' / split / f'sl_{base}.txt'
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text('\n'.join(yolo_lines))

            if split == 'train':
                total_train += 1
            else:
                total_val += 1

    zf.close()
    return total_train, total_val


def symlink_uecfood(uecfood_dir: Path, out_dir: Path) -> Tuple[int, int]:
    """创建 UECFOOD-256 图片和标签的符号链接到合并目录。"""
    train_imgs = 0
    val_imgs = 0

    for split in ['train', 'val']:
        # 链接图片
        src_img_dir = uecfood_dir / 'images' / split
        dst_img_dir = out_dir / 'images' / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        if src_img_dir.exists():
            for img_file in src_img_dir.iterdir():
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    dst = dst_img_dir / img_file.name
                    if not dst.exists():
                        try:
                            os.symlink(img_file.resolve(), dst)
                        except OSError:
                            # 符号链接失败时直接复制（Windows / 非 root）
                            import shutil
                            shutil.copy2(img_file, dst)
                    if split == 'train':
                        train_imgs += 1
                    else:
                        val_imgs += 1

        # 链接标签
        src_label_dir = uecfood_dir / 'labels' / split
        dst_label_dir = out_dir / 'labels' / split
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        if src_label_dir.exists():
            for label_file in src_label_dir.iterdir():
                if label_file.suffix == '.txt':
                    dst = dst_label_dir / label_file.name
                    if not dst.exists():
                        try:
                            os.symlink(label_file.resolve(), dst)
                        except OSError:
                            import shutil
                            shutil.copy2(label_file, dst)

    return train_imgs, val_imgs


def generate_merged_yaml(out_dir: Path, uecfood_names: Dict[int, str],
                          new_classes: List[str]) -> Path:
    """生成合并后的 data.yaml。"""
    # 构建完整类名列表
    all_names = [uecfood_names[i] for i in range(len(uecfood_names))]
    all_names.extend(new_classes)

    yaml_path = out_dir / 'data.yaml'
    content = (
        f'path: {out_dir.resolve()}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'\n'
        f'nc: {len(all_names)}\n'
        f'names: {json.dumps(all_names)}\n'
    )
    yaml_path.write_text(content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='合并 UECFOOD-256 + School Lunch 数据集')
    parser.add_argument('--uecfood', required=True, help='UECFOOD-256 目录（含 data.yaml）')
    parser.add_argument('--schoollunch', required=True, help='school_lunch.zip 路径')
    parser.add_argument('--out', required=True, help='合并输出目录')
    args = parser.parse_args()

    uecfood_dir = Path(args.uecfood)
    out_dir = Path(args.out)

    print('=' * 60)
    print('[1/5] 读取 UECFOOD-256 data.yaml...')
    uecfood_names = read_uecfood_names(uecfood_dir)

    print('\n[2/5] 构建类别映射...')
    mapping = build_mapping(uecfood_names)
    new_classes = [SCHOOL_LUNCH_CLASSES[k]
                   for k in sorted(mapping.keys())
                   if mapping[k] >= len(uecfood_names)]

    print('\n[3/5] 转换 School Lunch 数据集...')
    sl_train, sl_val = convert_school_lunch(
        args.schoollunch, out_dir, mapping, uecfood_names
    )
    print(f'  School Lunch: train={sl_train}, val={sl_val}')

    print('\n[4/5] 链接 UECFOOD-256 图片...')
    uec_train, uec_val = symlink_uecfood(uecfood_dir, out_dir)
    print(f'  UECFOOD-256:  train={uec_train}, val={uec_val}')
    print(f'  合并:          train={uec_train + sl_train}, val={uec_val + sl_val}')

    print('\n[5/5] 生成 data.yaml...')
    yaml_path = generate_merged_yaml(out_dir, uecfood_names, new_classes)
    print(f'[OK] {yaml_path}')
    print(f'[OK] 总计 {len(uecfood_names) + len(new_classes)} 类')
    print(f'       原有: {len(uecfood_names)} (UECFOOD-256)')
    print(f'       新增: {len(new_classes)} {new_classes}')
    print('=' * 60)
    print('[DONE] 合并完成!')
    print()
    print('下一步: 用合并后的数据增量微调模型')
    print(f'  python train_yolo.py --data {yaml_path} --model /root/autodl-tmp/runs/detect/train/weights/best.pt --epochs 50')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create an ImageFolder-style layout for AwA2 from its CSV splits.

Usage:
  python3 reorg_awa2_to_imagefolder.py --src /path/to/Animals_with_Attributes2 [--dst /path/to/output] [--symlink]

The script reads `classes.txt` for class names and `train.csv` and `val.csv` (or `test.csv`) for samples.
It creates `dst/train/<idx>_<classname>/...` and `dst/val/...` directories and copies or symlinks images.
"""
import os
import argparse
import csv
import shutil
from pathlib import Path


def read_classes(classes_txt):
    classes = []
    with open(classes_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # lines are like: ' 1\tantelope' or '1\tantelope'
            parts = line.split()
            name = parts[-1]
            classes.append(name)
    return classes


def read_csv_split(csv_path):
    samples = []
    if not os.path.exists(csv_path):
        return samples
    with open(csv_path, 'r') as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            img_path = row[1].strip()
            try:
                label = int(row[2])
            except Exception:
                try:
                    label = int(row[-1])
                except Exception:
                    continue
            samples.append((img_path, label))
    return samples


def safe_name(name):
    # keep it readable but safe for filesystem
    return name.replace('+', '_').replace(' ', '_')


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='AwA2 root (where classes.txt, train.csv live)')
    p.add_argument('--dst', default=None, help='Destination root for ImageFolder layout (defaults to src + "_imagefolder")')
    p.add_argument('--symlink', action='store_true', help='Create symlinks instead of copying files')
    args = p.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst) if args.dst else src + '_imagefolder'

    classes_txt = os.path.join(src, 'classes.txt')
    train_csv = os.path.join(src, 'train.csv')
    val_csv = os.path.join(src, 'val.csv')
    test_csv = os.path.join(src, 'test.csv')

    if not os.path.exists(classes_txt):
        raise FileNotFoundError(f'classes.txt not found at {classes_txt}')

    classes = read_classes(classes_txt)
    print(f'Found {len(classes)} classes')

    # prepare splits
    train_samples = read_csv_split(train_csv)
    if os.path.exists(val_csv):
        val_samples = read_csv_split(val_csv)
    else:
        val_samples = read_csv_split(test_csv)

    print(f'Train samples: {len(train_samples)}, Val samples: {len(val_samples)}')

    # create dest dirs
    train_root = os.path.join(dst, 'train')
    val_root = os.path.join(dst, 'val')
    ensure_dir(train_root)
    ensure_dir(val_root)

    # helper to place samples
    def place_samples(samples, target_root):
        copied = 0
        missing = 0
        for path, label in samples:
            # labels in classes.txt are 1-based
            idx = int(label) - 1
            if idx < 0 or idx >= len(classes):
                missing += 1
                continue
            cname = classes[idx]
            folder_name = f"{idx:03d}_{safe_name(cname)}"
            class_dir = os.path.join(target_root, folder_name)
            ensure_dir(class_dir)

            # image path may be absolute or relative to src/JPEGImages
            img_path = path
            if not os.path.isabs(img_path):
                # try relative to src
                candidate = os.path.join(src, img_path)
                if os.path.exists(candidate):
                    img_path = candidate
                else:
                    # try JPEGImages/<img_path>
                    candidate2 = os.path.join(src, 'JPEGImages', img_path)
                    if os.path.exists(candidate2):
                        img_path = candidate2

            if not os.path.exists(img_path):
                # try removing leading slashes
                img_path2 = img_path.lstrip('/')
                candidate3 = os.path.join(src, img_path2)
                if os.path.exists(candidate3):
                    img_path = candidate3

            if not os.path.exists(img_path):
                missing += 1
                continue

            dest_path = os.path.join(class_dir, os.path.basename(img_path))
            if args.symlink:
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.symlink(img_path, dest_path)
                except OSError:
                    # fallback to copy
                    shutil.copy2(img_path, dest_path)
            else:
                if not os.path.exists(dest_path):
                    shutil.copy2(img_path, dest_path)
            copied += 1

        return copied, missing

    print('Placing train samples...')
    t_copied, t_missing = place_samples(train_samples, train_root)
    print('Placing val samples...')
    v_copied, v_missing = place_samples(val_samples, val_root)

    print('Done.')
    print(f'Train copied: {t_copied}, missing: {t_missing}')
    print(f'Val copied: {v_copied}, missing: {v_missing}')
    print(f'ImageFolder layout created at: {dst}')


if __name__ == '__main__':
    main()

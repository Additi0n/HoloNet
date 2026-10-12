"""Bin2Img: turn executables into the 96 x 96 RGBA images used by HoloNet.

Two steps, run on a dataset laid out as <root>/<split>/<class>/<file>:

    python preprocessing/bin2img.py convert RAW/BODMAS/train RawIMG/BODMAS/train
    python preprocessing/bin2img.py resize  RawIMG/BODMAS/train RGBA/BODMAS/train

`convert` writes each file as an n x n image with 4 consecutive bytes per RGBA
pixel, n = ceil(sqrt(len / 4)), zero-padded (lossless). `resize` scales every
image to 96 x 96 with PIL's Image.resize and its default filter, which is bicubic
for RGBA and works on premultiplied alpha. This is the procedure that produced
the published images; the logic is taken unchanged from the original scripts
(Byte2Img.py, Sample2PNG.py, resized.py).
"""
from __future__ import annotations

import argparse
import math
import os

from PIL import Image

MODES = {1: "L", 3: "RGB", 4: "RGBA"}


def byte2img(sample_path: str, output_path: str, channels: int = 4) -> None:
    """Write the bytes of one file as a square image, `channels` bytes per pixel."""
    if channels not in MODES:
        raise ValueError(f"Unsupported number of channels: {channels}")
    with open(sample_path, "rb") as f:
        data = f.read()

    size = int(math.ceil(math.sqrt(len(data) / channels)))
    image = Image.new(MODES[channels], (size, size))

    required_length = size * size * channels
    if len(data) < required_length:
        data += b"\x00" * (required_length - len(data))  # zero padding

    if channels == 1:
        image.putdata(list(data[:required_length]))
    elif channels == 3:
        image.putdata([(data[i], data[i + 1], data[i + 2]) for i in range(0, required_length, 3)])
    else:
        image.putdata([(data[i], data[i + 1], data[i + 2], data[i + 3]) for i in range(0, required_length, 4)])
    image.save(output_path)


def convert_tree(source_dir: str, target_dir: str, channels: int = 4) -> None:
    """Convert <source_dir>/<class>/<file> into <target_dir>/<class>/<file stem>.png."""
    for label in sorted(os.listdir(source_dir)):
        label_path = os.path.join(source_dir, label)
        if not os.path.isdir(label_path):
            continue
        target_label_dir = os.path.join(target_dir, label)
        os.makedirs(target_label_dir, exist_ok=True)
        for sample_file in sorted(os.listdir(label_path)):
            sample_path = os.path.join(label_path, sample_file)
            if os.path.isfile(sample_path):
                out = os.path.join(target_label_dir, f"{os.path.splitext(sample_file)[0]}.png")
                byte2img(sample_path, out, channels)
                print(f"Converted {sample_path} to {out}")


def resize_tree(input_folder: str, output_folder: str, size: int = 96) -> None:
    """Resize <input_folder>/<class>/<image> to size x size, keeping the channel count."""
    for label in sorted(os.listdir(input_folder)):
        input_dir = os.path.join(input_folder, label)
        if not os.path.isdir(input_dir):
            continue
        output_dir = os.path.join(output_folder, label)
        os.makedirs(output_dir, exist_ok=True)
        for file in sorted(os.listdir(input_dir)):
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                continue
            try:
                with Image.open(os.path.join(input_dir, file)) as image:
                    image.resize((size, size)).save(os.path.join(output_dir, file))
            except Exception as e:  # keep going over corrupt files, as the original script did
                print("error:", os.path.join(output_dir, file), e)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="executables -> n x n byte images")
    c.add_argument("source_dir")
    c.add_argument("target_dir")
    c.add_argument("--channels", type=int, default=4, choices=sorted(MODES))
    r = sub.add_parser("resize", help="n x n byte images -> fixed-size images")
    r.add_argument("input_dir")
    r.add_argument("output_dir")
    r.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    if args.command == "convert":
        convert_tree(args.source_dir, args.target_dir, args.channels)
    else:
        resize_tree(args.input_dir, args.output_dir, args.size)


if __name__ == "__main__":
    main()

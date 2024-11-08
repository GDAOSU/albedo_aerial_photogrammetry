#!/usr/bin/env python

import os.path as osp
import argparse
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=osp.basename(__file__))
    parser.add_argument("infolder", type=str, help="Input Folder, support .exr")
    parser.add_argument("outfolder", type=str, help="Output Folder")
    parser.add_argument("--ldr_ext", type=str, default="png", choices=["png", "jpg"], help="LDR Extension")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma for HDR")
    parser.add_argument("--no_gamma", action="store_true", help="No gamma correction")
    parser.add_argument("--max_perc", type=float, default=98, help="Percentile for HDR")
    parser.add_argument("--min_perc", type=float, default=2, help="Percentile for HDR")

    args = parser.parse_args()
    # convert HDR to LDR
    in_dir = Path(args.infolder)
    out_dir = Path(args.outfolder)
    out_dir.mkdir(parents=True, exist_ok=True)

    HDR_FILES = list(sorted(in_dir.glob("*.exr")))

    for hdr_path in tqdm(HDR_FILES, desc="Converting HDR to LDR"):
        ldr_path = out_dir / f"{hdr_path.stem}.{args.ldr_ext}"

        img = imageio.imread(hdr_path)
        H, W = img.shape[:2]

        if img.ndim > 2 and not args.no_gamma:
            C = img.shape[2]
            if C == 3:
                img = img ** (1 / args.gamma)
            elif C == 4:
                img[..., :3] = img[..., :3] ** (1 / args.gamma)

        lb = np.nanpercentile(img, args.min_perc)
        ub = np.nanpercentile(img, args.max_perc)

        stretched = np.clip((img - lb) / (ub - lb) * 255, 0, 255).astype(np.uint8)

        imageio.imwrite(ldr_path, stretched)

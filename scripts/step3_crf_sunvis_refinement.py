from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage.morphology import erosion, dilation, disk
from skimage import color
from skimage.util import *
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import os.path as osp
from glob import glob
import numpy as np
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def crf_trimap(img, init_trimap, crf_iter=5, erode_radius=2):
    assert img.shape[:2] == init_trimap.shape[:2], f"Size mismatch {img.shape} vs {init_trimap.shape}"
    # >0.9 Visible  < 0.1 Unknown ~0.19 Invisible
    H, W = img.shape[:2]
    # Init masks from sunvis trimap
    lit_mask = init_trimap > 0.9
    shadow_mask = (init_trimap > 0.01) & (init_trimap <= 0.9)

    # Erode masks
    lit_mask = erosion(lit_mask, disk(erode_radius)) > 0
    shadow_mask = erosion(shadow_mask, disk(erode_radius)) > 0

    # Convert masks to label map
    label_map = np.zeros((H, W), dtype=np.int32)
    label_map[lit_mask] = 2
    label_map[shadow_mask] = 1
    # Dense CRF
    d = dcrf.DenseCRF2D(W, H, 2)  # width, height, nlabels
    U = unary_from_labels(label_map, 2, gt_prob=0.9, zero_unsure=True)
    d.setUnaryEnergy(U.reshape(2, -1))
    # d.addPairwiseBilateral(sxy=16, srgb=2, rgbim=(img*255.0).astype(np.uint8),
    #                        compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(
        sxy=16,
        srgb=32,
        rgbim=(img * 255.0).astype(np.uint8),
        compat=20,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    Q = d.inference(crf_iter)
    # Get refined label
    label_crf = np.argmax(Q, axis=0).reshape(H, W)

    return label_crf


def direct_trimap(img, init_trimap, crf_iter=50):
    assert img.shape[:2] == init_trimap.shape[:2]
    # >0.9 Visible  < 0.1 Unknown ~0.19 Invisible
    H, W = img.shape[:2]
    # Init masks from sunvis trimap
    lit_mask = init_trimap > 0.9
    shadow_mask = (init_trimap > 0.01) & (init_trimap <= 0.9)
    # Get refined label
    return lit_mask


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Create Thumbnail")
    parser.add_argument("input_folder", type=str, help="Input folder (ContextCapture Tiled OBJ)")
    parser.add_argument("--radius", type=int, default=2, help="Erosion radius")
    parser.add_argument("--crf_iter", type=int, default=5, help="CRF iteration")
    return parser


def _main():
    parser = _get_parser()
    args = parser.parse_args()

    work_dir = Path(args.input_folder)
    image_dir = work_dir / "image"
    thumbnail_dir = work_dir / "thumbnail"
    sunvis_dir = work_dir / "sunvis"
    for _dir in [image_dir, thumbnail_dir, sunvis_dir]:
        if not _dir.is_dir():
            logger.critical(f"{_dir} does not exist")
            parser.print_usage()
            return 1

    imgname_list = [p.stem for p in sunvis_dir.glob("*.png")]

    sunvis_crf_dir = sunvis_dir / "crf"
    sunvis_crf_dir.mkdir(exist_ok=True)

    for imgname in tqdm(imgname_list, desc="CRF"):
        img_path = image_dir / (imgname + ".exr")
        thumb_path = thumbnail_dir / (imgname + ".png")
        sunvis_path = sunvis_dir / (imgname + ".png")
        outcrf_path = sunvis_crf_dir / (imgname + ".png")

        if not img_path.exists() or not thumb_path.exists() or not sunvis_path.exists():
            continue
        if outcrf_path.exists():
            continue

        thumb = img_as_float(imageio.imread(thumb_path))
        sunvis = img_as_float(imageio.imread(sunvis_path))

        label_crf = crf_trimap(thumb, sunvis, crf_iter=args.crf_iter, erode_radius=args.radius)
        # label_crf = direct_trimap(thumb, sunvis)
        imageio.imwrite(outcrf_path, img_as_ubyte(label_crf * 255))
    logger.info("Done")


if __name__ == "__main__":
    sys.exit(_main())

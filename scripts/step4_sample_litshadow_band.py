from tqdm import tqdm

from shapely.geometry import LineString

# from skimage.morphology import erosion, dilation, disk
from skimage import color, filters, measure
from skimage.util import *
import cv2

# from scipy import interpolate
# from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt
import json
from multiprocessing import Pool
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import os
import os.path as osp
import numpy as np
import errno
from functools import lru_cache
import _pickle as cPickle
from bz2 import BZ2File
import time
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def require_files(paths):
    if isinstance(paths, str):
        v = paths
        if not osp.exists(v):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), v)
    elif isinstance(paths, dict):
        for _, v in paths.items():
            if not osp.exists(v):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), v)
    elif isinstance(paths, list):
        for v in paths:
            if not osp.exists(v):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), v)
    else:
        raise TypeError()


def check_files(paths):
    try:
        require_files(paths)
        return True
    except FileNotFoundError as e:
        return False


def interpolate_missing_pixels(
    image: np.ndarray, mask: np.ndarray, method: str = "nearest", fill_value: int = 0
) -> np.ndarray:
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y), method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def ensure_finite(img: np.ndarray) -> np.ndarray:
    """Inpaint NAN or INF value for input image

    Args:
        img (np.ndarray): input image

    Returns:
        np.ndarray: inpainted image
    """
    if len(img.shape) == 2:
        _mask = np.isnan(img)
        if _mask.any():
            img = interpolate_missing_pixels(img, _mask)
    elif len(img.shape) == 3:
        for c in range(img.shape[2]):
            img[..., c] = ensure_finite(img[..., c])
    return img

import OpenImageIO as oiio
from OpenImageIO import ImageInput, ImageSpec, ImageBuf, ImageBufAlgo
config = ImageSpec()
config["oiio:RawColor"] = 1
config["raw:ColorSpace"] = "Linear"

MAX_CACHE = 50

@lru_cache(MAX_CACHE)
def imreader(p: str) -> np.ndarray:
    p = str(p)
    _ext = osp.splitext(p)[1]
    if _ext.lower() == ".exr":
        imghandle = ImageInput.open(p, config)
        if imghandle is None:
            raise ValueError("Failed to open image: %s" % p)
        
        data = imghandle.read_image()
        imghandle.close()
        if data.ndim == 2:
            data = np.expand_dims(data, 2)
        assert data.ndim == 3
        return data
    else:
        data = img_as_float(imageio.imread(p)).astype(np.float32)
    if data.ndim == 2:
        data = np.expand_dims(data, 2)
    assert data.ndim == 3
    return data


def bilinear_interpolate(im: np.ndarray, x: np.array, y: np.array) -> np.ndarray:
    """Sample pixel on 2d image with nearest search.

    Args:
        im (np.ndarray): input image (Height, Width, Channel)
        x (np.ndarray): 1D array of point x (N,)
        y (np.ndarray): 1D array of point y (N,)

    Returns:
        np.ndarray: Sampled points (N, Channel)
    """
    multichannel = True
    if len(im.shape) == 3:
        multichannel = True
    else:
        multichannel = False

    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0, ...]
    Ib = im[y1, x0, ...]
    Ic = im[y0, x1, ...]
    Id = im[y1, x1, ...]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    if multichannel:
        return (
            wa[..., np.newaxis] * Ia + wb[..., np.newaxis] * Ib + wc[..., np.newaxis] * Ic + wd[..., np.newaxis] * Id
        )
    else:
        return wa * Ia + wb * Ib + wc * Ic + wd * Id


def nearest_interpolate(im: np.ndarray, x: np.array, y: np.array) -> np.ndarray:
    """Sample pixel on 2d image with nearest search.

    Args:
        im (np.ndarray): input image (Height, Width, Channel)
        x (np.ndarray): 1D array of point x (N,)
        y (np.ndarray): 1D array of point y (N,)

    Returns:
        np.ndarray: Sampled points (N, Channel)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x0 = np.round(x).astype(int)
    y0 = np.round(y).astype(int)
    x0 = np.clip(x0, 0, im.shape[1])
    y0 = np.clip(y0, 0, im.shape[0])
    Ia = im[y0, x0, ...]
    return Ia


class ProfileSampler:
    def __init__(self, shape: tuple, sample_center: np.ndarray, sample_dir: np.ndarray, radius: int):
        assert len(sample_center.shape) == 2 and sample_center.shape[-1] == 2, "unrecognized shape of sample_center"
        assert len(sample_dir.shape) == 2 and sample_dir.shape[-1] == 2, "unrecognized shape of sample_dir"
        assert len(sample_center) == len(sample_dir), "sample center and dir should match"
        assert len(shape) == 2, "Shape must be 2-dim"
        self.sample_center = sample_center
        self.sample_dir = sample_dir
        self.num_samples = len(sample_center)
        self.radius = radius
        self.height, self.width = shape[:2]

        self.generate_sample_buffer(radius)

    def generate_sample_buffer(self, radius):
        self.xprof = np.linspace(-radius, radius, 2 * radius + 1)
        self.prof_pt = np.einsum("ik,j->ijk", self.sample_dir, self.xprof) + self.sample_center[:, np.newaxis, :]
        self.sample_mask = np.logical_and.reduce(
            [
                self.prof_pt[..., 0].min(axis=1) > 0,
                self.prof_pt[..., 1].min(axis=1) > 0,
                self.prof_pt[..., 0].max(axis=1) < self.height,
                self.prof_pt[..., 1].max(axis=1) < self.width,
            ]
        )

    def sample_profile(self, si):
        ptprof = self.prof_pt[si]
        if self.sample_mask[si]:
            assert self.img.shape[:2] == (self.height, self.width), f"Image shape mismatch {self.img.shape}"
            vprof = bilinear_interpolate(self.img, ptprof[:, 1], ptprof[:, 0])
        else:
            vprof = self.empty_element
        return vprof

    def exec_python(self, img):
        self.img = img
        self.multichannel = len(self.img.shape) == 3
        if self.multichannel:
            self.empty_element = np.zeros((len(self.xprof), self.img.shape[-1]))
        else:
            self.empty_element = np.zeros(len(self.xprof))

        results = np.asarray([self.sample_profile(i) for i in range(self.num_samples)])

        del self.img
        del self.multichannel
        del self.empty_element
        return results

    def exec_mp(self, img):
        self.img = img
        self.multichannel = len(self.img.shape) == 3
        if self.multichannel:
            self.empty_element = np.zeros((len(self.xprof), self.img.shape[-1]))
        else:
            self.empty_element = np.zeros(len(self.xprof))

        results = np.array(Pool().map(self.sample_profile, np.arange(self.num_samples)))

        del self.img
        del self.multichannel
        del self.empty_element
        return results


#   model = {sobel prewitt scharr farid}
def find_gradient(img: np.ndarray, model="farid") -> np.ndarray:
    if model == "farid":
        hfunc = filters.farid_h
        vfunc = filters.farid_v
    elif model == "prewitt":
        hfunc = filters.prewitt_h
        vfunc = filters.prewitt_v
    elif model == "scharr":
        hfunc = filters.scharr_h
        vfunc = filters.scharr_v
    else:
        hfunc = filters.sobel_h
        vfunc = filters.sobel_v

    channelwise_edges_y = np.stack([hfunc(img[..., c]) for c in range(3)], axis=2)
    channelwise_edges_x = np.stack([vfunc(img[..., c]) for c in range(3)], axis=2)
    # Detect edges and gradient
    dx = channelwise_edges_x.mean(axis=2)
    dy = channelwise_edges_y.mean(axis=2)
    edge_mag = np.sqrt(dx**2 + dy**2)
    dx /= edge_mag + 1e-9
    dy /= edge_mag + 1e-9

    return np.stack([dy, dx], axis=2)


MIN_BOUNDARY_LENGTH = 200  # Eliminate very short shadows, which are most likely noises

import matplotlib.pyplot as plt


def find_litboundary(sunviscrf, MIN_BOUNDARY_LENGTH=100):
    contours = [
        c.astype(np.float32) for c in measure.find_contours(sunviscrf, level=0.5) if len(c) > MIN_BOUNDARY_LENGTH
    ]
    simpboundary = [cv2.approxPolyDP(np.expand_dims(l, 1), 2, False)[:, 0, :] for l in contours]
    
    if False: # Display the detected boundary
        plt.matshow(sunviscrf)
        for l in contours:
            print(l)
            plt.plot(l[:, 1], l[:, 0])

        plt.matshow(sunviscrf)
        for l in simpboundary:
            plt.plot(l[:, 1], l[:, 0])
            plt.scatter(l[:, 1], l[:, 0], marker="x")
        plt.show()
        
    return contours


def sample_a_contour(c, interval=1):
    assert c.dtype == np.float32
    lstring = LineString(c)
    pts = []
    i = 0
    for i in np.arange(0, lstring.length, interval):
        pts.append(lstring.interpolate(i).coords)
    return np.array(pts)


def sample_contours(contours, interval=1):
    pts = [sample_a_contour(c) for c in contours]
    return np.concatenate(pts, axis=0)


def process_sample_litshadow_band(data: dict, radius: int) -> dict:
    litboundary = find_litboundary(data["sunviscrf"][...,0])

    sample_center = [l for l in litboundary if len(l) > MIN_BOUNDARY_LENGTH]
    if len(sample_center) == 0:
        return None
    sample_center = np.concatenate(sample_center, axis=0)
    imgrad = find_gradient(filters.gaussian(data["img"], sigma=1, channel_axis=2))

    # dog_img = filters.difference_of_gaussians(
    #     data['img'], low_sigma=1, high_sigma=12, multichannel=True)
    height, width = data["img"].shape[:2]
    # Extract Distance map to the shadow boundary
    edt_sunvis = distance_transform_edt(filters.sobel(data["sunviscrf"]) == 0)
    # Second derivative of depth, detect high-contrast regions
    dog_depth = filters.difference_of_gaussians(data["depth"][..., 0], low_sigma=1, high_sigma=3)

    sample_grad = bilinear_interpolate(imgrad, sample_center[:, 1], sample_center[:, 0])

    sampler = ProfileSampler((height, width), sample_center, sample_grad, radius)

    if False: # Display footprint of bands
        fig, ax = plt.subplots()
        ax.matshow(data["img"])
        linepts = np.stack(
            [
                sample_center[:, ::-1] - radius * sample_grad[:, ::-1],
                sample_center[:, ::-1] + radius * sample_grad[:, ::-1],
            ],
            axis=1,
        )
        print("LineCollection Shape", linepts.shape)
        lines = LineCollection(linepts, color="red")
        ax.add_collection(lines)
        plt.show()

    profiles = {
        k: sampler.exec_mp(data[k])
        for k in (
            "img",
            "depth",
            "normal",
            "sunviscrf",
            "skyvisratio",
            # , 'geomid', 'barycentric'
        )
    }

    profiles["edt_sunvis"] = sampler.exec_mp(edt_sunvis)
    profiles["dog_depth"] = sampler.exec_mp(dog_depth)

    if False: # display sampled bands
        fig, ax = plt.subplots()
        ax.matshow(data['img'])
        # draw
        _L = len(sampler.sample_center)
        _start = int(_L*0.3) # percentile
        sel = np.s_[_start: _start + 100]
        sampled_centers = sampler.sample_center[sel]
        print("sampled centers", sampled_centers.shape)
        ax.scatter(sampled_centers[:, 1], sampled_centers[:, 0], marker="x", color="red")
        
        for k in ['img','depth','sunviscrf','normal', 'skyvisratio', 'edt_sunvis', 'dog_depth']:
            plt.matshow(profiles[k][sel])
            plt.title(k)
        plt.show()
        
    profiles["sampler"] = dict(
        width=sampler.width,
        height=sampler.height,
        radius=sampler.radius,
        centers=sampler.sample_center,
        dirs=sampler.sample_dir,
        xprof=sampler.xprof,
        prof_pt=sampler.prof_pt,
        mask=sampler.sample_mask,
    )

    return profiles


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Sample Lit Shadow Band")
    parser.add_argument("input_folder", type=str, help="Input folder (ContextCapture Tiled OBJ)")

    parser.add_argument("--radius", type=int, help="Radius of sampling", default=5)

    return parser


def _main():
    logging.basicConfig(level=logging.INFO)
    parser = _get_parser()
    args = parser.parse_args()

    work_dir = Path(args.input_folder)
    imagejson_path = work_dir / "imagedataset.json"
    # AOprop_path = work_dir / "models" / "aoprops.dat"
    # directionsamples_path = work_dir / "models" / "dirsamples.json"
    if not work_dir.is_dir():
        logger.critical("Input folder does not exist")
        parser.print_usage()
        return 1
    if not imagejson_path.exists():
        logger.critical(f"{imagejson_path} does not exist")
        return 1

    radius = args.radius

    litshadow_band_dir = work_dir / "ls_band"
    litshadow_band_dir.mkdir(parents=True, exist_ok=True)

    # Load Images
    with open(imagejson_path, "r") as f:
        imgdb = json.load(f)

    imgname_list = list(imgdb["Extrinsic"].keys())

    verbose = True
    compress = "none"

    for imgname in tqdm(imgname_list, desc="Sample Lit Shadow Band"):
        if verbose:
            logger.info(f"Processing {imgname}")
        imgmeta = imgdb["ImageMeta"][imgname]
        lsbandpkl = litshadow_band_dir / f"{imgname}.pkl"
        if osp.exists(lsbandpkl):
            if verbose:
                logger.info("File already exists")
            continue
        paths = dict(
            img=work_dir / "image" / (imgname + ".exr"),
            thumbnail=work_dir / "thumbnail" / (imgname + ".png"),
            sunviscrf=work_dir / "sunvis" / "crf" / (imgname + ".png"),
            skyvisratio=work_dir / "skyvis" / (imgname + ".png"),
            normal=work_dir / "normal" / (imgname + ".exr"),
            depth=work_dir / "depth" / (imgname + ".exr"),
            # geomid=osp.join(WORKDIR, 'geomid', imgname+'.exr'),
            # barycentric=osp.join(WORKDIR, 'barycentric', imgname+'.exr')
        )

        sunpos = np.array([imgmeta[k] for k in ("Sun:LocalPos_x", "Sun:LocalPos_y", "Sun:LocalPos_z")])

        if not check_files(paths):

            print("File not sufficient")
            continue

        if verbose:
            logger.info("Reading ...")
        _st = time.time()
        data = {k: imreader(v) for k, v in paths.items()}
        
        if False: # display raw bands
            for k, v in data.items():
                plt.matshow(v)
                plt.title(k)
            plt.show()
        
        if verbose:
            logger.info(f"{time.time()-_st:.3f}s")

        if verbose:
            logging.info("Preparing ...")
        _st = time.time()
        data["depth"] = ensure_finite(data["depth"])
        data["normal"] = ensure_finite(data["normal"])
        data["sunpos"] = sunpos
        # data['geomid'] = data['geomid'].astype(np.uint32)
        if verbose:
            logger.info(f"{time.time()-_st:.3f}s")

        if verbose:
            logger.info("Processing ...")
        _st = time.time()
        result = process_sample_litshadow_band(data, radius)

        if result is None:
            print(f"Failed to process {imgname}")
            continue

        if verbose:
            logger.info(f"{time.time()-_st:.3f}s")
        if compress == "bz2":
            with BZ2File(lsbandpkl + ".bz", "w") as fp:
                cPickle.dump(result, fp)
        else:
            with open(lsbandpkl, "wb") as fp:
                cPickle.dump(result, fp)


if __name__ == "__main__":
    sys.exit(_main())

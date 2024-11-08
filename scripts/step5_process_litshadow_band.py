import sys
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path as osp
from tqdm import tqdm
from scipy.optimize import least_squares
from tqdm import tqdm
from multiprocessing import Pool
import json
import warnings

warnings.filterwarnings("ignore")
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def strictly_decreasing(L):
    return (np.diff(L, axis=1) < 0).all(axis=1)


def strictly_increasing(L):
    return (np.diff(L, axis=1) > 0).all(axis=1)


def batch_multichannel_sigmoid(params, x):
    x0t = params[:, 0:1]
    kt = params[:, 1:2]
    bt = params[:, 2:5]
    Lt = params[:, 5:8]
    sigrec = np.expand_dims(Lt, 1) / np.expand_dims(1.0 + np.exp(-kt * (x - x0t)), 2) + np.expand_dims(bt, 1)
    return sigrec


def multichannel_sigmoid(param, x):
    x0t = param[0:1]
    kt = param[1:2]
    bt = param[2:5]
    Lt = param[5:8]
    sigrec = np.expand_dims(Lt, 0) / np.expand_dims(1.0 + np.exp(-kt * (x - x0t)), 1) + np.expand_dims(bt, 0)
    return sigrec


def myloss(p, x, y, w):
    sigrec = multichannel_sigmoid(p, x)
    residual = (sigrec - y) * w
    return residual.reshape(-1)


def exec_python(init_param, xprof, yprof, weight):
    opt_param = np.zeros_like(init_param)
    B = len(init_param)
    opt_mask = np.zeros(B) < 0
    for bi in tqdm(range(B)):
        res = least_squares(
            myloss, init_param[bi], args=(xprof[bi], yprof[bi], weight[bi]), loss="cauchy", method="trf", f_scale=5e-2
        )
        opt_mask[bi] = res.success
        opt_param[bi] = res.x
    return opt_param, opt_mask


def mysubprocess(_p, _x, _y, _w):
    res = least_squares(myloss, _p, args=(_x, _y, _w), loss="cauchy", method="trf", f_scale=5e-2)
    return res.x, res.success


def exec_mp(init_param, xprof, yprof, weight):
    B = len(init_param)
    opt_param = np.zeros_like(init_param)
    opt_mask = np.zeros(B) < 0
    output = Pool().starmap(mysubprocess, zip(init_param, xprof, yprof, weight))
    for bi, v in enumerate(output):
        opt_param[bi] = v[0]
        opt_mask[bi] = v[1]
    return opt_param, opt_mask


###################
MINIMUM_RADIANCE = 0.02
MAXIMUM_DOG_DEPTH = 0.05


#######################
def fit_lsband(lsband: dict) -> dict():
    imgband = lsband["img"]
    depband = lsband["depth"]
    norband = lsband["normal"]
    sunvisband = lsband["sunviscrf"]
    skyvisband = lsband["skyvisratio"]
    edtsunvisband = lsband["edt_sunvis"]
    dogdepband = lsband["dog_depth"]

    sample_radius = lsband["sampler"]["radius"]
    sample_xprof = lsband["sampler"]["xprof"]
    sample_cent = lsband["sampler"]["centers"]
    sample_grad = lsband["sampler"]["dirs"]
    # Filtering with Criteria
    ##############
    ## Sampler mask where all band fall within the image
    sampler_mask = lsband["sampler"]["mask"]
    ## Change of depth cannot be too big.
    dog_filter = dogdepband.max(axis=1) < MAXIMUM_DOG_DEPTH
    if dog_filter.sum() == 0:
        print(f"All killed by DOG filter.")
    ## sunvis monotonic.
    sunvis_filter = np.logical_and.reduce(
        [
            sunvisband.max(axis=1)[:, 0] == 1,
            sunvisband.min(axis=1)[:, 0] == 0,
            sunvisband[:, 0, 0] < sunvisband[:, -1, 0],
        ]
    )

    if sunvis_filter.sum() == 0:
        print(f"All killed by Sunvis filter.")

    ## EDT of sunvis
    edt_filter = np.logical_and.reduce(
        [
            strictly_decreasing(edtsunvisband[:, :sample_radius, 0]),
            strictly_increasing(edtsunvisband[:, sample_radius + 1 :, 0]),
            (edtsunvisband > 1).any(axis=1)[:, 0],
        ]
    )
    if edt_filter.sum() == 0:
        print(f"All killed by EDT filter.")

    ## Ensure minimum radiance in profile
    radiance_filter = imgband.min(2).min(1) > MINIMUM_RADIANCE

    if radiance_filter.sum() == 0:
        print(f"All killed by Radiance filter.")

    cmb = np.logical_and.reduce([sampler_mask, dog_filter, sunvis_filter, edt_filter, radiance_filter])
    _c = np.where(cmb)[0]

    if len(_c) == 0:
        print(f"Warning: No valid samples")

    # Fitting with scipy
    ####################
    B = len(_c)
    N = imgband.shape[1]
    C = imgband.shape[2]

    xprof = sample_xprof.reshape(1, N).repeat(B, 0).astype(np.float32)
    yprof = imgband[_c].astype(np.float32)
    weight = np.exp(-edtsunvisband[_c] / 5).reshape(B, N, 1).astype(np.float32)
    # print(xprof.shape, yprof.shape, weight.shape)

    init_param = np.zeros((B, 2 + 2 * C), dtype=np.float32)
    init_param[:, 1:2] = 1.0
    init_param[:, 2:5] = yprof.min(axis=1)
    init_param[:, 5:8] = yprof.max(axis=1) - init_param[:, 2:5]

    opt_param, opt_mask = exec_mp(init_param, xprof, yprof, weight)

    # Disable failed line
    cmb[_c[~opt_mask]] = False
    _c = np.where(cmb)[0]
    valid_opt_param = opt_param[opt_mask]

    assert len(_c) == len(valid_opt_param)

    out = dict(
        index=_c,
        opt_param=valid_opt_param,
        centers=sample_cent[_c],
        dirs=sample_grad[_c],
        radius=sample_radius,
        xprof=sample_xprof,
        img=imgband[_c],
        depth=depband[_c],
        normal=norband[_c],
        sunvis=sunvisband[_c],
        skyvis=skyvisband[_c],
        edtsunvis=edtsunvisband[_c],
        dogdepth=dogdepband[_c],
    )
    if "skycam" in lsband:
        out["skycam"] = lsband["skycam"][_c]
    return out


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Process Lit Shadow Band")
    parser.add_argument("input_folder", type=str, help="Input folder (ContextCapture Tiled OBJ)")

    return parser


def _main():
    parser = _get_parser()
    args = parser.parse_args()

    work_dir = Path(args.input_folder)
    imagejson_path = work_dir / "imagedataset.json"

    LITSHADOW_BAND_DIR = osp.join(work_dir, "ls_band")

    # Load Images
    with open(imagejson_path, "r") as f:
        imgdb = json.load(f)

    IMGLIST = list(imgdb["Extrinsic"].keys())

    for imgname in tqdm(IMGLIST):
        lsbandpkl = osp.join(LITSHADOW_BAND_DIR, f"{imgname}.pkl")
        lsbandsolutionpkg = osp.join(LITSHADOW_BAND_DIR, f"{imgname}.sol.pkl")

        if not osp.exists(lsbandpkl):
            print(f"WARN: {imgname} not found in {LITSHADOW_BAND_DIR}")
            continue
        with open(lsbandpkl, "rb") as fp:
            lsband = cPickle.load(fp)
        solution = fit_lsband(lsband)
        with open(lsbandsolutionpkg, "wb") as fp:
            cPickle.dump(solution, fp)


if __name__ == "__main__":
    sys.exit(_main())

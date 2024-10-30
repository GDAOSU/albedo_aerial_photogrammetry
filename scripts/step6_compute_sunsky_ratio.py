import traceback
import sys
from pickle import FALSE
import dateutil.parser
import pandas as pd
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def intensity2rgb(intensity: np.ndarray, ratio=1.0, color="viridis"):
    def adjust_display_range_ratio(data, ratio):
        max_intensity = np.max(data)
        threshold = max_intensity * ratio
        data[data > threshold] = threshold
        return data

    def normalize(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    intensity = adjust_display_range_ratio(intensity, ratio)
    scalar = normalize(intensity)
    cmap = plt.get_cmap(color)

    return cmap(scalar)


def sigmoid_percentile(p, params):
    x0 = params[:, 0:1]
    k = params[:, 1:2]
    b = params[:, 2:5]
    L = params[:, 5:8]

    _x = x0 - np.log(1.0 / p - 1.0) / k
    _y = b + L * p
    return _x, _y


def prefiltering(
    opt_param_band, imgband, normalband, sunvisband, skyvisband, xprofband, omega_sun, use_img_value=True
):
    ########################
    # Filtering by Shape of Sigmoid
    xshadow, _ = sigmoid_percentile(0.02, opt_param_band)
    xlit, _ = sigmoid_percentile(0.98, opt_param_band)
    filter_non_negative = (opt_param_band[:, 2:5] > 0).all(axis=1, keepdims=True)
    # Check the sigmoid funciton, make sure the bandwidth doesn't exceed radius
    filter_complete = np.logical_and.reduce([xshadow > xprofband.min(), xlit < xprofband.max(), xshadow < xlit])
    # filter outlier k with gaussian distribution assumption (<3 sigma)
    logk = np.log(opt_param_band[:, 1:2])
    mu_logk = np.nanmean(logk)
    sigma_logk = np.nanstd(logk)
    filter_logk = np.abs(logk - mu_logk) < 3 * sigma_logk

    prefilter_optmodel = np.logical_and.reduce([filter_non_negative, filter_complete, filter_logk])
    preindex = np.where(prefilter_optmodel)[0]

    # Need to apply the filter to ensure valid index of lit and shadow
    xshadow, yshadow_model = sigmoid_percentile(0.02, opt_param_band[preindex])
    xlit, ylit_model = sigmoid_percentile(0.98, opt_param_band[preindex])
    xidshadow = np.searchsorted(xprofband, xshadow)
    xidlit = np.searchsorted(xprofband, xlit)

    imgband_shadow = np.take_along_axis(imgband[preindex, ...], xidshadow.reshape(-1, 1, 1), 1)
    imgband_lit = np.take_along_axis(imgband[preindex, ...], xidlit.reshape(-1, 1, 1), 1)
    normalband_shadow = np.take_along_axis(normalband[preindex, ...], xidshadow.reshape(-1, 1, 1), 1)
    normalband_lit = np.take_along_axis(normalband[preindex, ...], xidlit.reshape(-1, 1, 1), 1)

    sunvisband_shadow = np.take_along_axis(sunvisband[preindex, :, 0], xidshadow.reshape(-1, 1), 1)
    sunvisband_lit = np.take_along_axis(sunvisband[preindex, :, 0], xidlit.reshape(-1, 1), 1)
    skyvisband_shadow = np.take_along_axis(skyvisband[preindex, :, 0], xidshadow.reshape(-1, 1), 1)
    skyvisband_lit = np.take_along_axis(skyvisband[preindex, :, 0], xidlit.reshape(-1, 1), 1)

    filter_positive_normal = np.logical_and(normalband_lit @ omega_sun > 0, normalband_shadow @ omega_sun > 0)
    # print(f'filter_positive_normal {filter_positive_normal.sum()}')
    filter_same_normal = (normalband_lit * normalband_shadow).sum(axis=2) > 0.9
    # print(f'filter_same_normal {filter_same_normal.sum()}')
    filter_sunvis = np.logical_and(sunvisband_lit > 0.8, sunvisband_shadow < 0.2)
    # print(f'filter_sunvis {filter_sunvis.sum()}')
    filter_skyvis = np.abs(skyvisband_lit - skyvisband_shadow) < 0.2
    # print(f'filter_skyvis {filter_skyvis.sum()}')
    if use_img_value:
        filter_brightness = np.logical_and(
            (imgband_shadow < imgband_lit).all(axis=2, keepdims=False),
            (imgband_shadow > 0.02).all(axis=2, keepdims=False),
        )
    else:
        filter_brightness = np.logical_and(
            (yshadow_model < ylit_model).all(axis=1, keepdims=True), (yshadow_model > 0.02).all(axis=1, keepdims=True)
        )
    # print(f'filter_brightness {filter_brightness.sum()}')
    filter_cmb = np.logical_and.reduce(
        [filter_brightness, filter_positive_normal, filter_same_normal, filter_sunvis, filter_skyvis]
    )
    # print(f'filter_cmb {filter_cmb.sum()}')

    xidlit = xidlit[filter_cmb]
    xidshadow = xidshadow[filter_cmb]
    normalband_pair = normalband_lit[filter_cmb] + normalband_shadow[filter_cmb]
    normalband_pair /= np.linalg.norm(normalband_pair, 2, axis=1, keepdims=True)
    skyvisband_pair = 0.5 * (skyvisband_lit[filter_cmb] + skyvisband_shadow[filter_cmb])
    if use_img_value:
        imgband_shadow = imgband_shadow[filter_cmb.ravel()]
        imgband_lit = imgband_lit[filter_cmb.ravel()]
    else:
        imgband_shadow = yshadow_model[filter_cmb.ravel()]
        imgband_lit = ylit_model[filter_cmb.ravel()]

    sample_filter = prefilter_optmodel.copy()
    sample_filter[preindex] = filter_cmb

    sample_index = np.where(sample_filter)[0]

    # validate
    assert (np.take_along_axis(imgband[sample_index, ...], xidlit.reshape(-1, 1, 1), axis=1) == imgband_lit).all()

    return sample_index, xidlit, xidshadow, imgband_lit, imgband_shadow, normalband_pair, skyvisband_pair


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Compute Lit Shadow Ratio (Sun/Sky Ratio)")
    parser.add_argument("input_folder", type=str, help="Input folder (ContextCapture Tiled OBJ)")
    parser.add_argument("--skymodel", type=str, choices=["Simple", "AO"], default="Simple", help="Sky Model")
    parser.add_argument("--logrithm_phi", action="store_true", help="Use logrithm of phi")
    return parser


def _main():
    parser = _get_parser()
    args = parser.parse_args()

    work_dir = Path(args.input_folder)
    imagejson_path = work_dir / "imagedataset.json"

    LITSHADOW_BAND_DIR = osp.join(work_dir, "ls_band")

    OUTPUT_PHI_JSON = osp.join(LITSHADOW_BAND_DIR, "phi.json")

    OPTION_skyvis_model = args.skymodel
    OPTION_logrithm_phi = args.logrithm_phi

    # CONST_PSI_SUN = np.array([1.0, 0.95, 0.93], dtype=np.float32)
    CONST_PSI_SUN = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Load Images
    with open(imagejson_path, "r") as f:
        imgdb = json.load(f)

    IMGLIST = list(imgdb["Extrinsic"].keys())

    frame_imgname = []
    frame_timestamp = []
    frame_opt_param_band = []
    frame_imgband = []
    frame_sunvisband = []
    frame_skyvisband = []
    frame_normalband = []
    frame_omega_sun = []

    for it, imgname in enumerate(IMGLIST):
        imgmeta = imgdb["ImageMeta"][imgname]
        lsbandpkl = osp.join(LITSHADOW_BAND_DIR, f"{imgname}.pkl")
        lsbandsolutionpkg = osp.join(LITSHADOW_BAND_DIR, f"{imgname}.sol.pkl")
        if not osp.exists(lsbandpkl) or not osp.exists(lsbandsolutionpkg):
            print(f"WARN: {imgname} is not processed")
            continue
        with open(lsbandsolutionpkg, "rb") as fp:
            lsbandsol = cPickle.load(fp)

        omega_sun = np.array([imgmeta[k] for k in ["Sun:LocalPos_x", "Sun:LocalPos_y", "Sun:LocalPos_z"]])
        capture_time = dateutil.parser.parse(imgmeta["DateTime"])

        frame_imgname.append(imgname)
        frame_timestamp.append(capture_time.timestamp())
        frame_opt_param_band.append(lsbandsol["opt_param"])
        frame_imgband.append(lsbandsol["img"])
        frame_normalband.append(lsbandsol["normal"])
        frame_sunvisband.append(lsbandsol["sunvis"])
        frame_skyvisband.append(lsbandsol["skyvis"])
        frame_omega_sun.append(omega_sun)

    xprofband = lsbandsol["xprof"]
    radiusband = lsbandsol["radius"]

    num_frames = len(frame_imgname)
    frame_numsamples = [len(op) for op in frame_opt_param_band]
    print(f"Found {num_frames} frames, constraints: {frame_numsamples}")

    frame_phi_skysun = []
    for fi in tqdm(range(num_frames)):
        print(f"Processing {frame_imgname[fi]}...")
        omega_sun = frame_omega_sun[fi]
        normalband = frame_normalband[fi]
        skyvisband = frame_skyvisband[fi]
        sample_index, xidlit, xidshadow, imgband_lit, imgband_shadow, normalband_pair, skyvisband_pair = prefiltering(
            frame_opt_param_band[fi],
            frame_imgband[fi],
            frame_normalband[fi],
            frame_sunvisband[fi],
            frame_skyvisband[fi],
            xprofband=xprofband,
            omega_sun=omega_sun,
        )

        active_skyvisband = None
        active_skyvisband_pair = None
        if OPTION_skyvis_model == "Simple":
            active_skyvisband = 0.5 * np.maximum(normalband[..., 2], 0) + 0.5
            active_skyvisband_pair = 0.5 * np.maximum(normalband_pair[..., 2], 0) + 0.5
        else:
            active_skyvisband = skyvisband
            active_skyvisband_pair = skyvisband_pair

        ######################################
        # imglit = rho* (psi_sun * max(0, normal @ omega_sun) + psi_sky * skyvis)
        # imgshadow = rho*(psi_sky * skyvis)
        # (imgshadow) / (imglit - imgshadow) = psi_sky * skyvis / (psi_sun * max(0, normal@omega_sun))
        # X = psi_sky / psi_sun = imgshadow * max(0, normal@omega_sun) / (img_lit-imgshadow) / skyvis
        # logX = log(psi_sky) - log(psi_sun) = log(imgshadow) + log(max(0,normal@omega_sun)) - log(img_lit-imgshadow) - log(skyvis)
        ########################################
        Xest = None
        if OPTION_logrithm_phi:
            logPhi = (
                np.log(imgband_shadow).squeeze(1)
                - np.log(imgband_lit - imgband_shadow).squeeze(1)
                + np.expand_dims(np.log(np.maximum(1e-6, normalband_pair @ omega_sun)), 1)
                - np.expand_dims(np.log(active_skyvisband_pair), 1)
            )
            Xest = logPhi
        else:
            Phi = (
                imgband_shadow.squeeze(1)
                * np.expand_dims(np.maximum(1e-6, normalband_pair @ omega_sun), 1)
                / (imgband_lit - imgband_shadow).squeeze(1)
                / np.expand_dims(active_skyvisband_pair, 1)
            )
            Xest = Phi

        if len(Xest) == 0:
            print(f"Warning: No valid samples")
            continue

        from scipy import stats

        if len(Xest.shape) > 3:
            kde = stats.gaussian_kde(Xest.T)
            probX = kde.evaluate(Xest.T)
        else:
            probX = np.ones(Xest.shape[0])

        if OPTION_logrithm_phi:
            ratio_sky_sun = np.exp(logPhi[probX.argmax()])
        else:
            ratio_sky_sun = Phi[probX.argmax()]

        psi_sky = ratio_sky_sun * CONST_PSI_SUN
        frame_phi_skysun.append(ratio_sky_sun)

        # Illumination = psi_sun * np.expand_dims(frame_sunvisband[fi] * np.maximum(
        #     normalband[fi]@omega_sun, 0), 2) + psi_sky*np.expand_dims(active_skyvisband, 2)

        # max_R = frame_imgband[fi].max(0).max(0)
        # R = np.minimum(frame_imgband[fi] / (Illumination+1e-6), max_R)

        if False:  # Visualize Probability
            import open3d as o3d

            cloud = o3d.geometry.PointCloud()
            ptsVec = o3d.utility.Vector3dVector(logPhi)
            cloud.points = ptsVec
            cloud.colors = o3d.utility.Vector3dVector(intensity2rgb(probX)[:, :3])
            cloud_mean, cloud_cov = cloud.compute_mean_and_covariance()
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cloud_cov[0, 0], origin=cloud_mean)
            o3d.visualization.draw_geometries([cloud, axes])

    result = dict(
        frame_name=frame_imgname,
        frame_timestamp=frame_timestamp,
        frame_omega_sun=frame_omega_sun,
        frame_phi=frame_phi_skysun,
        psi_sun=CONST_PSI_SUN,
        skyvis_model=OPTION_skyvis_model,
        estLogPhi=OPTION_logrithm_phi,
    )

    with open(OUTPUT_PHI_JSON, "w") as fp:
        json.dump(result, fp, cls=NumpyEncoder, indent=1)


if __name__ == "__main__":
    sys.exit(_main())

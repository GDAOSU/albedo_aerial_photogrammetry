import sys
import numpy as np
from tqdm import tqdm

import json
import imageio
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np
import time
import pickle
import scipy
from step4_sample_litshadow_band import ensure_finite, check_files, imreader

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from skimage.morphology import erosion, dilation, disk


def maskout_penumbra(binary_sunvis: np.ndarray, erode_radius=10):
    H, W = binary_sunvis.shape[:2]
    lit_mask = binary_sunvis > 0.5
    shadow_mask = binary_sunvis <= 0.5

    # Erode masks
    lit_mask = erosion(lit_mask[..., 0].astype(np.uint8), disk(erode_radius))[..., None] > 0
    shadow_mask = erosion(shadow_mask[..., 0].astype(np.uint8), disk(erode_radius))[..., None] > 0

    return np.logical_or(lit_mask, shadow_mask)


def get_illumination(normal, psi_sun, omega_sun, phi_skysun, skyvis, skyvis_model):
    if skyvis_model == "Simple":
        active_skyvis = 0.5 + 0.5 * np.maximum(normal[..., 2:3], 0)
    else:
        active_skyvis = skyvis

    psi_sky = phi_skysun * psi_sun
    sun_illum = psi_sun * np.expand_dims(np.maximum(normal @ omega_sun, 0), 2)
    sky_illum = psi_sky * active_skyvis

    return sun_illum, sky_illum


def refine_sunvis(
    img,
    init_sunvis,
    sun_illum,
    sky_illum,
    profile_pts,
    height,
    width,
    lambTVRf=0.01,
    lambTVAlpha=0.001,
    winradius=4,
):
    P = sun_illum
    Q = sky_illum
    M = P / (img + 1e-6)
    N = Q / (img + 1e-6)

    initAlpha = init_sunvis.copy()
    initL = initAlpha * P + Q

    refinedAlpha = initAlpha.copy()

    for b in tqdm(profile_pts):
        for i in range(b.shape[0] - 1):  # Loop over segments in boundary-chain
            seg = b[i : i + 2, :]
            segv = b[1, :] - b[0, :]
            segn = np.array([-segv[1], segv[0]])
            footprint = seg[0] + np.arange(-winradius, winradius + 1).reshape(-1, 1) * segn.reshape(1, 2)
            id0 = np.clip(footprint[:, 0], 0, height - 1).astype(np.int32)
            id1 = np.clip(footprint[:, 1], 0, width - 1).astype(np.int32)
            _X0 = refinedAlpha[id0, id1, 0]

            _d = _X0.shape[0]
            _M = []
            _N = []
            D = []
            DM = []
            Pvec = (np.abs(np.linspace(-_d, _d, _d)) ** 2 + 1) * 0.1
            Pvec[0] = 1e7
            Pvec[-1] = 1e7
            Pmat = np.diag(Pvec)
            _D = []
            for _c in range(3):
                _M.append(M[id0, id1, _c])
                _N.append(N[id0, id1, _c])
                _D.append((-np.eye(_d) + np.eye(_d, k=1))[:-1, :])
                DM.append((-np.diag(_M[_c]) + np.diag(_M[_c][1:], k=1))[:-1, :])
            _M = np.concatenate(_M)
            _N = np.concatenate(_N)
            D = scipy.linalg.block_diag(*_D)
            DM = np.vstack(DM)

            _X = np.linalg.solve(
                Pmat + lambTVRf * DM.T @ DM + lambTVAlpha * _D[0].T @ _D[0], Pmat @ _X0 - lambTVRf * DM.T @ D @ _N
            )
            refinedAlpha[id0, id1, 0] = np.clip(_X, 0, 1)
    return refinedAlpha


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Sample Lit Shadow Band")
    parser.add_argument("input_folder", type=str, help="Input folder (ContextCapture Tiled OBJ)")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument("--median_phi", action="store_true", help="Use median phi")
    parser.add_argument("--average_phi", action="store_true", help="Use average phi")
    parser.add_argument("--refine_sunvis", action="store_true", help="Refine sunvis")
    
    return parser


def _main() -> int:
    parser = _get_parser()
    args = parser.parse_args()

    work_dir = Path(args.input_folder)
    output_folder_name = Path(args.output_folder)
    imagejson_path = work_dir / "imagedataset.json"

    litshadow_band_dir = work_dir / "ls_band"

    if not litshadow_band_dir.exists():
        logger.critical("Error: no solution found")
        return 1

    phipack_path = litshadow_band_dir / "phi.json"
    phipack = json.load(phipack_path.open())
    frame_phi = dict(zip(phipack["frame_name"], np.array(phipack["frame_phi"])))

    median_phi = np.median(phipack["frame_phi"], axis=0, keepdims=True)
    mean_phi = np.mean(phipack["frame_phi"], axis=0, keepdims=True)

    frame_omega_sun = dict(zip(phipack["frame_name"], np.array(phipack["frame_omega_sun"])))
    psi_sun = np.array(phipack["psi_sun"])

    output_dir = work_dir / output_folder_name

    out_albedo_dir = output_dir / "albedo"
    out_mask_dir = output_dir / "mask"
    out_sunvis_dir = output_dir / "sunvis"
    out_sunshading_dir = output_dir / "sun_shading"
    out_skyshading_dir = output_dir / "sky_shading"

    out_albedo_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_sunvis_dir.mkdir(parents=True, exist_ok=True)
    out_sunshading_dir.mkdir(parents=True, exist_ok=True)
    out_skyshading_dir.mkdir(parents=True, exist_ok=True)

    # Load Images
    imgdb = json.load(imagejson_path.open())

    IMGLIST = list(imgdb["Extrinsic"].keys())
    verbose = True
    for imgname in tqdm(IMGLIST, desc="Decomposing Albedo"):

        out_albedo_path = out_albedo_dir / (f"{imgname}.exr")
        out_mask_path = out_mask_dir / (f"{imgname}.exr")
        out_sunvis_path = out_sunvis_dir / (f"{imgname}.exr")
        out_sunshading_path = out_sunshading_dir / (f"{imgname}.exr")
        out_skyshading_path = out_skyshading_dir / (f"{imgname}.exr")

        if verbose:
            logger.info(f"Processing {imgname}")
        imgmeta = imgdb["ImageMeta"][imgname]

        paths = dict(
            img=osp.join(work_dir, "image", imgname + ".exr"),
            thumbnail=osp.join(work_dir, "thumbnail", imgname + ".png"),
            sunviscrf=osp.join(work_dir, "sunvis", "crf", imgname + ".png"),
            skyvisratio=osp.join(work_dir, "skyvis", imgname + ".png"),
            normal=osp.join(work_dir, "normal", imgname + ".exr"),
            depth=osp.join(work_dir, "depth", imgname + ".exr"),
            lsbandpkl=osp.join(litshadow_band_dir, f"{imgname}.pkl"),
        )

        omega_sun = np.array(
            [imgmeta[k] for k in ("Sun:LocalPos_x", "Sun:LocalPos_y", "Sun:LocalPos_z")], dtype=np.float32
        )

        if not check_files(paths):
            continue

        data = dict(
            img=imreader(paths["img"]),
            sunviscrf=imreader(paths["sunviscrf"]),
            skyvisratio=imreader(paths["skyvisratio"]),
            depth=imreader(paths["depth"]),
            normal=imreader(paths["normal"]),
            lsband=pickle.load(open(paths["lsbandpkl"], "rb")),
        )
        data["geometry_mask"] = np.isfinite(data["depth"])
        assert data["geometry_mask"].ndim == 3, f"Invalid geometry mask shape {data['geometry_mask'].shape}"

        height, width = data["img"].shape[:2]

        data["normal"] = ensure_finite(data["normal"])
        data["sunpos"] = omega_sun

        _st = time.time()

        current_phi = frame_phi[imgname]
        if args.median_phi:
            current_phi = median_phi
        elif args.average_phi:
            current_phi = mean_phi

        sun_illum, sky_illum = get_illumination(
            data["normal"],
            psi_sun,
            omega_sun,
            current_phi,
            data["skyvisratio"],
            skyvis_model=phipack["skyvis_model"],
        )
        # Detect maximum reflectance in sunvis region
        # max_reflectance = ((data['img'] / (sun_illum+1e-7))*np.expand_dims(data['sunviscrf'],2)).reshape(-1,3).max(0) # Failed, very large
        max_reflectance = (data["img"] * data["sunviscrf"]).reshape(-1, 3).max(0)
        max_reflectance = max_reflectance * 1.2
        if (max_reflectance > 5).any():
            logger.warning(f"Warning: max reflectance is {max_reflectance}")

        if args.refine_sunvis:
            refined_alpha = refine_sunvis(
                data["img"],
                data["sunviscrf"],
                sun_illum,
                sky_illum,
                data["lsband"]["sampler"]["prof_pt"],
                height=height,
                width=width,
            )
        else:
            refined_alpha = data["sunviscrf"]

        # Generate mask from geometry and sunvis(to exclude penumbra)
        albedo_mask = np.logical_and(maskout_penumbra(data["sunviscrf"]), data["geometry_mask"])

        illumination = sun_illum * refined_alpha + sky_illum
        reflectance = data["img"] / (illumination + 1e-7)
        reflectance = np.clip(reflectance, 0, max_reflectance)

        #######
        imageio.imwrite(str(out_albedo_path), reflectance.astype(np.float32))
        imageio.imwrite(str(out_mask_path), albedo_mask.astype(np.float32))
        imageio.imwrite(str(out_sunvis_path), refined_alpha.astype(np.float32))
        imageio.imwrite(str(out_sunshading_path), sun_illum.astype(np.float32))
        imageio.imwrite(str(out_skyshading_path), sky_illum.astype(np.float32))

        if verbose:
            logger.info(f"{time.time()-_st:.3f}s")

    logger.info("Done")


if __name__ == "__main__":
    sys.exit(_main())

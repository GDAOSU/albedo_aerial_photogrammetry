#!/usr/bin/env python

import imageio
import argparse
from tqdm import tqdm
import json
import numpy as np
import os
import os.path as osp
from pathlib import Path

from functools import lru_cache
from skimage import img_as_float
from OpenImageIO import ImageInput, ImageSpec, ImageBuf, ImageBufAlgo, ImageOutput

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


def imwriter(data: np.ndarray, p: str):
    p = str(p)
    _ext = osp.splitext(p)[1]
    if _ext.lower() == ".exr":
        height, width, band = data.shape[0], data.shape[1], data.shape[2]
        exrspec = ImageSpec(width, height, band, str(data.dtype))
        out = ImageOutput.create(p)
        out.open(p, exrspec)
        out.write_image(data)
        out.close()
    else:
        imageio.imwrite(p, data)


class RadialTangentialModel:
    # x = X / Z
    # y = Y / Z
    # r = sqrt(x^2+y^2)
    # x' = x(1+ K1*r^2+K2*r^4+K3*r^6+K4*r^8) + (P1*(r^2+2*x^2)+2*P2*x*y)
    # y' = y(1+ K1*r^2+K2*r^4+K3*r^6+K4*r^8) + (P2*(r^2+2*y^2)+2*P1*x*y)
    # u = w*0.5 + cx + x'*f + x'*B1 + y*B2
    # v = h*0.5 + cy + y'*f

    # P = [X/Z Y/Z 1]
    # K = [f+B1 B2 cx+0.5w]
    #     [0 f cy+0.5h]
    #     [0 0 1]
    def __init__(self, **kwargs) -> None:
        self.width = kwargs.get("width", 0)
        self.height = kwargs.get("height", 0)
        self.f = kwargs.get("f", 1)
        self.cx = kwargs.get("cx", 0)
        self.cy = kwargs.get("cy", 0)
        self.b1 = kwargs.get("b1", 0)
        self.b2 = kwargs.get("b2", 0)
        self.k1 = kwargs.get("k1", 0)
        self.k2 = kwargs.get("k2", 0)
        self.k3 = kwargs.get("k3", 0)
        self.k4 = kwargs.get("k4", 0)
        self.p1 = kwargs.get("p1", 0)
        self.p2 = kwargs.get("p2", 0)

        self.center_undistorted_image = kwargs.get("center_principle", False)

        self.init_matrices()
        self.init_mapping_parameters()
        self.init_bilinear_interpolation()

    def init_matrices(self):
        self.distK = np.array([[self.f + self.b1, self.b2, self.cx], [0, self.f, self.cy], [0, 0, 1]])

        if self.center_undistorted_image:
            self.undistK = np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1]])
        else:
            self.undistK = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]])

    def resize(self, width, height):
        scaler = width / self.width
        scaler2 = height / self.height

        assert scaler == scaler2
        if scaler != scaler2:
            return False

        self.f *= scaler
        self.cx *= scaler
        self.cy *= scaler
        self.b1 *= scaler
        self.b2 *= scaler
        self.width = width
        self.height = height
        self.init_matrices()
        self.init_mapping_parameters()
        self.init_bilinear_interpolation()
        return True

    def init_mapping_parameters(self):
        undistRow, undistCol = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")

        undistUV1 = np.stack([undistCol, undistRow, np.ones_like(undistCol)], axis=2)
        undistxy1 = np.einsum("oc,hwc->hwo", np.linalg.inv(self.undistK), undistUV1)
        r2 = undistxy1[..., 0] ** 2 + undistxy1[..., 1] ** 2
        r_factor = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3 + self.k4 * r2**4
        px_factor = self.p1 * (r2 + 2 * undistxy1[..., 0] ** 2) + 2 * self.p2 * undistxy1[..., 0] * undistxy1[..., 1]
        py_factor = self.p2 * (r2 + 2 * undistxy1[..., 1] ** 2) + 2 * self.p1 * undistxy1[..., 0] * undistxy1[..., 1]

        distxy1 = np.stack(
            [
                r_factor * undistxy1[..., 0] + px_factor,
                r_factor * undistxy1[..., 1] + py_factor,
                np.ones_like(undistxy1[..., 1]),
            ],
            axis=2,
        )
        distUV1 = np.einsum("oc,hwc->hwo", self.distK, distxy1)
        self.distUV = distUV1[..., :2]

    def init_bilinear_interpolation(self):
        # bilinear interpolation
        self.x = self.distUV[..., 0]
        self.y = self.distUV[..., 1]
        self.x0 = np.floor(self.x).astype(np.int32)
        self.x1 = self.x0 + 1
        self.y0 = np.floor(self.y).astype(np.int32)
        self.y1 = self.y0 + 1

        self.x0 = np.minimum(np.maximum(self.x0, 0), self.width - 1)
        self.x1 = np.minimum(np.maximum(self.x1, 0), self.width - 1)
        self.y0 = np.minimum(np.maximum(self.y0, 0), self.height - 1)
        self.y1 = np.minimum(np.maximum(self.y1, 0), self.height - 1)

        self.wa = (self.x1 - self.x) * (self.y1 - self.y)
        self.wb = (self.x1 - self.x) * (self.y - self.y0)
        self.wc = (self.x - self.x0) * (self.y1 - self.y)
        self.wd = (self.x - self.x0) * (self.y - self.y0)

    def undistort_image(self, im):
        Ia = im[self.y0, self.x0]
        Ib = im[self.y1, self.x0]
        Ic = im[self.y0, self.x1]
        Id = im[self.y1, self.x1]

        undistIm = (
            self.wa[..., None] * Ia + self.wb[..., None] * Ib + self.wc[..., None] * Ic + self.wd[..., None] * Id
        )
        return undistIm

    def __repr__(self) -> str:
        return f"RadialTangential(f={self.f},cx={self.cx},cy={self.cy})"


###############################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=osp.basename(__file__))
    parser.add_argument("imagedb", type=str, help="Image Database (JSON File)")
    parser.add_argument("infolder", type=str, help="Input Folder, support .exr and .png files")
    parser.add_argument("outfolder", type=str, help="Output Folder")

    args = parser.parse_args()
    ###############
    HDRIMG_EXTENSION = "exr"
    imgdb_path = Path(args.imagedb)
    in_dir = Path(args.infolder)
    out_dir = Path(args.outfolder)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgdb = json.load(imgdb_path.open("r"))

    IMGLIST = list(imgdb["Extrinsic"].keys())

    camdistmodels = dict()
    for iname, ivalue in imgdb["Intrinsic"].items():
        camdistmodels[iname] = RadialTangentialModel(
            width=ivalue["Width"],
            height=ivalue["Height"],
            f=ivalue["focal"],
            cx=ivalue["cx"],
            cy=ivalue["cy"],
            k1=ivalue["K1"],
            k2=ivalue["K2"],
            k3=ivalue["K3"],
            p1=ivalue["P1"],
            p2=ivalue["P2"],
        )

    for imgname in tqdm(IMGLIST, desc="Undistorting"):
        in_path = osp.join(in_dir, f"{imgname}.{HDRIMG_EXTENSION}")
        out_path = osp.join(out_dir, f"{imgname}.{HDRIMG_EXTENSION}")

        imgext = imgdb["Extrinsic"][imgname]
        camname = imgext["Camera"]
        cammodel = camdistmodels[camname]

        img = imreader(in_path)
        imheight = img.shape[0]
        imwidth = img.shape[1]

        if imheight != cammodel.height or imwidth != cammodel.width:
            cammodel.resize(imwidth, imheight)

        # Undistort HDR Image
        undist_img = cammodel.undistort_image(img).astype(img.dtype)
        # Write to file
        imwriter(undist_img, out_path)

        # C = np.array([imgext["X"], imgext["Y"], imgext["Z"]]).reshape(3, 1)
        # R = np.array(
        #     [
        #         [imgext["r11"], imgext["r12"], imgext["r13"]],
        #         [imgext["r21"], imgext["r22"], imgext["r23"]],
        #         [imgext["r31"], imgext["r32"], imgext["r33"]],
        #     ]
        # )
        # t = -R @ C
        # # View Information
        # normalized_focal = cammodel.undistK[0, 0] / np.maximum(cammodel.width, cammodel.height)
        # normalized_cx = cammodel.undistK[0, 2] / cammodel.width
        # normalized_cy = cammodel.undistK[1, 2] / cammodel.height
        # camstr = f"{t[0,0]} {t[1,0]} {t[2,0]} {R[0,0]} {R[0,1]} {R[0,2]} {R[1,0]} {R[1,1]} {R[1,2]} {R[2,0]} {R[2,1]} {R[2,2]}\n"
        # camstr += f"{normalized_focal} 0 0 1 {normalized_cx} {normalized_cy}\n"
        # open(outldrcampath, "w").write(camstr)
    print("Done")

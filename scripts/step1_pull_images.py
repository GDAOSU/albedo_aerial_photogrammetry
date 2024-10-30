from distutils.command.config import config
import OpenImageIO as oiio
from OpenImageIO import ImageInput, ImageSpec, ImageOutput, ImageBuf, ImageBufAlgo
from skimage.transform import *
from skimage.util import *
from skimage.io import imsave
from skimage.exposure import equalize_adapthist, equalize_hist
import imageio
import argparse
from tqdm import tqdm
import sys
import os
import os.path as osp
import numpy as np

import matplotlib.pyplot as plt

import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def _get_parser():
    parser = argparse.ArgumentParser(description="Create Thumbnail")
    parser.add_argument("input_folder", type=str, help="Input folder (ContextCapture Tiled OBJ)")
    parser.add_argument("--nlevel", type=int, help="Downsample level", default=1)
    parser.add_argument("--synthetic", help="Render thumbnail", default=False, action="store_true")
    return parser

def _main():
    parser = _get_parser()
    args = parser.parse_args()
    
    work_dir = Path(args.input_folder)
    if not work_dir.is_dir():
        logger.critical("Input folder does not exist")
        return 1
    
    
    is_synthetic= args.synthetic
    nLevel = args.nlevel
    
    output_dir = work_dir /'image'
    thumbnail_dir = work_dir / 'thumbnail'
    imagejson_path = work_dir / 'imagedataset.json'
    ###
    dowmsampler = 2**nLevel

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    if not osp.exists(thumbnail_dir):
        os.makedirs(thumbnail_dir)
        
    if is_synthetic:
        run_imageio(output_dir, thumbnail_dir, imagejson_path, dowmsampler)
    else:
        run_oiio(output_dir, thumbnail_dir, imagejson_path, dowmsampler)

def run_imageio(OUTPUT_FOLDER, THUMB_FOLDER, IMAGEJSON_PATH, dowmsampler): # for rendering data with linear space
    with open(IMAGEJSON_PATH, 'r') as f:
        imgdoc = json.load(f)
        
    def equalizer(a): return equalize_hist(a)
    def equalizer(a): return equalize_adapthist(a, clip_limit=0.02)
    
    for k, v in tqdm(imgdoc['Extrinsic'].items(), desc='Create Thumbnail using imageio'):
        inpath = v['RedirectPath']
        outpath = osp.join(OUTPUT_FOLDER, str(k)+'.exr')
        thumbpath = osp.join(THUMB_FOLDER, str(k)+'.png')
        imgdoc['Extrinsic'][k]['ConvertedPath'] = outpath
        imgdoc['Extrinsic'][k]['ThumbnailPath'] = thumbpath
        
        imgraw = imageio.imread(inpath)
        height, width = imgraw.shape[:2]
        exrspec = ImageSpec(width, height, 3, 'float16')
        out = ImageOutput.create(outpath)
        out.open(outpath, exrspec)
        out.write_image(imgraw)
        out.close()

        # Adaptive Enhance (HistgramEqualize) and output 8bit image
        if not osp.exists(thumbpath):
            imghist = equalizer(imgraw/imgraw.max())
            imsave(thumbpath, img_as_ubyte(imghist))

    with open(IMAGEJSON_PATH, 'w') as f:
        json.dump(imgdoc, f, indent=1)
        
def run_oiio(OUTPUT_FOLDER, THUMB_FOLDER, IMAGEJSON_PATH, dowmsampler):
    with open(IMAGEJSON_PATH, 'r') as f:
        imgdoc = json.load(f)
    
    # TODO: This can be read with rawspeed. 
    # OpenImageIO & Libraw failed to read it.
    CROP_X = 4
    CROP_W = 4
    CROP_Y = 4
    CROP_H = 4
    
    config = ImageSpec()
    config["oiio:RawColor"] = 1
    # config["raw:apply_scene_linear_scale"] = 1
    # config["raw:use_camera_matrix"] = 0
    config["raw:ColorSpace"] = "Linear"

    def equalizer(a): return equalize_hist(a)
    def equalizer(a): return equalize_adapthist(a, clip_limit=0.02)

    for k, v in tqdm(imgdoc['Extrinsic'].items(), desc='Create Thumbnail using OpenImageIO'):
        inpath = v['RedirectPath']
        outpath = osp.join(OUTPUT_FOLDER, str(k)+'.exr')
        thumbpath = osp.join(THUMB_FOLDER, str(k)+'.png')
        imgdoc['Extrinsic'][k]['ConvertedPath'] = outpath
        imgdoc['Extrinsic'][k]['ThumbnailPath'] = thumbpath
        
        # Process Dowmsampled Radiance
        # if osp.exists(outpath):
        #     imghandle = ImageInput.open(outpath, config)
        #     imgraw = imghandle.read_image()
        #     imgraw = imgraw[CROP_Y:-CROP_H,CROP_X:-CROP_W,:]
        #     height, width = imgraw.shape[:2]
        # else:
        imghandle = ImageInput.open(inpath, config)
        if imghandle is None:
            print(oiio.geterror())
            continue
        imgraw = imghandle.read_image()
        imgraw = imgraw[CROP_Y:-CROP_H,CROP_X:-CROP_W,:]
        imghandle.close()

        height, width = imgraw.shape[:2]
        height //= dowmsampler
        width //= dowmsampler

        imgraw = resize(imgraw, (height, width))
        exrspec = ImageSpec(width, height, 3, 'float16')
        out = ImageOutput.create(outpath)
        out.open(outpath, exrspec)
        out.write_image(imgraw)
        out.close()
        
        if False:
            # Compare with the result for ISPRS Congress 22
            plt.matshow(imgraw)
            RAWBuf = ImageBuf(inpath)
            RAWBuf = ImageBufAlgo.colorconvert(RAWBuf, RAWBuf.spec().extra_attribs['oiio:ColorSpace'], 'Linear')
            RAWpx = np.asarray(RAWBuf.get_pixels())
            RAWpx = RAWpx[CROP_Y:-CROP_H,CROP_X:-CROP_W,:]
            RAWpx = resize(RAWpx, (height, width))
            width, height = RAWBuf.spec().full_width, RAWBuf.spec().full_height
            exrspec = ImageSpec(width, height, 3, 'float16')
            plt.matshow(RAWpx)
            plt.show()

        # Adaptive Enhance (HistgramEqualize) and output 8bit image
        if not osp.exists(thumbpath):
            imghist = equalizer(imgraw)
            imsave(thumbpath, img_as_ubyte(imghist))

    with open(IMAGEJSON_PATH, 'w') as f:
        json.dump(imgdoc, f, indent=1)

    print('Done.')


if __name__ == '__main__':
    sys.exit(_main())
    
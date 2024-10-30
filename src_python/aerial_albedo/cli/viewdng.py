#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import sys
from OpenImageIO import ImageInput
import imageio

import os
import os.path as osp


if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='input DNG file')
    parser.add_argument('saveas', nargs='?', type=str, help='save as ordinary formats (default PNG)')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    imgpath = args.input
    ext = str.lower(osp.splitext(imgpath)[1])
    try:
        if ext == 'dng':
            imginput = ImageInput.open(imgpath)
            img = imginput.read_image()
        elif ext == 'exr':
            import pyexr
            img = pyexr.read(imgpath)
        else:
            img = imageio.imread(imgpath)
    except Exception as e:
        print(e)
        print(f'Cannot read file {imgpath}')
        sys.exit(1)

    plt.imshow(img)
    plt.show()
    
    outpath = args.saveas
    if outpath:
        if osp.splitext(outpath)[1] == '': outpath += '.png'
        imageio.imwrite(outpath, img)
        

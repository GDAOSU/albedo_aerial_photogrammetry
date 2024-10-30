#!/usr/bin/env python3

import time
import pyproj
import math
from math import radians
import OpenImageIO as oiio
from OpenImageIO import ImageInput
import argparse
from aerial_albedo.calc_sun import get_sun_position
from tqdm import tqdm
import sys
import os
import os.path as osp
from typing import Iterable
from lxml import etree
from datetime import datetime
import numpy as np
import re
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import pymap3d

import logging

DIMNAMES = ["x", "y", "z"]

logger = logging.getLogger(__name__)


def search_file_in_folders(hint: os.PathLike, paths: List[os.PathLike]) -> Optional[os.PathLike]:
    """Search a file in a list of paths

    Args:
        hint (os.PathLike): _description_
        paths (List[os.PathLike]): _description_

    Returns:
        Optional[os.PathLike]: Found accessible path
    """
    hint = Path(hint)
    if hint.exists():
        return hint
    assert isinstance(paths, Iterable)
    if not hint.is_absolute():
        for p in paths:
            _probe_path = Path(p) / hint
            if _probe_path.exists():
                return str(_probe_path)

    namehint = hint.name
    for p in paths:
        _probe_path = Path(p) / namehint
        if _probe_path.exists():
            return str(_probe_path)
    return None


def read_metadict_from_file(imgpath: os.PathLike) -> Dict:
    """Read metadata from image file using OpenImageIO

    Args:
        imgpath (os.PathLike): _description_

    Returns:
        Dict: _description_
    """
    imghandle = ImageInput.open(imgpath)
    if imghandle is None:
        logger.critical(oiio.geterror())
        return None
    spec = imghandle.spec()
    metadict = {}
    for s in spec.extra_attribs:
        if s.name == "GPS:Latitude":
            metadict["GPS:Latitude"] = s.value[0] + s.value[1] / 60.0 + s.value[2] / 3600.0
        elif s.name == "GPS:Longitude":
            metadict["GPS:Longitude"] = s.value[0] + s.value[1] / 60.0 + s.value[2] / 3600.0
        elif s.name == "GPS:Altitude":
            metadict["GPS:Altitude"] = s.value
        elif s.name == "GPS:LatitudeRef":
            metadict["GPS:LatitudeRef"] = s.value
        elif s.name == "GPS:LongitudeRef":
            metadict["GPS:LongitudeRef"] = s.value
        elif s.name == "DateTime":
            _obj = datetime.strptime(s.value, "%Y-%m-%d %H:%M:%S")
            metadict["DateTime"] = s.value
            metadict["Date:Year"] = _obj.year
            metadict["Date:Month"] = _obj.month
            metadict["Date:Day"] = _obj.day
            metadict["Time:Hour"] = _obj.hour
            metadict["Time:Minute"] = _obj.minute
            metadict["Time:Second"] = _obj.second
        elif s.name == "Exif:ISOSpeedRatings":
            metadict["Exif:ISOSpeedRatings"] = s.value
        elif s.name == "FNumber":
            metadict["FNumber"] = s.value
        elif s.name == "ExposureTime":
            metadict["ExposureTime"] = s.value
        elif s.name == "Exif:ShutterSpeedValue":
            metadict["Exif:ShutterSpeedValue"] = s.value
        elif s.name == "Exif:ApertureValue":
            metadict["Exif:ApertureValue"] = s.value

    if "GPS:LatitudeRef" in metadict:
        if metadict["GPS:LatitudeRef"] == "S":
            metadict["GPS:Latitude"] *= -1
    if "GPS:LongitudeRef" in metadict:
        if metadict["GPS:LongitudeRef"] == "W":
            metadict["GPS:Longitude"] *= -1

    del metadict["GPS:LatitudeRef"]
    del metadict["GPS:LongitudeRef"]

    return metadict


def calculate_sun_from_metadict(imgmeta: Dict, default_utc_zone: int = 0) -> Tuple[float, float, float]:
    """Calculate sun position from metadata

    Args:
        imgmeta (Dict): _description_
        default_utc_zone (int, optional): _description_. Defaults to 0.

    Returns:
        Tuple[float, float, float]: Sun position in ENU coordinate
    """
    localtime = imgmeta["Time:Hour"] + imgmeta["Time:Minute"] / 60.0 + imgmeta["Time:Second"] / 3600.0
    latitude = imgmeta["Sol:Latitude"] if "Sol:Latitude" in imgmeta else imgmeta["GPS:Latitude"]
    longitude = imgmeta["Sol:Longitude"] if "Sol:Longitude" in imgmeta else imgmeta["GPS:Longitude"]

    north_offset = 0

    if "Time:UTCZone" in imgmeta:
        utc_zone = imgmeta["Time:UTCZone"]
    else:
        utc_zone = default_utc_zone

    month = imgmeta["Date:Month"]
    day = imgmeta["Date:Day"]
    year = imgmeta["Date:Year"]

    north_offset = 0

    sun = get_sun_position(localtime, latitude, longitude, north_offset, utc_zone, month, day, year)

    # To light direction
    _x = math.cos(radians(sun.elevation)) * math.sin(radians(sun.azimuth))
    _y = math.cos(radians(sun.elevation)) * math.cos(radians(sun.azimuth))
    _z = math.sin(radians(sun.elevation))

    return (_x, _y, _z)


def _get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Convert ContextCapture block xml to image dataset json")
    parser.add_argument("blockxml", type=str, help="Path to block xml file")
    parser.add_argument("outjson", type=str, help="Path to output json file")
    parser.add_argument("--default_redirect", type=str, nargs="+", help="Redirect image path to a new folder")
    parser.add_argument("--default_utczone", type=int, default=0, help="UTC zone")

    return parser


def _main():
    logging.basicConfig(level=logging.INFO)
    
    parser = _get_args()
    args = parser.parse_args()

    blockxml_path = Path(args.blockxml)
    outjson_path = Path(args.outjson)

    if not blockxml_path.exists():
        logger.critical(f"Cannot find block xml file: {blockxml_path}")
        parser.print_usage()
        return 1

    # optional
    DEFAULT_UTCZONE = args.default_utczone
    DEFAULT_IMG_SEARCH_PATH = args.default_redirect if args.default_redirect else []
    DEFAULT_IMG_SEARCH_PATH.insert(0, blockxml_path.parent)
    
    for p, search_path in enumerate(DEFAULT_IMG_SEARCH_PATH):
        logger.info(f"Searching images in [{p}] {search_path}")


    _start = time.time()
    ###########################
    # IO
    tree = etree.parse(blockxml_path.open("r"))
    intrinsicParams = {}
    extrinsicParams = {}
    imgmetaParams = {}

    # Resolve Global SRS table
    SRSpar_node = tree.find("SpatialReferenceSystems")
    SRSpar = {}

    for SRS_node in SRSpar_node.findall("SRS"):
        id = SRS_node.findtext("Id")
        name = SRS_node.findtext("Name")
        definition = SRS_node.findtext("Definition")
        SRSpar[id] = {"name": name, "definition": definition}

    # Blocks : An reconstruction
    block_collection = tree.findall("Block")
    block_node = block_collection[0]  # Only process one block
    name = block_node.findtext("Name")
    blockSRSid = block_node.findtext("SRSId")
    assert blockSRSid in SRSpar, "Cannot find Reference System"
    SRS = SRSpar[blockSRSid]

    _m = re.match(r"ENU:(.*?),(.*)", SRS["definition"])
    isENU = _m is not None
    if isENU:
        # assert len(_m.groups()) == 2, f"Invalid ENU System: {SRS['definition']}"
        lat0, lon0 = [float(t) for t in _m.groups()]
        del _m
        logger.info(f"ENU: {lat0}, {lon0}")
    else:
        datacrs = pyproj.CRS.from_string(SRS["definition"])
        wgs84 = pyproj.CRS.from_string("epsg:4326")
        transformer = pyproj.Transformer.from_crs(datacrs, wgs84)

    logger.info(f"Processing block: {name}")

    # Photogroups: From a same camera
    photogroup_collection = block_node.find("Photogroups").findall("Photogroup")
    for platformID, photogroup_node in enumerate(photogroup_collection):
        # Camera Shared Intrinsics
        cameraname = photogroup_node.findtext("Name")
        cameraWidth = int(photogroup_node.findtext("ImageDimensions/Width"))
        cameraHeight = int(photogroup_node.findtext("ImageDimensions/Height"))
        if photogroup_node.findtext("FocalLength") is not None:
            sensorSize = float(photogroup_node.findtext("SensorSize"))
            focalmm = float(photogroup_node.findtext("FocalLength"))
            focalpx = float(focalmm * cameraWidth / sensorSize)
        elif photogroup_node.findtext("FocalLengthPixels") is not None:
            focalpx = float(photogroup_node.findtext("FocalLengthPixels"))
        principal_x = float(photogroup_node.findtext("PrincipalPoint/x"))
        principal_y = float(photogroup_node.findtext("PrincipalPoint/y"))
        distortion = dict(K1=0, K2=0, K3=0, P1=0, P2=0)
        if photogroup_node.find("Distortion") is not None:
            distortion = {
                k: float(photogroup_node.findtext(f"Distortion/{k}")) for k in ["K1", "K2", "K3", "P1", "P2"]
            }
            distortion["distmodel"] = "radial-tangential"

        skew = float(photogroup_node.findtext("Skew"))
        aspect_ratio = float(photogroup_node.findtext("AspectRatio"))

        normalizer = max(cameraWidth, cameraHeight)
        K = np.array(
            [
                [focalpx / normalizer, skew, principal_x / normalizer],
                [0, aspect_ratio * focalpx / normalizer, principal_y / normalizer],
                [0, 0, 1],
            ]
        )

        intrinsicParams[cameraname] = {
            "Name": cameraname,
            "Width": cameraWidth,
            "Height": cameraHeight,
            "focal": focalpx,
            "cx": principal_x,
            "cy": principal_y,
            "skew": skew,
            "aspect": aspect_ratio,
            "k11": K[0, 0],
            "k12": K[0, 1],
            "k13": K[0, 2],
            "k21": K[1, 0],
            "k22": K[1, 1],
            "k23": K[1, 2],
            "k31": K[2, 0],
            "k32": K[2, 1],
            "k33": K[2, 2],
            **distortion,
        }

        # Images poses and meta
        photo_collection = photogroup_node.findall("Photo")
        for poseID, photo_node in tqdm(
            enumerate(photo_collection), total=len(photo_collection), desc="Scan images in XML"
        ):
            imgid = int(photo_node.findtext("Id"))
            imgpath = photo_node.findtext("ImagePath")
            redirect_imgpath = search_file_in_folders(imgpath, paths=DEFAULT_IMG_SEARCH_PATH)
            assert redirect_imgpath is not None, f"cannot find {imgpath}"
            imgRotMat = np.zeros((3, 3))
            imgCenter = np.zeros(3)
            for i in range(3):
                for j in range(3):
                    imgRotMat[i, j] = float(photo_node.findtext(f"Pose/Rotation/M_{i}{j}"))
                imgCenter[i] = float(photo_node.findtext(f"Pose/Center/{DIMNAMES[i]}"))

            imgmeta = None
            # Read from EXIF
            photoexif_node = photo_node.find("ExifData")
            if photoexif_node is not None:
                imgmeta = {}
                imgmeta["GPS:Latitude"] = float(photoexif_node.findtext("GPS/Latitude"))
                imgmeta["GPS:Longitude"] = float(photoexif_node.findtext("GPS/Longitude"))
                imgmeta["GPS:Altitude"] = float(photoexif_node.findtext("GPS/Altitude"))

                _obj = datetime.strptime(photoexif_node.findtext("DateTimeOriginal"), "%Y-%m-%dT%H:%M:%S")
                imgmeta["DateTime"] = str(_obj)
                imgmeta["Date:Year"] = _obj.year
                imgmeta["Date:Month"] = _obj.month
                imgmeta["Date:Day"] = _obj.day
                imgmeta["Time:Hour"] = _obj.hour
                imgmeta["Time:Minute"] = _obj.minute
                imgmeta["Time:Second"] = _obj.second
                imgmeta["Time:UTCZone"] = 0  # ContextCapture convert it to UTC0

            elif redirect_imgpath is not None:
                imgmeta = read_metadict_from_file(redirect_imgpath)
            else:
                logger.warning("No valid EXIF information fould")

            if isENU:
                # Convert EXT to LLA and put in metadata
                imgLat, imgLon, imgAlt = pymap3d.enu2geodetic(
                    imgCenter[0], imgCenter[1], imgCenter[2], lat0=lat0, lon0=lon0, h0=0
                )
            else:
                imgLat, imgLon = transformer.transform(imgCenter[0], imgCenter[1])
                imgAlt = imgCenter[2]

            imgmeta["Sol:Latitude"] = imgLat
            imgmeta["Sol:Longitude"] = imgLon
            imgmeta["Sol:Altitude"] = imgAlt
            imgname = osp.splitext(osp.basename(imgpath))[0]  # imgname is not unique

            try:
                sunpos_local = calculate_sun_from_metadict(imgmeta, DEFAULT_UTCZONE)
            except:
                sunpos_local = [0, 0, 0]
                logger.warning("Metadata corrupted:", imgname)

            imgmeta.update(
                {
                    "Sun:LocalPos_x": sunpos_local[0],
                    "Sun:LocalPos_y": sunpos_local[1],
                    "Sun:LocalPos_z": sunpos_local[2],
                }
            )
            imgmetaParams[imgid] = imgmeta

            extrinsicParams[imgid] = {
                "PhotoID": imgid,
                "OriginPath": imgpath,
                "RedirectPath": redirect_imgpath,
                "Camera": cameraname,
                "X": imgCenter[0],
                "Y": imgCenter[1],
                "Z": imgCenter[2],
                "r11": imgRotMat[0, 0],
                "r12": imgRotMat[0, 1],
                "r13": imgRotMat[0, 2],
                "r21": imgRotMat[1, 0],
                "r22": imgRotMat[1, 1],
                "r23": imgRotMat[1, 2],
                "r31": imgRotMat[2, 0],
                "r32": imgRotMat[2, 1],
                "r33": imgRotMat[2, 2],
            }

    # Output section

    outjson_path.parent.mkdir(parents=True, exist_ok=True)
    with outjson_path.open("w") as f:
        json.dump(
            {
                "metadata": {"SRS": SRS["definition"]},
                "Intrinsic": intrinsicParams,
                "Extrinsic": extrinsicParams,
                "ImageMeta": imgmetaParams,
            },
            f,
            indent=1,
        )
    _duration = time.time() - _start

    logger.info(f"Done. {len(extrinsicParams)} images processed in {_duration:.2f} seconds")


if __name__ == "__main__":
    sys.exit(_main())

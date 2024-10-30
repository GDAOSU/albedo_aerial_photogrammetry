from glob import glob
import os.path as osp
import sys
from tqdm import tqdm
import json
import trimesh
from lxml import etree
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert ContextCapture Tiled OBJ to Model dataset")
    parser.add_argument("input_folder", type=str, help="Input folder (ContextCapture Tiled OBJ)")
    parser.add_argument("output_json", type=str, help="Output json path")
    return parser


def _main() -> int:
    parser = _get_parser()
    args = parser.parse_args()

    # Args
    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        logger.critical("Input folder does not exist")
        return 1

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)


    metadata_xml_path = input_folder / "metadata.xml"
    data_folder = input_folder / "Data"

    if not metadata_xml_path.exists():
        logger.critical("Cannot find `metadata.xml`")
        return 1
    if not data_folder.exists():
        logger.critical("Cannot find `Data` folder")
        return 1

    tree = etree.parse(metadata_xml_path.open("r"))

    SRSinfo = tree.findtext("SRS")
    SRSorigin = [float(i) for i in tree.findtext("SRSOrigin").split(",")]

    logger.info(SRSinfo)
    logger.info(SRSorigin)

    modellist = []
    modellist += sorted(list(glob(osp.join(data_folder, "**", "*.obj"), recursive=True)))
    modellist += sorted(list(glob(osp.join(data_folder, "**", "*.ply"), recursive=True)))

    modelsinfo = {}

    flag_parse_model = True
    if flag_parse_model:
        vertid_offset = 0
        faceid_offset = 0
    for modelpath in tqdm(modellist, desc="Parse OBJ models"):
        modelname = osp.splitext(osp.basename(modelpath))[0]
        print(modelpath)
        modelprops = dict()
        if flag_parse_model:
            mesh = trimesh.load(modelpath)
            num_vert = len(mesh.vertices)
            num_face = len(mesh.faces)
            bb = mesh.bounding_box.bounds
            modelprops = {
                "NumVert": num_vert,
                "NumFace": num_face,
                "MinBounds": bb[0, :].tolist(),
                "MaxBounds": bb[1, :].tolist(),
                "VertIdOffset": vertid_offset,
                "FaceIdOffset": faceid_offset,
            }
            vertid_offset += num_vert
            faceid_offset += num_face

        modelsinfo[modelname] = {
            "RedirectPath": modelpath,
            "RelPath": osp.relpath(modelpath, input_folder),
            **modelprops,
        }

    with open(output_json_path, "w") as f:
        json.dump({"metadata": {"SRS": SRSinfo, "SRSOrigin": SRSorigin}, "models": modelsinfo}, f, indent=1)

    logger.info("Done.")


if __name__ == "__main__":
    sys.exit(_main())

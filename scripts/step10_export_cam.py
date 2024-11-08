from pathlib import Path
import numpy as np
import os
import os.path as osp
import argparse
import json
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=osp.basename(__file__))
    parser.add_argument("imagedb", type=str, help="Image Database (JSON File)")
    parser.add_argument("outfolder", type=str, help="Output Folder")
    parser.add_argument("--format", type=str, default=".json", choices=[".json", ".png.cam"], help="Output Format")
    parser.add_argument("--undist", action="store_true", help="Strip Distortion")
    args = parser.parse_args()
    ###############
    imgdb_path = Path(args.imagedb)
    out_dir = Path(args.outfolder)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgdb = json.load(imgdb_path.open("r"))

    if args.format == ".png.cam":
        if not args.undist:
            raise ValueError("Only support undistorted camera for .png.cam format")

    cam_intrinsics = dict()
    for iname, ivalue in imgdb["Intrinsic"].items():

        width = ivalue["Width"]
        height = ivalue["Height"]
        f = ivalue.get("focal", 1)
        cx = ivalue.get("cx", 0)
        cy = ivalue.get("cy", 0)
        b1 = ivalue.get("b1", 0)
        b2 = ivalue.get("b2", 0)
        k1 = ivalue.get("k1", 0)
        k2 = ivalue.get("k2", 0)
        k3 = ivalue.get("k3", 0)
        k4 = ivalue.get("k4", 0)
        p1 = ivalue.get("p1", 0)
        p2 = ivalue.get("p2", 0)

        if args.undist:
            cam = dict(
                K=[[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
            )
        else:
            cam = dict(
                f=f,
                cx=cx,
                cy=cy,
                b1=b1,
                b2=b2,
                k1=k1,
                k2=k2,
                k3=k3,
                k4=k4,
                p1=p1,
                p2=p2,
                K=[[f + b1, b2, cx], [0, f, cy], [0, 0, 1]],
            )
        cam.update({"width": width, "height": height})
        cam_intrinsics[iname] = cam

    IMGLIST = list(imgdb["Extrinsic"].keys())
    for imgname in tqdm(IMGLIST, desc="Exporting Camera"):
        out_path = out_dir / f"{imgname}{args.format}"

        imgext = imgdb["Extrinsic"][imgname]
        camname = imgext["Camera"]

        cam_pose = dict()
        cam_pose.update(cam_intrinsics[camname])

        C = np.array([imgext["X"], imgext["Y"], imgext["Z"]]).reshape(3, 1)
        R = np.array(
            [
                [imgext["r11"], imgext["r12"], imgext["r13"]],
                [imgext["r21"], imgext["r22"], imgext["r23"]],
                [imgext["r31"], imgext["r32"], imgext["r33"]],
            ]
        )
        t = -R @ C
        cam_pose["R"] = R.tolist()
        cam_pose["C"] = C.tolist()

        if args.format == ".json":
            out_path.write_text(json.dumps(cam_pose, indent=1))
        elif args.format == ".png.cam":
            # View Information
            width = cam_pose["width"]
            height = cam_pose["height"]
            normalized_focal = cam_pose["K"][0][0] / np.maximum(width, height)
            normalized_cx = cam_pose["K"][0][2] / width
            normalized_cy = cam_pose["K"][1][2] / height
            camstr = f"{t[0,0]} {t[1,0]} {t[2,0]} {R[0,0]} {R[0,1]} {R[0,2]} {R[1,0]} {R[1,1]} {R[1,2]} {R[2,0]} {R[2,1]} {R[2,2]}\n"
            camstr += f"{normalized_focal} 0 0 1 {normalized_cx} {normalized_cy}\n"

            out_path.write_text(camstr)


import os
import shutil
import numpy as np
import json
import cv2
from importlib.resources import files
from panda_express.perception.zed.zed_cam import ZedCamera

def capture(dataset_name: str, capture_id: str = "0"):
    cam = ZedCamera(serial_number=35317039)
    bgra = cam.get_bgra_frame()
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth = cam.get_foundation_depth_frame()
    K = cam.get_intrinsics()[0]
    cam.close()
    extrinsics_path = files("panda_express").joinpath("perception/zed/X_WE.npy")
    extrinsics = np.load(extrinsics_path)
    json_dict = dict()
    for i in range(4):
        for j in range(4):
            json_dict[f"t_{i}{j}"] = extrinsics[i, j]
    json_dict["width"] = rgb.shape[1]
    json_dict["height"] = rgb.shape[0]
    json_dict["fx"] = K[0, 0]
    json_dict["fy"] = K[1, 1]
    json_dict["cx"] = K[0, 2]
    json_dict["cy"] = K[1, 2]
    json_dict["blur_score"] = 0.0

    os.makedirs(f"{dataset_name}/keyframes/cameras/", exist_ok=True)
    os.makedirs(f"{dataset_name}/keyframes/depth/", exist_ok=True)
    os.makedirs(f"{dataset_name}/keyframes/images/", exist_ok=True)
    os.makedirs(f"{dataset_name}/keyframes/confidence/", exist_ok=True)

    with open(f"{dataset_name}/keyframes/cameras/{capture_id}.json", "w") as f:
        json.dump(json_dict, f, indent=2)
       
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{dataset_name}/keyframes/images/{capture_id}.jpg", bgr)
    # Convert depth from meters to millimeters and save as uint16
    depth_mm = (depth * 1000).astype("uint16")
    cv2.imwrite(f"{dataset_name}/keyframes/depth/{capture_id}.png", depth_mm)
    
    print(f"{np.max(depth_mm)=}")
    print(f"{np.min(depth_mm)=}")


    confidence = 255 * np.ones((depth_mm.shape[0], depth_mm.shape[1]), dtype=np.uint8)
    cv2.imwrite(f"{dataset_name}/keyframes/confidence/{capture_id}.png", confidence)


    # Create zip archive of the dataset folder
    # shutil.make_archive(dataset_name, 'zip', dataset_name)
    # print(f"Created {dataset_name}.zip")

if __name__ == "__main__":
        # capture("FR-WB1")
        # capture("FR-OC1")
        capture("FR-KB1")
        # capture("FR-PB1")
        # capture("FR-MM1")

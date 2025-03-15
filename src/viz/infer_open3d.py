import os
import cv2
import PIL
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.backend.terrain_infer import TerrainSegFormer
from src.viz.fast_infer import FastModels
from simple_colors import green
from tqdm.auto import tqdm
from src.backend.utils.std_utils import read_compr_images_from_bag
from src.backend.utils.open3d_utils import save_pcd, get_pcd_colors_from_image


def image_to_pointcloud(image: PIL.Image.Image, fm: FastModels, vis: str = "raw"):
    """
    vis: str
        One of ["raw", "nspred"]  
    """
    full_pred_seg, accumulated_results, terrain_results, depth_results, gsam_results = fm.predict_new(image)
    cv2_image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2_pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_image_np, full_pred_seg)
    pilnp_pred_overlay = cv2.cvtColor(cv2_pred_overlay, cv2.COLOR_BGR2RGB)
    pcd = depth_results[1]
    pcd.colors = get_pcd_colors_from_image(pilnp_pred_overlay if vis == "nspred" else image)
    return pcd


def save_bagfile_to_pcds(bag_file, image_topic, pcd_dir, fm: FastModels, vis: str = "raw"):
    os.makedirs(pcd_dir, exist_ok=True)
    print(green(f"Reading images from the bag file: {bag_file}", ["bold"]))
    images = read_compr_images_from_bag(bag_file, image_topic)
    print(green(f"Number of images read: {len(images)}", ["bold"]))

    for i, image in enumerate(tqdm(images, desc="Saving pcds", total=len(images))):
        pcd = image_to_pointcloud(image, fm, vis)
        save_pcd(pcd, os.path.join(pcd_dir, f"{i:06}.pcd"))

    print(green(f"Saved pcds to {pcd_dir}", ["bold"]))


if __name__ == "__main__":
    fm = FastModels()
    save_bagfile_to_pcds(bag_file="test/demonstrations/2.bag",
                         image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
                         pcd_dir="test/demonstrations/pcds",
                         fm=fm,
                         vis="nspred")

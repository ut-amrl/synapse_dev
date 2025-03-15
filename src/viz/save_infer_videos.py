import os
import cv2
import rosbag
from PIL import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.backend.terrain_infer import TerrainSegFormer
import matplotlib.pyplot as plt
from src.viz.fast_infer import FastModels
from simple_colors import red, green
from tqdm.auto import tqdm


def bag_to_video(bagfile_path, save_dirpath, image_topic, FPS=5):
    os.makedirs(save_dirpath, exist_ok=True)
    # Open the ROS bag file
    bag = rosbag.Bag(bagfile_path, "r")
    bridge = CvBridge()
    fm = FastModels()

    # Video writers
    # Raw video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    # terrain
    writer_terrain = None
    # gsam
    writer_gsam = None
    # depth
    writer_depth = None
    # full
    writer_full = None

    print(green(f"Processing number of messages: {bag.get_message_count(image_topic)}", ["bold"]))

    for i, (topic, msg, t) in enumerate(tqdm(bag.read_messages(topics=[image_topic]), desc="Processing Bag", total=bag.get_message_count(image_topic))):
        cv2_frame_np = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        pil_img = Image.fromarray(cv2.cvtColor(cv2_frame_np, cv2.COLOR_BGR2RGB))
        full_pred_seg, accumulated_results, terrain_results, depth_results, gsam_results = fm.predict_new(pil_img)

        # Initialize the raw video writer
        if writer is None:
            h, w = cv2_frame_np.shape[:2]
            writer = cv2.VideoWriter(os.path.join(save_dirpath, "raw.mp4"), fourcc, FPS, (w, h), True)
        writer.write(cv2_frame_np)

        # terrain
        pred_img_terrain = terrain_results[0]
        pred_img_terrain = cv2.cvtColor(np.asarray(pred_img_terrain), cv2.COLOR_RGB2BGR)
        if writer_terrain is None:
            h, w = pred_img_terrain.shape[:2]
            writer_terrain = cv2.VideoWriter(os.path.join(save_dirpath, "terrain.mp4"), fourcc, FPS, (w, h), True)
        writer_terrain.write(pred_img_terrain)

        # gsam
        gsam_ann_img = gsam_results[0]
        cv2_gsam_pred = cv2.cvtColor(np.asarray(gsam_ann_img), cv2.COLOR_RGB2BGR)
        if writer_gsam is None:
            h, w = cv2_gsam_pred.shape[:2]
            writer_gsam = cv2.VideoWriter(os.path.join(save_dirpath, "gsam.mp4"), fourcc, FPS, (w, h), True)
        writer_gsam.write(cv2_gsam_pred)

        # depth
        pred_metric_depth = depth_results[0]
        normalized_image = cv2.normalize(pred_metric_depth, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        colormap = plt.get_cmap('inferno')
        colored_image = colormap(normalized_image)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        if writer_depth is None:
            h, w = colored_image.shape[:2]
            writer_depth = cv2.VideoWriter(os.path.join(save_dirpath, "depth.mp4"), fourcc, FPS, (w, h), True)
        writer_depth.write(colored_image)

        # full
        cv2_pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_frame_np, full_pred_seg)
        if writer_full is None:
            h, w = cv2_pred_overlay.shape[:2]
            writer_full = cv2.VideoWriter(os.path.join(save_dirpath, "full.mp4"), fourcc, FPS, (w, h), True)
        writer_full.write(cv2_pred_overlay)

    # Release resources
    writer.release()
    writer_terrain.release()
    writer_gsam.release()
    writer_depth.release()
    writer_full.release()
    bag.close()


if __name__ == "__main__":
    bagpath = "test/demonstrations/2.bag"
    save_dirpath = "test/demonstrations/VIDEOS/2"
    bag_to_video(bagpath, save_dirpath, "/zed2i/zed_node/left/image_rect_color/compressed")

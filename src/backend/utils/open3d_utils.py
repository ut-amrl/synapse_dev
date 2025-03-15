import os
import open3d as o3d
import PIL
import numpy as np
import sys
from tqdm import tqdm
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


def get_pcd_colors_from_image(pil_img: PIL.Image.Image):
    colors = np.asarray(pil_img).reshape(-1, 3) / 255.0
    return o3d.utility.Vector3dVector(colors)


def pcd_from_np(pc_np, color_rgb_list=None):
    pcd = o3d.geometry.PointCloud()
    xyz = pc_np[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    c = [1, 0.647, 0] if color_rgb_list is None else color_rgb_list   # default orange color
    pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(xyz), 1)))
    return pcd


def save_pcd(pointcloud, filename="pointcloud.pcd"):
    """Save an Open3D point cloud to a PCD file."""
    o3d.io.write_point_cloud(filename, pointcloud)


def load_pcd(filename="pointcloud.pcd"):
    """Load a PCD file as an Open3D point cloud."""
    return o3d.io.read_point_cloud(filename)


def visualize_pcd(pcd, camera_params_jsonpath="config/open3d_cameraview_params.json"):
    vis, ctr = init_vis(point_size=0.85)
    vis.add_geometry(pcd)
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    # vis.add_geometry(axis_pcd)
    if camera_params_jsonpath is not None:
        param = o3d.io.read_pinhole_camera_parameters(camera_params_jsonpath)
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    vis.run()    # keeping the visualization window open until the user closes it
    vis.destroy_window()


def init_vis(point_size=0.85):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    ctr = vis.get_view_control()
    return vis, ctr


def write_o3d_camera_params(pcd_path, jsonpath):
    """
    Using a pcd file, calibrate the view and then write the camera parameters to a json file.
    """
    pcd = load_pcd(pcd_path)
    vis, ctr = init_vis(point_size=0.85)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(jsonpath, camera_params)
    vis.destroy_window()


def save_pcd_viz_images(pcd_dir, output_dir, camera_params_jsonpath="config/open3d_cameraview_params.json", delay=0.1):
    os.makedirs(output_dir, exist_ok=True)
    pcd_files = sorted(os.listdir(pcd_dir))
    pcds = [load_pcd(os.path.join(pcd_dir, f)) for f in pcd_files]
    vis, ctr = init_vis(point_size=0.85)
    if camera_params_jsonpath is not None:
        param = o3d.io.read_pinhole_camera_parameters(camera_params_jsonpath)

    for i, pcd in enumerate(tqdm(pcds, desc="pcds", total=len(pcds))):
        vis.clear_geometries()
        vis.add_geometry(pcd)
        if camera_params_jsonpath is not None:
            ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)
        vis.capture_screen_image(os.path.join(output_dir, pcd_files[i].replace(".pcd", ".png")))
    vis.destroy_window()


if __name__ == "__main__":
    # pcd = load_pcd("test/demonstrations/pcds/000000.pcd")
    # visualize_pcd(pcd)

    save_pcd_viz_images(pcd_dir="test/demonstrations/pcds", output_dir="test/demonstrations/pcd_imgs")

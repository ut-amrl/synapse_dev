#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(os.path.join(os.path.dirname(__file__), "../.."))


import rospy
import numpy as np
import torch
import PIL
from PIL import Image
import cv2
from sensor_msgs.msg import Image as ROSImage, CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import open3d as o3d
import struct

from src.backend.depth2 import DepthAny2
from third_party.jackal_calib import JackalLidarCamCalibration


class DepthNode:
    def __init__(self):
        rospy.init_node('depth_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw/compressed')
        self.test_image_path = rospy.get_param('~test_image_path', 'test/000000.png')
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        self.use_test_mode = rospy.get_param('~use_test_mode', False)

        # Initialize models
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            self.metric_depth_model = DepthAny2()
            self.lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)

        # ROS setup
        self.bridge = CvBridge()

        # Publishers
        self.depth_pub = rospy.Publisher('/depth/depth_image', ROSImage, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/depth/pointcloud', PointCloud2, queue_size=1)

        # Subscriber or timer for test mode
        if not self.use_test_mode:
            if 'compressed' in self.image_topic:
                self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback)
            else:
                self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.image_callback)
        else:
            # Test mode - publish at fixed rate
            self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)

        rospy.loginfo(f"Depth node initialized - Test mode: {self.use_test_mode}")
        if self.use_test_mode:
            rospy.loginfo(f"Test image path: {self.test_image_path}")
            rospy.loginfo(f"Publish rate: {self.publish_rate}")

    def image_callback(self, msg):
        try:
            # Convert ROS image to PIL
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

            pil_image = Image.fromarray(cv_image)

            # Process with depth model
            self.process_and_publish(pil_image, msg.header)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def timer_callback(self, event):
        try:
            rospy.loginfo("Depth timer callback triggered - processing test image")
            # Load test image
            pil_image = Image.open(self.test_image_path)
            rospy.loginfo(f"Loaded test image: {pil_image.size}")

            # Create header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera"

            # Process with depth model
            self.process_and_publish(pil_image, header)
            rospy.loginfo("Published depth results")

        except Exception as e:
            rospy.logerr(f"Error in timer callback: {e}")

    @torch.inference_mode()
    def depth_metric_pred(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            pil_img_np = np.asarray(image.convert('RGB'))
            cv2_img = cv2.cvtColor(pil_img_np, cv2.COLOR_RGB2BGR)
            depth_arr = self.metric_depth_model.predict(cv2_img)
            pred = depth_arr.squeeze()
            resized_pred = Image.fromarray(pred).resize((self.lidar_cam_calib.img_width, self.lidar_cam_calib.img_height), Image.NEAREST)
        return np.asarray(resized_pred)

    @torch.inference_mode()
    def get_pc_from_depth(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            z = self.depth_metric_pred(image)
            image = image.convert('RGB')
            FX = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][0, 0]
            FY = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][1, 1]
            CX = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][0, 2]
            CY = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][1, 2]
            K = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix']
            d = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['dist_coeffs']
            R = np.eye(3)
            x, y = np.meshgrid(np.arange(self.lidar_cam_calib.img_width), np.arange(self.lidar_cam_calib.img_height))
            # undistort pixel coordinates
            pcs_coords = np.stack((x.flatten(), y.flatten()), axis=-1).astype(np.float64)
            undistorted_pcs_coords = cv2.undistortPoints(pcs_coords.reshape(1, -1, 2), K, d, R=R, P=K)
            undistorted_pcs_coords = np.swapaxes(undistorted_pcs_coords, 0, 1).squeeze().reshape((-1, 2))
            x, y = np.split(undistorted_pcs_coords, 2, axis=1)
            x = x.reshape(self.lidar_cam_calib.img_height, self.lidar_cam_calib.img_width)
            y = y.reshape(self.lidar_cam_calib.img_height, self.lidar_cam_calib.img_width)
            # back project (along the camera ray) the pixel coordinates to 3D using the depth
            x = (x - CX) / FX
            y = (y - CY) / FY
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.asarray(image).reshape(-1, 3) / 255.0
            # convert to open3d point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return z, pcd

    def o3d_to_ros_pointcloud2(self, pcd, header):
        """Convert Open3D point cloud to ROS PointCloud2 message"""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Create PointCloud2 message
        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.is_dense = False
        pc2_msg.is_bigendian = False

        # Define fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pc2_msg.fields = fields
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width

        # Pack data
        buffer = []
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            buffer.append(struct.pack('fffI', x, y, z, rgb))

        pc2_msg.data = b''.join(buffer)
        return pc2_msg

    @torch.inference_mode()
    def process_and_publish(self, image: PIL.Image.Image, header: Header):
        z, pcd = self.get_pc_from_depth(image)

        try:
            # Convert depth to ROS message
            depth_msg = self.bridge.cv2_to_imgmsg(z.astype(np.float32), "32FC1")
            depth_msg.header = header

            # Convert point cloud to ROS message
            pc2_msg = self.o3d_to_ros_pointcloud2(pcd, header)

            # Publish
            self.depth_pub.publish(depth_msg)
            self.pointcloud_pub.publish(pc2_msg)

        except Exception as e:
            rospy.logerr(f"Error publishing depth results: {e}")


if __name__ == '__main__':
    try:
        node = DepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

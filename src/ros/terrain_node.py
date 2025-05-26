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
from sensor_msgs.msg import Image as ROSImage, CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge
import threading
import time

from src.backend.terrain_infer import TerrainSegFormer


class TerrainNode:
    def __init__(self):
        rospy.init_node('terrain_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw/compressed')
        self.test_image_path = rospy.get_param('~test_image_path', 'test/000000.png')
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        self.use_test_mode = rospy.get_param('~use_test_mode', False)

        # Initialize model
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            self.terrain_model = TerrainSegFormer()
            self.terrain_model.load_model_inference()

        # ROS setup
        self.bridge = CvBridge()

        # Publishers
        self.pred_img_pub = rospy.Publisher('/terrain/pred_img', ROSImage, queue_size=1)
        self.pred_seg_pub = rospy.Publisher('/terrain/pred_seg', ROSImage, queue_size=1)

        # Subscriber or timer for test mode
        if not self.use_test_mode:
            if 'compressed' in self.image_topic:
                self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback)
            else:
                self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.image_callback)
        else:
            # Test mode - publish at fixed rate
            self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)

        rospy.loginfo(f"Terrain node initialized - Test mode: {self.use_test_mode}")
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

            # Process with terrain model
            self.process_and_publish(pil_image, msg.header)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def timer_callback(self, event):
        try:
            rospy.loginfo("Timer callback triggered - processing test image")
            # Load test image
            pil_image = Image.open(self.test_image_path)
            rospy.loginfo(f"Loaded test image: {pil_image.size}")

            # Create header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera"

            # Process with terrain model
            self.process_and_publish(pil_image, header)
            rospy.loginfo("Published terrain results")

        except Exception as e:
            rospy.logerr(f"Error in timer callback: {e}")

    @torch.inference_mode()
    def process_and_publish(self, image: PIL.Image.Image, header: Header):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            pred_img, pred_seg = self.terrain_model.predict_new(image)
            pred_seg_np = pred_seg.squeeze().cpu().numpy()

        try:
            # Convert pred_img to ROS message
            if isinstance(pred_img, PIL.Image.Image):
                pred_img_cv = cv2.cvtColor(np.array(pred_img), cv2.COLOR_RGB2BGR)
            else:
                pred_img_cv = pred_img

            pred_img_msg = self.bridge.cv2_to_imgmsg(pred_img_cv, "bgr8")
            pred_img_msg.header = header

            # Convert pred_seg to ROS message (preserve original values as 16-bit)
            pred_seg_cv = pred_seg_np.astype(np.uint16)
            pred_seg_msg = self.bridge.cv2_to_imgmsg(pred_seg_cv, "mono16")
            pred_seg_msg.header = header

            # Publish
            self.pred_img_pub.publish(pred_img_msg)
            self.pred_seg_pub.publish(pred_seg_msg)

        except Exception as e:
            rospy.logerr(f"Error publishing terrain results: {e}")


if __name__ == '__main__':
    try:
        node = TerrainNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

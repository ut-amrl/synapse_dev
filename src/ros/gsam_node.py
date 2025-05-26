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
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension, String, UInt8MultiArray
from cv_bridge import CvBridge
import json

from src.backend.gsam import GSAM as FastGSAM


class GSAMNode:
    def __init__(self):
        rospy.init_node('gsam_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw/compressed')
        self.test_image_path = rospy.get_param('~test_image_path', 'test/000000.png')
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        self.use_test_mode = rospy.get_param('~use_test_mode', False)

        # Initialize model
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            self.gsam = FastGSAM(box_threshold=0.25, text_threshold=0.25, nms_threshold=0.6)
            self.gsam_object_dict = {
                "person": 0,
                "bush": 1,
                "car": 2,
                "pole": 3,
                "entrance": 4,
                "staircase": 5
            }

        # ROS setup
        self.bridge = CvBridge()

        # Publishers
        self.ann_img_pub = rospy.Publisher('/gsam/annotated_image', ROSImage, queue_size=1)
        self.detections_pub = rospy.Publisher('/gsam/detections', String, queue_size=1)
        self.masks_pub = rospy.Publisher('/gsam/masks', UInt8MultiArray, queue_size=1)

        # Subscriber or timer for test mode
        if not self.use_test_mode:
            if 'compressed' in self.image_topic:
                self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback)
            else:
                self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.image_callback)
        else:
            # Test mode - publish at fixed rate
            self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)

        rospy.loginfo(f"GSAM node initialized - Test mode: {self.use_test_mode}")
        if self.use_test_mode:
            rospy.loginfo(f"Test image path: {self.test_image_path}")
            rospy.loginfo(f"Publish rate: {self.publish_rate}")

    def image_callback(self, msg):
        try:
            # Convert ROS image to CV2
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process with GSAM model
            self.process_and_publish(cv_image, msg.header)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def timer_callback(self, event):
        try:
            rospy.loginfo("GSAM timer callback triggered - processing test image")
            # Load test image
            pil_image = Image.open(self.test_image_path)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            rospy.loginfo(f"Loaded test image: {pil_image.size}")

            # Create header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera"

            # Process with GSAM model
            self.process_and_publish(cv_image, header)
            rospy.loginfo("Published GSAM results")

        except Exception as e:
            rospy.logerr(f"Error in timer callback: {e}")

    @torch.inference_mode()
    def process_and_publish(self, cv_image, header: Header):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            ann_img, dets, per_class_mask = self.gsam.predict_and_segment_on_image(cv_image, list(self.gsam_object_dict.keys()))

        try:
            # Convert annotated image to ROS message
            ann_img_msg = self.bridge.cv2_to_imgmsg(ann_img, "bgr8")
            ann_img_msg.header = header

            # Convert detections to JSON string
            # Extract data from supervision.Detections object
            detections_data = {
                "xyxy": dets.xyxy.tolist() if dets.xyxy is not None else [],
                "confidence": dets.confidence.tolist() if dets.confidence is not None else [],
                "class_id": dets.class_id.tolist() if dets.class_id is not None else [],
                "object_dict": self.gsam_object_dict,
                "timestamp": header.stamp.to_sec()
            }
            detections_msg = String()
            detections_msg.data = json.dumps(detections_data)

            # Convert per_class_mask to UInt8MultiArray (preserve boolean as 0/1)
            masks_msg = UInt8MultiArray()
            if per_class_mask is not None and per_class_mask.size > 0:
                # Convert boolean to uint8 (0/1) and flatten
                masks_flat = per_class_mask.astype(np.uint8).flatten()
                masks_msg.data = masks_flat.tolist()

                # Store dimensions for reconstruction
                masks_msg.layout.dim = [
                    MultiArrayDimension(label="classes", size=per_class_mask.shape[0], stride=per_class_mask.size),
                    MultiArrayDimension(label="height", size=per_class_mask.shape[1], stride=per_class_mask.shape[1] * per_class_mask.shape[2]),
                    MultiArrayDimension(label="width", size=per_class_mask.shape[2], stride=per_class_mask.shape[2])
                ]
            else:
                # Empty masks
                masks_msg.data = []
                masks_msg.layout.dim = [
                    MultiArrayDimension(label="classes", size=0, stride=0),
                    MultiArrayDimension(label="height", size=0, stride=0),
                    MultiArrayDimension(label="width", size=0, stride=0)
                ]

            # Publish
            self.ann_img_pub.publish(ann_img_msg)
            self.detections_pub.publish(detections_msg)
            self.masks_pub.publish(masks_msg)

        except Exception as e:
            rospy.logerr(f"Error publishing GSAM results: {e}")


if __name__ == '__main__':
    try:
        node = GSAMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

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
import json
import os
from sensor_msgs.msg import Image as ROSImage, CompressedImage, PointCloud2
from std_msgs.msg import Header, Float32MultiArray, String, UInt8MultiArray
from cv_bridge import CvBridge
from pykdtree.kdtree import KDTree
import open3d as o3d
import struct

from third_party.jackal_calib import JackalLidarCamCalibration


class ProcessorNode:
    def __init__(self):
        rospy.init_node('processor_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw/compressed')
        self.test_image_path = rospy.get_param('~test_image_path', 'test/000000.png')
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        self.do_car = rospy.get_param('~do_car', True)
        self.use_test_mode = rospy.get_param('~use_test_mode', False)

        # Initialize calibration
        self.lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)

        # Setup image dump directory
        self.dump_dir = os.path.join(os.path.dirname(__file__))
        os.makedirs(self.dump_dir, exist_ok=True)

        # GSAM object dictionary (should match gsam_node)
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

        # Store latest messages from each topic
        self.latest_image = None
        self.latest_terrain_seg = None
        self.latest_depth = None
        self.latest_pointcloud = None
        self.latest_gsam_masks = None

        # Publishers
        self.result_pub = rospy.Publisher('/processor/result', ROSImage, queue_size=1)
        self.prediction_mask_pub = rospy.Publisher('/processor/prediction_mask', ROSImage, queue_size=1)  # Main output!
        self.terrain_mask_pub = rospy.Publisher('/processor/terrain_mask', ROSImage, queue_size=1)
        self.distance_arrays_pub = rospy.Publisher('/processor/distance_arrays', String, queue_size=1)
        self.in_the_way_pub = rospy.Publisher('/processor/in_the_way', ROSImage, queue_size=1)

        # Independent subscribers - no synchronization
        self.terrain_seg_sub = rospy.Subscriber('/terrain/pred_seg', ROSImage, self.terrain_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber('/depth/depth_image', ROSImage, self.depth_callback, queue_size=1)
        self.pointcloud_sub = rospy.Subscriber('/depth/pointcloud', PointCloud2, self.pointcloud_callback, queue_size=1)
        self.gsam_masks_sub = rospy.Subscriber('/gsam/masks', UInt8MultiArray, self.gsam_callback, queue_size=1)

        # Image subscriber for original image
        if not self.use_test_mode:
            if 'compressed' in self.image_topic:
                self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1)
            else:
                self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.image_callback, queue_size=1)

        # Timer for processing at fixed rate
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)

        rospy.loginfo(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Processor node initialized - Test mode: {self.use_test_mode}")
        if self.use_test_mode:
            rospy.loginfo(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Test image path: {self.test_image_path}")
            rospy.loginfo(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Publish rate: {self.publish_rate}")

    def image_callback(self, msg):
        """Store latest image message"""
        self.latest_image = msg

    def terrain_callback(self, msg):
        """Store latest terrain segmentation message"""
        self.latest_terrain_seg = msg

    def depth_callback(self, msg):
        """Store latest depth message"""
        self.latest_depth = msg

    def pointcloud_callback(self, msg):
        """Store latest pointcloud message"""
        self.latest_pointcloud = msg

    def gsam_callback(self, msg):
        """Store latest GSAM masks message"""
        self.latest_gsam_masks = msg

    def timer_callback(self, event):
        """Process latest available data at fixed rate"""
        try:
            # Check if we have all required data
            if self.use_test_mode:
                # In test mode, use test image
                pil_image = Image.open(self.test_image_path)
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "camera"
            else:
                # In normal mode, need image from camera
                if self.latest_image is None:
                    rospy.logwarn_throttle(5, "Processor: No image received yet")
                    return

                # Convert image message to PIL
                if isinstance(self.latest_image, CompressedImage):
                    np_arr = np.frombuffer(self.latest_image.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                else:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "rgb8")

                pil_image = Image.fromarray(cv_image)
                header = self.latest_image.header

            # Check for other required data
            if self.latest_terrain_seg is None:
                rospy.logwarn_throttle(5, "Processor: No terrain segmentation received yet")
                return
            if self.latest_depth is None:
                rospy.logwarn_throttle(5, "Processor: No depth data received yet")
                return
            if self.latest_pointcloud is None:
                rospy.logwarn_throttle(5, "Processor: No pointcloud received yet")
                return
            if self.latest_gsam_masks is None:
                rospy.logwarn_throttle(5, "Processor: No GSAM masks received yet")
                return

            rospy.loginfo("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Processor: Processing with latest available data")
            self.process_messages(pil_image, self.latest_terrain_seg, self.latest_depth,
                                  self.latest_pointcloud, self.latest_gsam_masks, header)
            rospy.loginfo("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Processor: Published final results")

        except Exception as e:
            rospy.logerr(f"Error in timer callback: {e}")

    def ros_pointcloud2_to_o3d(self, pc_msg):
        """Convert ROS PointCloud2 message to Open3D point cloud"""
        try:
            # Parse point cloud data
            points = []
            colors = []

            point_step = pc_msg.point_step
            row_step = pc_msg.row_step
            width = pc_msg.width
            height = pc_msg.height

            rospy.loginfo(f"PointCloud info: width={width}, height={height}, point_step={point_step}, row_step={row_step}, data_len={len(pc_msg.data)}")

            # Check if we have enough data
            expected_data_len = height * row_step
            if len(pc_msg.data) < expected_data_len:
                rospy.logwarn(f"Insufficient pointcloud data: expected {expected_data_len}, got {len(pc_msg.data)}")
                # Create empty pointcloud
                pcd = o3d.geometry.PointCloud()
                return pcd

            # Parse field information to understand the data layout
            field_names = [field.name for field in pc_msg.fields]
            field_offsets = {field.name: field.offset for field in pc_msg.fields}
            field_datatypes = {field.name: field.datatype for field in pc_msg.fields}

            rospy.loginfo(f"PointCloud fields: {field_names}")
            rospy.loginfo(f"Field offsets: {field_offsets}")

            # Try to parse based on available fields
            for i in range(0, len(pc_msg.data), point_step):
                if i + point_step > len(pc_msg.data):
                    break

                try:
                    # Extract x, y, z (assuming they're floats at the beginning)
                    if i + 12 <= len(pc_msg.data):
                        x, y, z = struct.unpack('fff', pc_msg.data[i:i + 12])

                        # Skip invalid points
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            continue

                        points.append([x, y, z])

                        # Try to extract RGB if available
                        if 'rgb' in field_names and i + 16 <= len(pc_msg.data):
                            try:
                                rgb_packed = struct.unpack('I', pc_msg.data[i + 12:i + 16])[0]
                                # Extract RGB from packed integer
                                r = (rgb_packed >> 16) & 0xFF
                                g = (rgb_packed >> 8) & 0xFF
                                b = rgb_packed & 0xFF
                                colors.append([r / 255.0, g / 255.0, b / 255.0])
                            except:
                                colors.append([0.5, 0.5, 0.5])  # Default gray
                        else:
                            colors.append([0.5, 0.5, 0.5])  # Default gray

                except struct.error as e:
                    rospy.logwarn(f"Struct unpack error at index {i}: {e}")
                    continue
                except Exception as e:
                    rospy.logwarn(f"Error parsing point at index {i}: {e}")
                    continue

            rospy.loginfo(f"Parsed {len(points)} points from pointcloud")

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            if len(points) > 0:
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                if len(colors) == len(points):
                    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

            return pcd

        except Exception as e:
            rospy.logerr(f"Error converting pointcloud: {e}")
            # Return empty pointcloud on error
            pcd = o3d.geometry.PointCloud()
            return pcd

    def reconstruct_masks(self, masks_msg):
        """Reconstruct per_class_mask from Float32MultiArray message"""
        try:
            rospy.loginfo(f"Masks message type: {type(masks_msg)}")
            rospy.loginfo(f"Masks data length: {len(masks_msg.data) if hasattr(masks_msg, 'data') else 'no data attr'}")

            if not hasattr(masks_msg, 'data') or not masks_msg.data:
                rospy.loginfo("No mask data available")
                return np.array([])

            # Get dimensions
            if not hasattr(masks_msg, 'layout') or not hasattr(masks_msg.layout, 'dim'):
                rospy.logwarn("No layout dimensions in mask message")
                return np.array([])

            dims = masks_msg.layout.dim
            rospy.loginfo(f"Mask dimensions count: {len(dims)}")

            if len(dims) != 3:
                rospy.logwarn(f"Invalid mask dimensions: expected 3, got {len(dims)}")
                return np.array([])

            classes = dims[0].size
            height = dims[1].size
            width = dims[2].size

            rospy.loginfo(f"Mask dimensions: classes={classes}, height={height}, width={width}")

            if classes == 0 or height == 0 or width == 0:
                rospy.loginfo("Zero-sized mask dimensions")
                return np.array([])

            expected_size = classes * height * width
            actual_size = len(masks_msg.data)
            rospy.loginfo(f"Expected mask data size: {expected_size}, actual: {actual_size}")

            if actual_size != expected_size:
                rospy.logwarn(f"Mask data size mismatch: expected {expected_size}, got {actual_size}")
                return np.array([])

            # Reshape data and convert back to boolean
            rospy.loginfo("Converting mask data to numpy array")
            rospy.loginfo(f"Mask data type: {type(masks_msg.data)}")
            rospy.loginfo(f"First few bytes: {masks_msg.data[:10] if len(masks_msg.data) > 10 else masks_msg.data}")

            # Handle different data formats
            if isinstance(masks_msg.data, (bytes, bytearray)):
                # Data is raw bytes - convert directly
                masks_flat = np.frombuffer(masks_msg.data, dtype=np.uint8)
            else:
                # Data is a list or array - convert normally
                masks_flat = np.array(masks_msg.data, dtype=np.uint8)

            rospy.loginfo(f"Converted to numpy array with shape: {masks_flat.shape}")
            rospy.loginfo("Reshaping mask data")
            per_class_mask = masks_flat.reshape((classes, height, width)).astype(bool)
            rospy.loginfo(f"Successfully reconstructed masks with shape: {per_class_mask.shape}")

            return per_class_mask

        except Exception as e:
            import traceback
            rospy.logerr(f"Error reconstructing masks: {e}")
            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            return np.array([])

    def process_messages(self, image: PIL.Image.Image, terrain_msg, depth_msg, pc_msg, masks_msg, header):
        try:
            rospy.loginfo("Step 1: Converting terrain segmentation")
            # Convert terrain segmentation
            terrain_seg = self.bridge.imgmsg_to_cv2(terrain_msg, "mono16")
            rospy.loginfo(f"Terrain seg shape: {terrain_seg.shape}, dtype: {terrain_seg.dtype}")

            rospy.loginfo("Step 2: Converting depth image")
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            rospy.loginfo(f"Depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")

            rospy.loginfo("Step 3: Converting point cloud")
            # Convert point cloud
            pcd = self.ros_pointcloud2_to_o3d(pc_msg)

            rospy.loginfo("Step 4: Reconstructing masks")
            # Reconstruct masks
            per_class_mask = self.reconstruct_masks(masks_msg)
            rospy.loginfo(f"Per class mask shape: {per_class_mask.shape if per_class_mask.size > 0 else 'empty'}")

            rospy.loginfo("Step 5: Processing logic")
            # Process using the original logic
            terrain_pred_seg, distance_to_arrays, in_the_way_mask = self.process(
                image, terrain_seg, pcd, per_class_mask
            )

            rospy.loginfo("Step 6: Generating final result")
            # Generate final result
            result = self.predict_new_logic(image, terrain_pred_seg, distance_to_arrays, in_the_way_mask)

            rospy.loginfo("Step 7: Dumping images")
            # Dump images for debugging
            self.dump_images(image, terrain_seg, depth_image, per_class_mask, result, terrain_pred_seg, in_the_way_mask)

            rospy.loginfo("Step 8: Publishing results")
            # Publish results
            self.publish_results(result, terrain_pred_seg, distance_to_arrays, in_the_way_mask, header)

        except Exception as e:
            import traceback
            rospy.logerr(f"Error processing messages: {e}")
            rospy.logerr(f"Traceback: {traceback.format_exc()}")

    def process(self, image: PIL.Image.Image, terrain_seg, pcd, per_class_mask):
        """Process the inputs using the original FastModels logic"""
        terrain_pred_seg = terrain_seg.copy()
        terrain_pred_seg[terrain_pred_seg == 0] = 20
        terrain_pred_seg[(terrain_pred_seg == 2) | (terrain_pred_seg == 6) | (terrain_pred_seg == 9)] = 0
        terrain_pred_seg[terrain_pred_seg != 0] = 1
        terrain_pred_seg_bool = terrain_pred_seg.astype(bool)
        inv_terrain_pred_seg_bool = ~terrain_pred_seg_bool

        all_pc_coords = np.asarray(pcd.points).reshape((-1, 3))
        x, y = np.meshgrid(np.arange(self.lidar_cam_calib.img_width), np.arange(self.lidar_cam_calib.img_height))
        all_pixel_locs = np.stack((x.flatten(), y.flatten()), axis=-1)
        terrain_pixel_locs = all_pixel_locs[inv_terrain_pred_seg_bool.flatten()]
        terrain_pc_coords = all_pc_coords[inv_terrain_pred_seg_bool.flatten()]

        distance_to_arrays = {}
        if per_class_mask.size > 0:
            for cidx in range(per_class_mask.shape[0]):
                dist_arr = np.ones((image.height, image.width)) * (-1.0)
                class_mask = per_class_mask[cidx].squeeze()
                all_mask_values = class_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
                class_pixel_locs = all_pixel_locs[all_mask_values]
                class_pc_coords = all_pc_coords[all_mask_values]
                if class_pc_coords is None or class_pc_coords.shape[0] == 0:
                    dist_arr[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = np.inf
                else:
                    kdtree = KDTree(class_pc_coords)
                    distances, _ = kdtree.query(terrain_pc_coords)
                    dist_arr[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = distances
                distance_to_arrays[cidx] = dist_arr
        else:
            # No detections - fill with inf
            for cidx in range(len(self.gsam_object_dict)):
                dist_arr = np.ones((image.height, image.width)) * np.inf
                distance_to_arrays[cidx] = dist_arr

        in_the_way_mask = np.ones((image.height, image.width), dtype=np.uint8) * (-1)
        tpred_seg = terrain_seg.copy()
        non_traversable_seg = ((tpred_seg == 0) | (tpred_seg == 1) | (tpred_seg == 3) | (tpred_seg == 13)).astype(bool)
        linearized_non_traversable_seg = non_traversable_seg.flatten()
        non_traversable_pc_coords = all_pc_coords[linearized_non_traversable_seg]

        if non_traversable_pc_coords.shape[0] > 0 and terrain_pc_coords.shape[0] > 0:
            tree_nontraversable = KDTree(non_traversable_pc_coords)
            distances, _ = tree_nontraversable.query(terrain_pc_coords)
            too_far_mask = (distances > 2.5).astype(np.uint8)
            in_the_way_mask[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = too_far_mask
        else:
            if terrain_pixel_locs.shape[0] > 0:
                in_the_way_mask[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = 0

        return terrain_pred_seg, distance_to_arrays, in_the_way_mask

    def _terrain(self, terrain_pred_seg):
        return terrain_pred_seg

    def _in_the_way(self, in_the_way_mask):
        return in_the_way_mask

    def _distance_to(self, class_name: str, distance_to_arrays):
        return distance_to_arrays[self.gsam_object_dict[class_name]]

    def _frontal_distance(self, class_name: str, distance_to_arrays):
        return self._distance_to(class_name, distance_to_arrays)

    def predict_new_logic(self, image: PIL.Image.Image, terrain_pred_seg, distance_to_arrays, in_the_way_mask):
        """Apply the original prediction logic"""
        if self.do_car:
            return ((self._terrain(terrain_pred_seg) == 0) &
                    (self._distance_to("person", distance_to_arrays) > 2.0) &
                    (self._distance_to("pole", distance_to_arrays) > 0.0) &
                    (self._in_the_way(in_the_way_mask) == 0) &
                    (self._frontal_distance("entrance", distance_to_arrays) > 3.5) &
                    (self._distance_to("bush", distance_to_arrays) > 0.0) &
                    (self._frontal_distance("staircase", distance_to_arrays) > 3.5) &
                    (self._distance_to("car", distance_to_arrays) > 8.0)
                    ).astype(np.uint8)
        return ((self._terrain(terrain_pred_seg) == 0) &
                (self._distance_to("person", distance_to_arrays) > 2.0) &
                (self._distance_to("pole", distance_to_arrays) > 0.0) &
                (self._in_the_way(in_the_way_mask) == 0)
                ).astype(np.uint8)

    def publish_results(self, result, terrain_pred_seg, distance_to_arrays, in_the_way_mask, header):
        try:
            # Publish main result (the final prediction mask - this is the key output!)
            result_msg = self.bridge.cv2_to_imgmsg((result * 255).astype(np.uint8), "mono8")
            result_msg.header = header
            self.result_pub.publish(result_msg)

            # Also publish on dedicated prediction mask topic for clarity
            prediction_mask_msg = self.bridge.cv2_to_imgmsg(result.astype(np.uint8), "mono8")
            prediction_mask_msg.header = header
            self.prediction_mask_pub.publish(prediction_mask_msg)

            # Publish terrain mask
            terrain_msg = self.bridge.cv2_to_imgmsg(terrain_pred_seg.astype(np.uint8), "mono8")
            terrain_msg.header = header
            self.terrain_mask_pub.publish(terrain_msg)

            # Publish distance arrays as JSON
            distance_data = {
                "distance_arrays": {str(k): v.tolist() for k, v in distance_to_arrays.items()},
                "object_dict": self.gsam_object_dict,
                "timestamp": header.stamp.to_sec()
            }
            distance_msg = String()
            distance_msg.data = json.dumps(distance_data)
            self.distance_arrays_pub.publish(distance_msg)

            # Publish in_the_way mask
            in_the_way_msg = self.bridge.cv2_to_imgmsg(in_the_way_mask.astype(np.uint8), "mono8")
            in_the_way_msg.header = header
            self.in_the_way_pub.publish(in_the_way_msg)

        except Exception as e:
            rospy.logerr(f"Error publishing results: {e}")

    def dump_images(self, original_image, terrain_seg, depth_image, per_class_mask, final_pred_mask, terrain_pred_seg, in_the_way_mask):
        """Dump intermediate and final results as images for debugging"""
        try:
            # 1. Original image
            original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.dump_dir, "01_original_image.jpg"), original_cv)

            # 2. Terrain segmentation (raw from terrain node)
            terrain_vis = (terrain_seg * 255 / terrain_seg.max()).astype(np.uint8) if terrain_seg.max() > 0 else terrain_seg.astype(np.uint8)
            cv2.imwrite(os.path.join(self.dump_dir, "02_terrain_segmentation.png"), terrain_vis)

            # 3. Depth image (normalized for visualization)
            depth_vis = (depth_image / depth_image.max() * 255).astype(np.uint8) if depth_image.max() > 0 else np.zeros_like(depth_image, dtype=np.uint8)
            cv2.imwrite(os.path.join(self.dump_dir, "03_depth_image.png"), depth_vis)

            # 4. GSAM masks (combined visualization)
            if per_class_mask.size > 0:
                # Create a colored mask showing all detected objects
                combined_mask = np.zeros((per_class_mask.shape[1], per_class_mask.shape[2], 3), dtype=np.uint8)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

                for cidx in range(per_class_mask.shape[0]):
                    mask = per_class_mask[cidx].squeeze() > 0.5
                    color = colors[cidx % len(colors)]
                    combined_mask[mask] = color

                cv2.imwrite(os.path.join(self.dump_dir, "04_gsam_detections.png"), combined_mask)
            else:
                # No detections
                empty_mask = np.zeros((original_image.height, original_image.width, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(self.dump_dir, "04_gsam_detections.png"), empty_mask)

            # 5. Processed terrain mask
            terrain_processed_vis = (terrain_pred_seg * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.dump_dir, "05_terrain_processed.png"), terrain_processed_vis)

            # 6. In-the-way mask
            in_the_way_vis = np.zeros_like(in_the_way_mask, dtype=np.uint8)
            in_the_way_vis[in_the_way_mask == 0] = 255  # Safe areas in white
            in_the_way_vis[in_the_way_mask == 1] = 128  # Blocked areas in gray
            cv2.imwrite(os.path.join(self.dump_dir, "06_in_the_way_mask.png"), in_the_way_vis)

            # 7. Final prediction mask (THE MAIN OUTPUT)
            final_vis = (final_pred_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.dump_dir, "07_FINAL_PREDICTION_MASK.png"), final_vis)

            # 8. Overlay final prediction on original image
            overlay = original_cv.copy()
            green_mask = np.zeros_like(overlay)
            green_mask[final_pred_mask == 1] = [0, 255, 0]  # Green for traversable areas
            overlay = cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0)
            cv2.imwrite(os.path.join(self.dump_dir, "08_final_overlay.jpg"), overlay)

        except Exception as e:
            rospy.logwarn(f"Error dumping images: {e}")


if __name__ == '__main__':
    try:
        node = ProcessorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

# ROS Nodes for FastModels

This directory contains ROS nodes that break apart the original `FastModels` class into separate, modular components.

## Nodes

### 1. `terrain_node.py`
- **Purpose**: Runs TerrainSegFormer model for terrain segmentation
- **Subscribes to**: Image topic (configurable)
- **Publishes**:
  - `/terrain/pred_img` - Predicted terrain image
  - `/terrain/pred_seg` - Terrain segmentation mask

### 2. `depth_node.py`
- **Purpose**: Runs DepthAny2 model for metric depth estimation
- **Subscribes to**: Image topic (configurable)
- **Publishes**:
  - `/depth/depth_image` - Depth image
  - `/depth/pointcloud` - Generated point cloud

### 3. `gsam_node.py`
- **Purpose**: Runs GSAM model for object detection and segmentation
- **Subscribes to**: Image topic (configurable)
- **Publishes**:
  - `/gsam/annotated_image` - Image with detection annotations
  - `/gsam/detections` - Detection results as JSON
  - `/gsam/masks` - Per-class segmentation masks

### 4. `processor_node.py`
- **Purpose**: Performs downstream processing using outputs from all three models
- **Subscribes to**: All outputs from the three model nodes + original image
- **Publishes**:
  - `/processor/result` - Final traversability result (scaled 0-255)
  - `/processor/prediction_mask` - **MAIN OUTPUT**: Final prediction mask (0/1 values)
  - `/processor/terrain_mask` - Processed terrain mask
  - `/processor/distance_arrays` - Distance arrays as JSON
  - `/processor/in_the_way` - In-the-way mask

## Usage

### Running with Live Camera Feed
```bash
# Launch all nodes with live camera feed
roslaunch src/ros/launch_all_nodes.launch image_topic:=/camera/rgb/image_raw/compressed

# Or with uncompressed images
roslaunch src/ros/launch_all_nodes.launch image_topic:=/camera/rgb/image_raw
```

### Running in Test Mode (30 Hz with test image)
```bash
# Launch all nodes in test mode using test/000000.png
roslaunch src/ros/launch_all_nodes.launch use_test_mode:=true

# Or run individual nodes in test mode
rosrun synapse_dev terrain_node.py _image_topic:=""
rosrun synapse_dev depth_node.py _image_topic:=""
rosrun synapse_dev gsam_node.py _image_topic:=""
rosrun synapse_dev processor_node.py _image_topic:=""
```

### Running Individual Nodes
```bash
# Run terrain node only
rosrun synapse_dev terrain_node.py

# Run depth node only
rosrun synapse_dev depth_node.py

# Run GSAM node only
rosrun synapse_dev gsam_node.py

# Run processor node only (requires other nodes to be running)
rosrun synapse_dev processor_node.py
```

## Parameters

- `image_topic`: Input image topic (default: `/camera/rgb/image_raw/compressed`)
- `test_image_path`: Path to test image for test mode (default: `test/000000.png`)
- `publish_rate`: Publishing rate in Hz for test mode (default: 30.0)
- `do_car`: Enable car detection logic in processor (default: true)

## Topics

### Input Topics
- Image topic (configurable, default: `/camera/rgb/image_raw/compressed`)

### Output Topics
- `/terrain/pred_img` - Terrain prediction image
- `/terrain/pred_seg` - Terrain segmentation
- `/depth/depth_image` - Depth estimation
- `/depth/pointcloud` - Point cloud
- `/gsam/annotated_image` - Annotated detection image
- `/gsam/detections` - Detection results
- `/gsam/masks` - Segmentation masks
- `/processor/result` - Final traversability result (scaled 0-255)
- **`/processor/prediction_mask`** - **MAIN OUTPUT**: Final prediction mask (0/1 values)
- `/processor/terrain_mask` - Processed terrain mask
- `/processor/distance_arrays` - Distance calculations
- `/processor/in_the_way` - Obstacle mask

## Notes

- All nodes support both compressed and uncompressed image topics
- Test mode runs at 30 Hz using a static test image when `image_topic` is set to `None` or empty
- The processor node uses message synchronization to align inputs from all three model nodes
- GPU acceleration is automatically used when available
- The original `FastModels` logic is preserved in the processor node
- **The `/processor/prediction_mask` topic contains the final boolean traversability mask (0/1 values) that corresponds to the first return value of the original `FastModels.predict_new()` method**
- **The processor node automatically saves debug images to `src/ros/` directory, overwriting on each update to prevent storage bloat**

## Debug Images

The processor node automatically saves the following debug images to `src/ros/`:

1. `01_original_image.jpg` - Input image
2. `02_terrain_segmentation.png` - Raw terrain segmentation from terrain node
3. `03_depth_image.png` - Depth estimation from depth node
4. `04_gsam_detections.png` - Object detections from GSAM node (colored by class)
5. `05_terrain_processed.png` - Processed terrain mask
6. `06_in_the_way_mask.png` - Obstacle avoidance mask
7. `07_FINAL_PREDICTION_MASK.png` - **Final traversability prediction mask**
8. `08_final_overlay.jpg` - Final prediction overlaid on original image (green = traversable) 
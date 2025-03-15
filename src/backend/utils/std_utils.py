import json
import numpy as np
from PIL import Image
import os
from scipy import ndimage
from scipy.ndimage import uniform_filter
import rosbag
from cv_bridge import CvBridge
import cv2
from tqdm import tqdm


def images_to_video(cv2_images: list, video_path: str, fps=10):
    height, width, _ = cv2_images[0].shape  # Get dimensions from first frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), True)
    for img in tqdm(cv2_images, desc="Writing Video"):
        out.write(img)
    out.release()


def image_cropper(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define the coordinates for cropping
    left = 395
    top = 394
    right = 690
    bottom = 690

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Save the cropped image
    cropped_image.save(image_path)


def reader(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    return content


def writer(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)


def append_writer(filepath, content):
    with open(filepath, 'a') as f:
        f.write(content)


def json_reader(filepath):
    with open(filepath, 'r') as f:
        dict_content = json.load(f)
    return dict_content


def json_writer(filepath, dict_content):
    with open(filepath, 'w') as f:
        json.dump(dict_content, f, indent=4)


def smoothing_filter(segmentation, window_size=11):
    def calculate_mode(values):
        counts = np.bincount(values.astype(int))
        return np.argmax(counts)
    filtered_seg = ndimage.generic_filter(segmentation, function=calculate_mode, size=window_size, mode='nearest')
    return np.asarray(filtered_seg).astype(np.asarray(segmentation).dtype).reshape(segmentation.shape)


def compute_confidence_array(binary_array, window_size=21):
    conf_score = uniform_filter(binary_array.astype(float), size=window_size, mode='constant', cval=0.0)
    return conf_score


def read_compr_images_from_bag(bag_file, topic):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file)
    images = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        cv2_frame_np = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        images.append(cv2_frame_np)
    bag.close()
    return images


if __name__ == "__main__":
    # random 0 1 segmentation
    seg = np.random.randint(0, 2, (100, 100))
    seg[0:60, 0:60] = 1
    seg[0:40, 0:40] = 0
    smoothed_seg = smoothing_filter(seg)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(seg, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(compute_confidence_array(smoothing_filter(seg)), cmap='gray')
    plt.title("Smoothed")
    plt.show()

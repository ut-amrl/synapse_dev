import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
from datasets import load_dataset
import numpy as np
from terrainseg.inference import TerrainSegFormer
from utilities.std_utils import reader, json_reader
import cv2
from PIL import Image
from safety.faster_ns_inference import FasterImageInference, FasterImageInferenceCaP
from safety._visprog_inference import infer_visprog
from llm._vlm import get_vlm_response
from segments import SegmentsClient
import re
from simple_colors import red, green


def bind_method(instance, name):
    def method(arg):
        dynamic_method = getattr(instance, name)
        return dynamic_method(arg)
    return method


def reduce_array(arr, grid_size=20):
    # Dimensions of the new grid
    new_height, new_width = grid_size, grid_size
    # Size of blocks
    block_height, block_width = arr.shape[0] // new_height, arr.shape[1] // new_width
    # Reduced array
    reduced = np.zeros((new_height, new_width), dtype=int)
    for i in range(new_height):
        for j in range(new_width):
            # Extracting each block
            block = arr[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            # Counting 1s and 0s
            ones = np.sum(block)
            # Determining the value for reduced array cell
            reduced[i, j] = 1 if ones > (block_height * block_width) / 100 else 0
    return reduced


def project_to_original(reduced_array, original_height=540, original_width=960):
    block_height, block_width = original_height // reduced_array.shape[0], original_width // reduced_array.shape[1]
    projected = np.zeros((original_height, original_width), dtype=int)
    for i in range(reduced_array.shape[0]):
        for j in range(reduced_array.shape[1]):
            projected[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width] = reduced_array[i, j]
    return projected


def gpt4v_save_pred(sdi, root_dir, method_num, pre_prompt):
    prompt = "You will see the new image now."
    pred_root_dir = os.path.join(root_dir, f"methods_preds/{method_num}")
    os.makedirs(pred_root_dir, exist_ok=True)
    api_key = os.environ.get("SEGMENTSAI_API_KEY")
    client = SegmentsClient(api_key)
    all_samples = client.get_samples(sdi,
                                     labelset="ground-truth",
                                     sort="name",
                                     direction="asc")

    for i, sample in enumerate(all_samples):
        print(f"Processing {i+1}/{len(all_samples)}")
        filename = sample.name
        url = sample.attributes.image.url
        noext_filename = os.path.splitext(filename)[0]
        text = get_vlm_response(pre_prompt, prompt, url)
        try:
            match = re.search(r'```python(.*?)```', text, re.DOTALL)
            code = match.group(1).strip()
            code = f"arr = {code}"
            namespace = {
                "arr": None
            }
            exec(code, namespace)
            arr = np.array(namespace['arr']).reshape((20, 20)).astype(np.uint8)
        except:
            arr = np.zeros((20, 20), dtype=np.uint8)
            print(f"Error in parsing response from gpt4v:")
            print("------------------------------------------------------------- RESPONSE")
            print(text)
            print("-------------------------------------------------------------")
        arr = project_to_original(arr)
        arr[arr == 0] = 2
        flat_arr = arr.reshape(-1).astype(np.uint8)
        flat_arr.tofile(os.path.join(pred_root_dir, f"{noext_filename}.bin"))


def ns_save_pred(hfdi, root_dir, method_num, ldips_infer_ns_obj: FasterImageInference, start=0, step_size=1):
    pred_root_dir = os.path.join(root_dir, f"methods_preds/{method_num}")
    fi_data_dir = os.path.join(root_dir, "fi_data")
    os.makedirs(pred_root_dir, exist_ok=True)
    os.makedirs(fi_data_dir, exist_ok=True)
    ds = load_dataset(hfdi)['train']
    for i in range(start, len(ds), step_size):
        print(f"Processing {i+1}/{len(ds)}")
        img_name = ds[i]['name']
        noext_name = os.path.splitext(img_name)[0]
        pc_name = f"{noext_name}.bin"
        img_path = os.path.join(root_dir, "images", img_name)
        pc_path = os.path.join(root_dir, "pcs", pc_name)
        cv2_img = cv2.imread(img_path)
        pc_xyz = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        ldips_infer_ns_obj.set_state(fi_data_dir=fi_data_dir,
                                     noext_name=noext_name,
                                     img_bgr=cv2_img,
                                     pc_xyz=pc_xyz)
        is_safe_mask = np.zeros((cv2_img.shape[0], cv2_img.shape[1]), dtype=np.uint8)
        for k in range(cv2_img.shape[0]):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing row: {k}/{cv2_img.shape[0]}")
            for j in range(cv2_img.shape[1]):
                print(f"Processing column: {j}/{cv2_img.shape[1]}", end='\r')
                try:
                    is_safe_mask[k, j] = is_safe((j, k))
                except:
                    print(red(f"\nError in processing pixel: {j}/{cv2_img.shape[1]}\n", "bold"))
                    is_safe_mask[k, j] = False
            print("\033[F\033[K", end="")  # Move up and clear the line

        is_safe_mask[is_safe_mask == 0] = 2
        flat_is_safe_mask = is_safe_mask.reshape(-1).astype(np.uint8)
        flat_is_safe_mask.tofile(os.path.join(pred_root_dir, f"{noext_name}.bin"))


def nn_save_pred(hfdi, hfmi, root_dir, method_num):
    """
    Given the huggingface dataset identifier and model identifier, runs inference on the dataset and saves the predictions as bins
    """
    pred_root_dir = os.path.join(root_dir, f"methods_preds/{method_num}")
    os.makedirs(pred_root_dir, exist_ok=True)
    s = TerrainSegFormer(hf_dataset_name=hfdi,
                         hf_model_name=hfmi,
                         hf_model_ver=None)
    s.load_model_inference()
    s.prepare_dataset()
    eval_ds = s.ds
    for i in range(len(eval_ds)):
        print(f"Processing {i+1}/{len(eval_ds)}")
        pil_img = eval_ds[i]['pixel_values']
        W, H = pil_img.size
        img_name = eval_ds[i]['name']
        noext_img_name = os.path.splitext(img_name)[0]
        img_np = np.array(pil_img).reshape((H, W, 3))
        _, nn_seg = s.predict_new(pil_img)
        nn_seg_np = np.array(nn_seg).reshape((H, W))
        flat_nn_seg_np = nn_seg_np.reshape(-1).astype(np.uint8)
        flat_nn_seg_np.tofile(os.path.join(pred_root_dir, f"{noext_img_name}.bin"))


def visprog_save_pred(hfdi, root_dir, method_num, prompted=False):
    pred_root_dir = os.path.join(root_dir, f"methods_preds/{method_num}")
    os.makedirs(pred_root_dir, exist_ok=True)
    s = TerrainSegFormer(hf_dataset_name=hfdi,
                         hf_model_ver=None)
    s.load_model_inference()
    s.prepare_dataset()
    eval_ds = s.ds
    if prompted:
        prompt = "Safe location = sidewalk or concrete, and far away from objects. Replace safe location with snow."
    else:
        prompt = "Replace safe location for a robot to pull over to with snow."
    for i in range(len(eval_ds)):
        print(green(f"Processing {i+1}/{len(eval_ds)}", "bold"))
        pil_img = eval_ds[i]['pixel_values']
        W, H = pil_img.size
        img_name = eval_ds[i]['name']
        noext_img_name = os.path.splitext(img_name)[0]
        try:
            result, prog_state, html_str = infer_visprog(pil_img, prompt)
            unified_mask = np.any([mask_dict['mask'] != 0 for mask_dict in prog_state['OBJ1']], axis=0).astype(np.uint8)
        except:
            print(red(f"Error in visprog: {i+1}/{len(eval_ds)}", "bold"))
            unified_mask = np.zeros((H, W), dtype=np.uint8)
        unified_mask[unified_mask == 0] = 2
        flat_seg_np = unified_mask.reshape(-1).astype(np.uint8)
        flat_seg_np.tofile(os.path.join(pred_root_dir, f"{noext_img_name}.bin"))


def nn_depth_save_pred(hfdi, hfmi, root_dir, method_num):
    """
    Given the huggingface dataset identifier and model identifier, runs inference on the dataset and saves the predictions as bins
    """
    pred_root_dir = os.path.join(root_dir, f"methods_preds/{method_num}")
    os.makedirs(pred_root_dir, exist_ok=True)
    s = TerrainSegFormer(hf_dataset_name=hfdi,
                         hf_model_name=hfmi,
                         hf_model_ver=None)
    s.load_model_inference()
    s.prepare_dataset()
    eval_ds = s.ds
    for i in range(len(eval_ds)):
        print(f"Processing {i+1}/{len(eval_ds)}")
        pil_img = eval_ds[i]['pixel_values']
        W, H = pil_img.size
        img_name = eval_ds[i]['name']
        noext_img_name = os.path.splitext(img_name)[0]
        img_np = np.array(pil_img).reshape((H, W, 3))
        depth_image = Image.open(os.path.join(root_dir, "depth", f"{noext_img_name}.png"))
        _, nn_seg = s.predict_new_with_depth(pil_img, depth_image)
        nn_seg_np = np.array(nn_seg).reshape((H, W))
        flat_nn_seg_np = nn_seg_np.reshape(-1).astype(np.uint8)
        flat_nn_seg_np.tofile(os.path.join(pred_root_dir, f"{noext_img_name}.bin"))


def gt_save(hfdi, root_dir):
    """
    Saves the ground truth labels as bins
    """
    gt_root_dir = os.path.join(root_dir, "gt_preds")
    os.makedirs(gt_root_dir, exist_ok=True)
    ds_dict = load_dataset(hfdi)
    ds = ds_dict['train']
    for i in range(len(ds)):
        print(f"Processing {i+1}/{len(ds)}")
        pil_img = ds[i]['pixel_values']
        W, H = pil_img.size
        label_img = ds[i]['labels']
        img_name = ds[i]['name']
        noext_img_name = os.path.splitext(img_name)[0]
        label_np = np.array(label_img).reshape((H, W))
        flat_label_np = label_np.reshape(-1).astype(np.uint8)
        flat_label_np.tofile(os.path.join(gt_root_dir, f"{noext_img_name}.bin"))


if __name__ == "__main__":
    root_dir = os.path.join(nspl_root_dir, "evals_data_safety/utcustom")

    methods_metadata = json_reader(os.path.join(nspl_root_dir, "scripts/safety/methods_metadata.json"))

    parser = argparse.ArgumentParser()
    NSCL_MODE = "test"
    parser.add_argument("--eval_di", type=str, default="sam1120/safety-utcustom-TEST")
    parser.add_argument("--root_dirname", type=str, default="test")  # wrt to root_dir defined above
    parser.add_argument("--method_num", type=int, default=1)
    parser.add_argument("--ns_sketch_num", type=int, default=29)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step_size", type=int, default=1)
    args = parser.parse_args()

    eval_root_dir = os.path.join(root_dir, args.root_dirname)
    if not os.path.exists(os.path.join(eval_root_dir, "gt_preds")):
        print(f"Saving ground truth labels for {args.root_dirname}...")
        gt_save(hfdi=args.eval_di,
                root_dir=eval_root_dir)

    if methods_metadata[str(args.method_num)]["type"] == "nn":
        if "depth" in methods_metadata[str(args.method_num)]["name"]:
            nn_depth_save_pred(hfdi=args.eval_di,
                               hfmi=methods_metadata[str(args.method_num)]["model-hfmi"],
                               root_dir=eval_root_dir,
                               method_num=args.method_num)
        else:
            nn_save_pred(hfdi=args.eval_di,
                         hfmi=methods_metadata[str(args.method_num)]["model-hfmi"],
                         root_dir=eval_root_dir,
                         method_num=args.method_num)
    elif methods_metadata[str(args.method_num)]["type"] == "ns":
        # ns setup
        hitl_llm_state = json_reader(os.path.join(nspl_root_dir, "scripts/llm/state.json"))
        DOMAIN = hitl_llm_state["domain"]
        if methods_metadata[str(args.method_num)]["name"] == "ns-hitl":
            filled_lfps_sketch = json_reader(os.path.join(nspl_root_dir, "scripts/llm/seqn_filled_lfps_sketches.json"))[str(args.ns_sketch_num)]
            fi = FasterImageInference(DOMAIN, NSCL_MODE)
        # ablations
        elif methods_metadata[str(args.method_num)]["name"] == "ns-direct":
            filled_lfps_sketch = json_reader(os.path.join(nspl_root_dir, "scripts/llm/ablation_direct/seqn_filled_lfps_sketches.json"))["29"]
            fi = FasterImageInference(DOMAIN)
        elif methods_metadata[str(args.method_num)]["name"] == "ns-directplus":
            filled_lfps_sketch = json_reader(os.path.join(nspl_root_dir, "scripts/llm/ablation_directplus/seqn_filled_lfps_sketches.json"))["29"]
            fi = FasterImageInference(DOMAIN)
        elif methods_metadata[str(args.method_num)]["name"] == "ns-code-as-policies":
            filled_lfps_sketch = json_reader(os.path.join(nspl_root_dir, "scripts/llm/ablation_cap/seqn_filled_lfps_sketches.json"))["29"]
            fi = FasterImageInferenceCaP(DOMAIN)
        elif methods_metadata[str(args.method_num)]["name"] == "ns-notraj":
            filled_lfps_sketch = json_reader(os.path.join(nspl_root_dir, "scripts/llm/ablation_notraj/seqn_filled_lfps_sketches.json"))["29"]
            fi = FasterImageInference(DOMAIN)
        terrain = fi._terrain
        in_the_way = fi._in_the_way
        slope = fi._slope
        for method_name in ['distance_to_' + obj for obj in DOMAIN["objects"]]:
            globals()[method_name] = bind_method(fi, f"_{method_name}")
        for method_name in ['frontal_distance_' + obj for obj in DOMAIN["objects"]]:
            globals()[method_name] = bind_method(fi, f"_{method_name}")
        exec(filled_lfps_sketch)
        ns_save_pred(hfdi=args.eval_di,
                     root_dir=eval_root_dir,
                     method_num=args.method_num,
                     ldips_infer_ns_obj=fi,
                     start=args.start,
                     step_size=args.step_size)
    elif methods_metadata[str(args.method_num)]["type"] == "vlm":
        if "gpt4v" in methods_metadata[str(args.method_num)]["name"]:
            pre_prompt = reader(os.path.join(nspl_root_dir, f"scripts/llm/preprompts/{methods_metadata[str(args.method_num)]['preprompt-filename']}"))
            gpt4v_save_pred(sdi=args.eval_di,
                            root_dir=eval_root_dir,
                            method_num=args.method_num,
                            pre_prompt=pre_prompt)
        elif "visprog" in methods_metadata[str(args.method_num)]["name"]:
            visprog_save_pred(hfdi=args.eval_di,
                              root_dir=eval_root_dir,
                              method_num=args.method_num,
                              prompted="prompted" in methods_metadata[str(args.method_num)]["name"])

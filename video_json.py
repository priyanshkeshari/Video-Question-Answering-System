import os
import json
import cv2
from PIL import Image
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from torch import FloatTensor
from models import blip_model, blip_processor, yolo_model
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.tokenization_utils_base import BatchEncoding



model_id = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blip_model: BlipForConditionalGeneration = (
    BlipForConditionalGeneration.from_pretrained(model_id).to(device)  # type: ignore
)


def generate_caption(image_region):
    if isinstance(image_region, np.ndarray):
        image_region = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))

    # Convert to device properly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get processed inputs
    inputs = blip_processor(images=image_region, return_tensors="pt")
    pixel_values: FloatTensor = inputs["pixel_values"]  # type: ignore
    pixel_values = pixel_values.to(device) # type: ignore

    with torch.no_grad():
        output_ids = blip_model.generate(
            pixel_values=pixel_values,  # ✅ No more EncodingFast here
            max_length=100
        )

    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption




def detect_and_caption_objects(image_path, expand_margin_threshold=0.1, object_neglection_threshold=30):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    results = yolo_model(image_path)
    frame_data = []

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        box_width, box_height = x2 - x1, y2 - y1
        if box_width < object_neglection_threshold or box_height < object_neglection_threshold:
            continue

        margin_x = int(expand_margin_threshold * box_width)
        margin_y = int(expand_margin_threshold * box_height)
        x1_exp = max(0, x1 - margin_x)
        y1_exp = max(0, y1 - margin_y)
        x2_exp = min(img_width, x2 + margin_x)
        y2_exp = min(img_height, y2 + margin_y)

        region = image.crop((x1_exp, y1_exp, x2_exp, y2_exp))
        caption = generate_caption(region)

        frame_data.append({
            "caption": caption,
            "center": [(x1 + x2) // 2, (y1 + y2) // 2]
        })

    return frame_data

def mark_motion_status(all_frames, fps=1, threshold=2.0):
    object_tracks = defaultdict(list)

    for frame_idx, frame in enumerate(all_frames):
        for obj_idx, obj in enumerate(frame):
            obj_id = f"{frame_idx}_{obj_idx}"  # Changed: removed label
            object_tracks[obj_id].append((frame_idx, obj["center"]))

    motion_status = defaultdict(lambda: "steady")

    for obj_id, track in object_tracks.items():
        if len(track) < 2:
            continue
        total_movement = 0.0
        for i in range(1, len(track)):
            _, (x1, y1) = track[i-1]
            _, (x2, y2) = track[i]
            dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
            total_movement += dist
        if total_movement > threshold:
            motion_status[obj_id] = "moving"

    for frame_idx, frame in enumerate(all_frames):
        for obj_idx, obj in enumerate(frame):
            obj_id = f"{frame_idx}_{obj_idx}"  # Consistent ID
            obj["caption"] += f" ({motion_status[obj_id]})"
            del obj["center"]  # remove center after motion detection

    return all_frames

def process_frames_and_caption(folder_path, fps=1):
    frame_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith('.jpg')
    ])

    all_frames_data = []
    for frame_path in tqdm(frame_files, desc="Processing Frames"):
        frame_data = detect_and_caption_objects(frame_path)
        all_frames_data.append(frame_data)

    all_frames_data = mark_motion_status(all_frames_data, fps=fps)
    return all_frames_data

def build_temporal_object_dict(result):
    temporal_dict = {}
    total_frames = len(result)

    for i in range(total_frames):
        key = f"time_{i}"
        temporal_dict[key] = result[max(0, i-2):min(total_frames, i+3)]


    return temporal_dict

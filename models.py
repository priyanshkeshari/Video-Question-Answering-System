from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os

# --- YOLO model ---
yolo_model = YOLO("yolov8s.pt")

# --- BLIP model ---
captioning_model_id = "Salesforce/blip-image-captioning-base"

# Offload folder
OFFLOAD_DIR = "offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Processor
blip_processor = BlipProcessor.from_pretrained(captioning_model_id, use_fast=True)

try:
    # Try auto device placement with offloading
    blip_model = BlipForConditionalGeneration.from_pretrained(
        captioning_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        offload_folder=OFFLOAD_DIR
    )
except ValueError as e:
    print(f"[WARNING] Auto device_map failed: {e}")
    print(f"[INFO] Loading fully on {device}")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        captioning_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    # ✅ Move model to device
    blip_model = blip_model.device  # This is safe & correct

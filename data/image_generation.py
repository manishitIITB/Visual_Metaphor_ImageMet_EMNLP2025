"""
Stable Diffusion 3.5 Image Generation Script
--------------------------------------------
This script:
1. Loads a pickled list of elaborations.
2. Generates images from the elaborations using Stable Diffusion 3.5.
3. Saves the generated images as a pickled list of PIL images. 

Default: single-image generation per prompt per iteration.
"""


import os
import gc
import pickle
from tqdm import tqdm

import torch
from diffusers import StableDiffusion3Pipeline


# ================================
# Configuration
# ================================
MODEL_NAME = "stabilityai/stable-diffusion-3.5-large"
INPUT_FILE = "elaborations.pkl"
OUTPUT_FILE = "images.pkl"
DEVICE = "cuda:0"

NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALES = [3]   # Can add multiple scales if needed

# ================================
# Load Model
# ================================
print(f"Loading model: {MODEL_NAME}")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(DEVICE)


# ================================
# Load Elaborations
# ================================

# Input file is a pickled list of elaborations.

with open(INPUT_FILE, "rb") as f:
    v_elabs = pickle.load(f)

print(f"# of Elaborations (raw): {len(v_elabs)}")


# ================================
# Image Generation
# ================================

list_of_images = []
for ve, _ in tqdm(v_elabs, desc="Generating images"):
    temp_images = []
    for scale in GUIDANCE_SCALES:
        image = pipe(
            ve,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=scale,
        ).images[0]
        temp_images.append(image)
    list_of_images.append(temp_images)

print(f"# of Image generations: {len(list_of_images)}")


# ============================================================
# Note that this implementation generates a single image per iteration. But batched generation is also possible.
# 
# Alternate (Batched) Version (Pseudo-code):
# 1. Set a batch size (e.g., 8 or 16, depending on GPU memory).
# 2. Collect all elaboration prompts into a list.
# 3. To generate images in batches:
#       a. Split the list of elaboration prompts into chunks of size `BATCH_SIZE`.
#       b. Pass each chunk to the diffusion pipeline in one call.
#       c. Collect the list of generated images returned.
#       d. Assign each image back to its corresponding prompt.
# 4. After looping through all batches,
#    you will have a list of image lists (one per prompt).
# This approach is more efficient than single-image generation
# but requires careful handling of batching and GPU memory.
# ============================================================


# ================================
# Save Outputs
# ================================
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(list_of_images, f)

print(f"Saved generated images to {OUTPUT_FILE}")


# ================================
# Cleanup
# ================================
del pipe
gc.collect()
torch.cuda.set_device(int(DEVICE.split(":")[-1]))
torch.cuda.empty_cache()

print("Cleanup complete. GPU memory released.")
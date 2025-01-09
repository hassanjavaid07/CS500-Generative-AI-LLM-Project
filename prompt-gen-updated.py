import os
import json
import time
from PIL import Image
import torch
import clip
from torch.utils.data import DataLoader
from torchvision import transforms


device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define materials and colors
materials = ["leather", "fabric", "rubber", "synthetic"]
colors = ["black", "white", "brown", "red"]

# Cache material and color embeddings
def cache_embeddings():
    material_texts = clip.tokenize(materials).to(device)
    color_texts = clip.tokenize(colors).to(device)
    with torch.no_grad():
        material_embeddings = model.encode_text(material_texts)
        color_embeddings = model.encode_text(color_texts)
    material_embeddings = material_embeddings / material_embeddings.norm(dim=-1, keepdim=True)
    color_embeddings = color_embeddings / color_embeddings.norm(dim=-1, keepdim=True)
    return material_embeddings, color_embeddings

material_embeddings, color_embeddings = cache_embeddings()

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def process_batch(image_paths, shoe_type_batch, base_dir):
    image_inputs = torch.cat([preprocess_image(img_path) for img_path in image_paths], dim=0)
    with torch.no_grad():
        image_embeddings = model.encode_image(image_inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    
    results = []
    for i, image_embedding in enumerate(image_embeddings):
        # Compute similarity for materials
        material_scores = (image_embedding @ material_embeddings.T).squeeze(0)
        best_material_idx = material_scores.argmax()
        best_material = materials[best_material_idx]
        material_confidence = material_scores[best_material_idx].item()

        # Compute similarity for colors
        color_scores = (image_embedding @ color_embeddings.T).squeeze(0)
        best_color_idx = color_scores.argmax()
        best_color = colors[best_color_idx]
        color_confidence = color_scores[best_color_idx].item()

        # Final prompt logic
        final_material = best_material if material_confidence > 0.2 else "an unspecified"
        final_color = best_color if color_confidence > 0.2 else "an unspecified"
        shoe_type = shoe_type_batch[i]
        prompt = f"A photo of {shoe_type} with {final_color} color and {final_material} material"

        # Extract and normalize relative path
        relative_path = os.path.relpath(image_paths[i], base_dir).replace(os.sep, "/")
        
        results.append({
            "image_path": relative_path,
            "new_prompt": prompt,
            "shoe_type": shoe_type
        })
    return results

# Read image paths and shoe types from JSON
def get_paths_json(json_file, base_dir):
    with open(json_file, "r") as file:
        data = json.load(file)
    image_metadata = [(os.path.join(base_dir, entry["image_path"]), entry["shoe_type"]) for entry in data]
    return image_metadata

# Main processing function
def process_dataset(json_file, base_dir, output_json, batch_size=8):
    image_metadata = get_paths_json(json_file, base_dir)
    all_results = []
    start_time = time.time()

    for i in range(0, len(image_metadata), batch_size):
        batch = image_metadata[i:i + batch_size]
        batch_paths, batch_shoe_types = zip(*batch)
        # print(batch)
        results = process_batch(batch_paths, batch_shoe_types, base_dir)
        all_results.extend(results)

        elapsed_time = time.time() - start_time
        print(f"Processed {i + len(batch)} / {len(image_metadata)} images. Elapsed time: {elapsed_time:.2f}s")

    # Save results to JSON
    with open(output_json, "w") as json_file:
        json.dump(all_results, json_file, indent=4)

    print(f"Processing completed. Results saved to {output_json}")

# Set paths and run
json_file_path = r".\orig_prompts_data.json"
dataset_path = r".\ut-zap50k\images" 
output_json_path = r".\compositional_ut_zappos_new_prompts_cpu.json" 

process_dataset(json_file_path, dataset_path, output_json_path, batch_size=16)

import json
from pathlib import Path

roots = [Path('/ssdwork/wangpeng/datasets/dl3dv/DL3DV_RE10K')]
merged_index = {}
data_stages = ("train",)  # for dl3dv

for data_stage in data_stages:
    for root in roots:
        # Load the root's index.
        with (root / data_stage / "index.json").open("r") as f:
            index = json.load(f)
        index = {k: Path(root / data_stage / v) for k, v in index.items()}

        # The constituent datasets should have unique keys.
        assert not (set(merged_index.keys()) & set(index.keys()))

        # Merge the root's index into the main index.
        merged_index = {**merged_index, **index}

# 73319 scenes for re10k, 10136 scenes for dl3dv
scenes = list(merged_index.keys())  # Convert to list for indexing/shuffling

# New code to pair scenes with .jpg images
image_folder = Path('/ssdwork/wangpeng/datasets/wikiart/images_combine/train')  # Specify your folder containing .jpg files
output_json = Path('/ssdwork/wangpeng/datasets/wikiart/images_combine/train/scene_style_mapping_all.json')    # Output JSON file path

# Get all .jpg files in the folder, 86399 images in total
jpg_files = [f.name for f in image_folder.glob('*.jpg') if f.is_file()]
# Alternatively, if you want full paths: jpg_files = [str(f) for f in image_folder.glob('*.jpg')]

# Handle mismatched counts
if len(scenes) != len(jpg_files):
    if len(jpg_files) < len(scenes):
        print(f"Warning: Only {len(jpg_files)} images for {len(scenes)} scenes. Truncating scenes.")
        scenes = scenes[:len(jpg_files)]
    else:
        print(f"Warning: Only {len(scenes)} scenes for {len(jpg_files)} images. Truncating images.")
        # jpg_files = jpg_files[:len(scenes)]
        jpg_files = jpg_files[-len(scenes):]

# Create the one-to-one mapping
scene_image_mapping = dict(zip(scenes, jpg_files))

# read in original re10k mapping
with open('/ssdwork/wangpeng/datasets/wikiart/images_combine/train/scene_style_mapping.json', 'r') as f:
    re10k_scene_image_mapping = json.load(f)

scene_image_mapping = {**re10k_scene_image_mapping, **scene_image_mapping}

# Write to JSON
with output_json.open('w') as f:
    json.dump(scene_image_mapping, f, indent=4)

print(f"Generated JSON file with {len(scene_image_mapping)} scene-image pairs at {output_json}")

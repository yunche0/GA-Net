import os
import json

# JSON file path
json_file_path = "/root/autodl-tmp/PIXIU/data/oxford_flowers/split_zhou_OxfordFlowers.json"
# Directory containing image files
image_dir = "/root/autodl-tmp/PIXIU/data/oxford_flowers/jpg/"


with open(json_file_path, 'r') as f:
    data = json.load(f)

renamed_files = set()

for item in data['train']:
    original_filename = item[0].strip('\\')
    new_filename = f"{item[2]}_{original_filename[-9:]}"

    original_filepath = os.path.join(image_dir, original_filename)
    new_filepath = os.path.join(image_dir, new_filename)

    if os.path.exists(original_filepath):
        os.rename(original_filepath, new_filepath)
        renamed_files.add(new_filepath)
        print(f"Renamed: {original_filepath} -> {new_filepath}")
    else:
        print(f"File not found: {original_filepath}")

for filename in os.listdir(image_dir):
    filepath = os.path.join(image_dir, filename)
    if filepath not in renamed_files and filename.startswith('image_'):
        os.remove(filepath)
        print(f"Deleted: {filepath}")

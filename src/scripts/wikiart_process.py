import os
import shutil
import random

# set path
source_root = "your_path/wikiart/images"  # TODO: orginal path
combine_root = "your_path/wikiart_combine/images_combine"  # TODO: taget path
train_root = os.path.join(combine_root, "train")
test_root = os.path.join(combine_root, "test")

os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

categories = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]

train_count, test_count = 0, 0

for category in categories:
    category_path = os.path.join(source_root, category)
    if category in ["train", "test"]:  # train and test
        continue

    files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

    random.shuffle(files)

    # 90% train and 10% test
    split_idx = int(len(files) * 0.9)
    train_files = files[:split_idx]
    test_files = files[split_idx:]


    for file in train_files:
        new_filename = f"{category}_{file}"  # e.g., "abstract_001.jpg"
        shutil.copy(os.path.join(category_path, file), os.path.join(train_root, new_filename))
        train_count += 1

    for file in test_files:
        new_filename = f"{category}_{file}"  # e.g., "abstract_002.jpg"
        shutil.copy(os.path.join(category_path, file), os.path.join(test_root, new_filename))
        test_count += 1

    print(f"✅ {category}: {len(train_files)} -> train, {len(test_files)} -> test")

print(f"total {train_count} train images, {test_count} test images.")

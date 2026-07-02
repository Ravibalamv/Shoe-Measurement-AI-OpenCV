import os
import random
import shutil

# Dataset folder
dataset_path = "train"

images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# Output folders
output = "dataset"

splits = ["train", "val", "test"]

for split in splits:
    os.makedirs(os.path.join(output, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output, split, "labels"), exist_ok=True)

# Get all image files
images = [f for f in os.listdir(images_path)
          if f.endswith((".jpg", ".jpeg", ".png"))]

random.shuffle(images)

n = len(images)

train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]


def copy_files(file_list, split):
    for img in file_list:

        shutil.copy(
            os.path.join(images_path, img),
            os.path.join(output, split, "images", img)
        )

        label = os.path.splitext(img)[0] + ".txt"

        if os.path.exists(os.path.join(labels_path, label)):
            shutil.copy(
                os.path.join(labels_path, label),
                os.path.join(output, split, "labels", label)
            )


copy_files(train_imgs, "train")
copy_files(val_imgs, "val")
copy_files(test_imgs, "test")

print("Done!")

print("Train:", len(train_imgs))
print("Validation:", len(val_imgs))
print("Test:", len(test_imgs))
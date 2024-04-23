import os
import random
import shutil
from collections import defaultdict

raw_train_folder = 'dataset/TrainIJCNN2013'
raw_test_folder = 'dataset/TestIJCNN2013Download'
raw_gt_file = 'dataset/gt.txt'
train_folder = 'dataset/train'
val_folder = 'dataset/val'
test_folder = 'dataset/test'
train_gt_file = 'dataset/train_gt.txt'
val_gt_file = 'dataset/train_gt.txt'

def split_data(gt_file, train_file, val_file, train_ratio=0.8):

  with open(gt_file, 'r') as gt:
    image_data = defaultdict(list)  # Dictionary to store data for each image
    for line in gt:
      image_id, x1, y1, x2, y2, label = line.strip().split(';')
      image_data[image_id].append(f"{image_id};{x1};{y1};{x2};{y2};{label}\n")

    images = list(image_data.items())  # List of (image_id, data) tuples
    random.shuffle(images)  # Randomly shuffle the image list

    train_size = int(len(images) * train_ratio)
    train_images, val_images = images[:train_size], images[train_size:]

    with open(train_file, 'w') as train, open(val_file, 'w') as val:
      for image_id, data in sorted(train_images, key=lambda x: x[0]):
        train.writelines(data)
      for image_id, data in sorted(val_images, key=lambda x: x[0]):
        val.writelines(data)


def split_train_folder(raw_train_folder, train_gt_file, val_gt_file, train_folder, val_folder):
  train_images, val_images = set(), set()
  with open(train_gt_file, 'r') as train, open(val_gt_file, 'r') as val:
    for line in train:
      image_id = line.strip().split(';')[0]
      train_images.add(image_id)
    for line in val:
      image_id = line.strip().split(';')[0]
      val_images.add(image_id)

  os.makedirs(train_folder)
  os.makedirs(val_folder)

  for filename in os.listdir(raw_train_folder):
    if filename in train_images:
      shutil.copy(os.path.join(raw_train_folder, filename), os.path.join(train_folder, filename))
    elif filename in val_images:
      shutil.copy(os.path.join(raw_train_folder, filename), os.path.join(val_folder, filename))

def create_test_folder(src_test_folder, dst_test_folder):
  shutil.copytree(src_test_folder, dst_test_folder)

def clean_up(paths):
    for path in paths:
      if os.path.isfile(path):
        os.remove(path)
      elif os.path.isdir(path):
        shutil.rmtree(path)


split_data(raw_gt_file, train_gt_file, val_gt_file)
split_train_folder(raw_train_folder, train_gt_file, val_gt_file, train_folder, val_folder)
create_test_folder(raw_test_folder, test_folder)
clean_up([raw_train_folder, raw_test_folder, raw_gt_file])

import os
import cv2
import imgaug.augmenters as iaa

# Define augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(rotate=(-30, 30)),  # random rotations
    iaa.GaussianBlur(sigma=(0, 1.0)),  # gaussian blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),  # gaussian noise
    #iaa.GammaContrast((0.5, 2.0)),  # Apply gamma contrast adjustment
    #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 2.0))  # Sharpen the image
    #iaa.ChangeColorTemperature((2000, 15000))  # Change color temperature
    # You can add more augmentation techniques here
])

# Input and output folders
print('path', os.path)
input_folder = "/Users/abhishekverma/Desktop/GTech/DL_2024Spr/detr-main/datasets/SKU110K 1000-2000/train"
output_folders = ["fliplr", "affine", "gblur", "gnoise"] #, "gamcon", "sharp", "color"]  # Add more output folders as needed

# Ensure output folders exist, create if not
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Iterate over files in input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_paths = [os.path.join(folder, filename) for folder in output_folders]

        # Read image
        image = cv2.imread(input_path)

        # Apply augmentation pipeline
        augmented_images = seq(images=[image] * len(output_folders))

        # Save augmented images to output folders
        for idx, augmented_image in enumerate(augmented_images):
            cv2.imwrite(output_paths[idx], augmented_image)

print("Image augmentation completed.")

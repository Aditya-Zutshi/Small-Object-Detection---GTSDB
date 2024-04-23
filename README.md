# Small-Object-Detection---GTSDB

## Setting Up Dataset for Small-Object-Detection---GTSDB Project

1. **Download the Dataset:**
   - [Download GTSDB Dataset](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html)
   - Download the following files:
     - `gt.txt`
     - `TestIJCNN2013.zip`
     - `TrainIJCNN2013.zip`

2. **Unzip and Organize:**
   - Create a folder named"dataset" folder in your project directory.
   - Unzip the downloaded `TestIJCNN2013.zip` and `TrainIJCNN2013.zip` files in the dataset folder
   - Run prepare_dataset.py

### Project Structure After Setup:

```bash
Small-Object-Detection---GTSDB/
│
├── dataset/
│   ├── train/
│   │   ├── (training images)
│   │   └── ...
│   ├── val/
│   │   ├── (validation images)
│   │   └── ...
│   ├── test/
│   │   ├── (testing images)
│   │   └── ...
│   ├── train_gt.txt/
│   └── val_gt.txt
│ 
├── (other project files and folders)
└── README.md
```




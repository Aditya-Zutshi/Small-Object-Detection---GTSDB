# Small-Object-Detection---GTSDB

Follow these steps to set up the dataset for your project "Small-Object-Detection---GTSDB":

1. **Download the GTSDB dataset**: Get the dataset from the provided link.
2. **Create Project Folder**: Create a folder named "Small-Object-Detection---GTSDB" on your local machine.
3. **Create Dataset Structure**:
   - Inside the project folder, create a folder named "dataset".
   - Place the downloaded training and testing data in separate folders named "train" and "test" under the "dataset" folder.
4. **Add Ground Truth File**:
   - Put the `gt.txt` file, which contains ground truth annotations, directly under the "dataset" folder.

After completing these steps, your project structure should resemble the following:

Small-Object-Detection---GTSDB/
├── dataset/
│   ├── train/
│   │   ├── (training images)
│   │   └── ...
│   ├── test/
│   │   ├── (testing images)
│   │   └── ...
│   └── gt.txt
│
├── (other project files and folders)
└── README.md

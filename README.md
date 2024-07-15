# AirbusShipDetection
This project aims to detect ships in satellite images using a segmentation approach. The project is structured into several stages, including data preprocessing, model training, and evaluation. The project is implemented using PyTorch and the UNet architecture.

## Project Structure
### 'data/'
'processed/' - Directory containing preprocessed data (X.npy and y.npy).

'train/' - Directory containing original training images 

train_ship_segmentations_v2.csv
 
### 'src/'			
	
 1. data.py - Contains functions for data loading, preprocessing, and balancing:
load_images_and_mask
rle_to_pixels
apply_mask
split_data
sample_ships
balance_dataset
Note: Run this script first to generate images_with_ships and images_without_ships lists. I ran this script in a Kaggle kernel due to insufficient RAM locally. Images are resized to (64, 64) due to time and resource constraints.

2.preprocess.py - Run this script second to generate X.npy and y.npy in the data/processed directory.

3.model.py - Implements the UNet model using PyTorch. I chose PyTorch due to prior experience with segmentation tasks and time constraints prevented the use of TensorFlow.

4.train.py - Run this script after obtaining X.npy and y.npy for model training. The script uses a balanced dataset technique and stratified splitting. The trained model is saved as model.pth in the src directory. The model is trained on a balanced dataset and evaluated on the original dataset.

5.inference.py - Contains the evaluation loop to compute accuracy, F1 macro score, Dice coefficient, and test loss.

 ### 'utils/'

1.dataset.py - Contains the CustomDataset class.

2.transforms.py - Contains transforms to tensor.


Exploratory_Data_Analysis.ipynb - Jupyter notebook for initial data exploration.

requirements.txt - File containing required Python packages.

## Running the Project
### Step 1: Preprocess Data
Run the data.py script to load and preprocess the data(Specify the path to directories with data - img_dir and csv_file). This will generate the images_with_ships and images_without_ships lists. Execute this step in a Kaggle kernel if you encounter memory issues locally.

### Step 2: Generate Processed Data
Run the preprocess.py script to create the X.npy and y.npy files in the data/processed directory. ( You have to specify the directory in save_dir (data/processed)

### Step 3: Train the Model
Run the train.py script to train the model(Specify the path for X.npy and y.npy. They have to be in data/processed). This script will:

Load the preprocessed data.
Split the data into training, validation, and test sets using stratified sampling.
Balance the training dataset.
Train the UNet model on the balanced dataset.
Save the trained model as model.pth.

### Step 4: Evaluate the Model
Run the inference.py script to evaluate the model. This script will compute the accuracy, F1 macro score, Dice coefficient, and test loss.

## Future Work
Due to time and electricity constraints, further experiments and optimizations are needed for better training and parameter tuning. Future improvements include:

Trying different techniques for balancing the dataset.
Implementing Dice loss.
Testing different models.
Applying various data augmentation techniques (e.g., flips, affine transformations).
Adjusting batch sizes.
Training on full-sized images.
Requirements
Refer to requirements.txt for the list of required Python packages.

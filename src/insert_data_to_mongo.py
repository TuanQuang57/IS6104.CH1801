import os
from bson.binary import Binary
from torch.utils.data import DataLoader
from EmbeddingSpace import EmbeddingSpace
from mongo_connection import db


# Define dataset paths dictionary
dataset_paths = {'full': ["../256x256/photo/tx_000000000000", "../256x256/sketch/tx_000000000000"],
                 'mini': ["../256x256/photo/tx_000000000000", "../256x256/sketch/tx_000000000000"]}

# Set DATASET_NAME
DATASET_NAME = 'full'  # or 'mini' based on your requirement

# Load dataset paths based on DATASET_NAME
PHOTO_DATASET_PATH, SKETCHES_DATASET_PATH = dataset_paths[DATASET_NAME]

# Define collection names for photos and sketches
PHOTOS_COLLECTION = db['Photos']
SKETCHES_COLLECTION = db['Sketches']

# Function to save images from a folder and its subfolders to MongoDB collection
def save_images_to_mongodb(folder_path, collection):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                print('Added')
                file_path = os.path.join(root, filename)
                with open(file_path, 'rb') as file:
                    binary_data = Binary(file.read())
                    class_label = root.split('\\')[-1]
                    print(class_label)
                    collection.insert_one({'image': binary_data, 'class': class_label})

    # Add index to the 'image' field after inserting all documents
    collection.create_index([("image", 1)])



# Save photos to MongoDB
save_images_to_mongodb(PHOTO_DATASET_PATH, PHOTOS_COLLECTION)

# Save sketches to MongoDB
save_images_to_mongodb(SKETCHES_DATASET_PATH, SKETCHES_COLLECTION)



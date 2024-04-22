import os
from bson.binary import Binary
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

                    # Extracting common identifier from filename
                    file_name = os.path.splitext(filename)[0]
                    collection.insert_one({'image': binary_data, 'class': class_label, 'common_identifier': file_name})

    # Add index to the 'image' field after inserting all documents
    collection.create_index([("image", 1)])



# Save photos to MongoDB
save_images_to_mongodb(PHOTO_DATASET_PATH, PHOTOS_COLLECTION)

# Save sketches to MongoDB
save_images_to_mongodb(SKETCHES_DATASET_PATH, SKETCHES_COLLECTION)

# Update documents in Photos collection with sketches information
for photo_doc in PHOTOS_COLLECTION.find():
    common_identifier = photo_doc['common_identifier']
    sketches = []
    # Find sketches associated with the current photo
    for sketch_doc in SKETCHES_COLLECTION.find({'common_identifier': {'$regex': f'^{common_identifier}-[0-9]+$'}}):
        sketches.append(sketch_doc['common_identifier'])
    PHOTOS_COLLECTION.update_one({'_id': photo_doc['_id']}, {'$set': {'sketches': sketches}})

# Update documents in Sketches collection with photo information
for sketch_doc in SKETCHES_COLLECTION.find():
    common_identifier = sketch_doc['common_identifier']
    photo_id = common_identifier.split('-')[0]  # Extracting photo id from sketch common identifier
    # Find the corresponding photo
    photo_doc = PHOTOS_COLLECTION.find_one({'common_identifier': photo_id})
    if photo_doc:
        SKETCHES_COLLECTION.update_one({'_id': sketch_doc['_id']}, {'$set': {'photo': photo_id}})



import tkinter as tk
from PIL import Image, ImageTk
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from EmbeddingSpace import *
from Networks import SiameseNetwork
import pyautogui
import torchvision.transforms as transforms
import os
from mongo_connection import db
from mongo_image_ds import MongoImageDataset
import io


'''
SCRIPT DESCRIPTION:
This script displays a GUI where you can draw sketches and obtain the corresponding images.
'''
# Define collection names for photos and sketches
PHOTOS_COLLECTION = db['Photos']
SKETCHES_COLLECTION = db['Sketches']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"We're using {DEVICE} in canvas.py")

# ====================================
#               CONFIG
# ====================================

# Pick a Dataset (you can use the dictionary up here as reference)
DATASET_NAME = 'full'

# Pick an embedding size
#   Must coincide with the model weights
OUTPUT_EMBEDDING = 16

# Choose a Weight Path
#   After the training your weight are going to be saved here
WEIGHT_PATH = f"../weights/{DATASET_NAME}-{OUTPUT_EMBEDDING}-contrastive-resnet34.pth"

# Pick a K (for the K-Precision)
#   It is used show k retrieved images
K = 12

# Pick a Batch Size
BATCH_SIZE = 16

# Pick a Backbone
#   The backbone represents the neural network within the siamese network, 
#   after which several linear layers will be applied to produce an embedding of size EMBEDDING_SIZE.
backbone = models.resnet34()
net = SiameseNetwork(output=OUTPUT_EMBEDDING, backbone=backbone).to(DEVICE)
net.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create dataset
photo_dataset = MongoImageDataset(PHOTOS_COLLECTION, transform=transform)

# Load Dataset
workers = 0

images_loader = DataLoader(photo_dataset, shuffle=False, num_workers=workers, pin_memory=True, batch_size=BATCH_SIZE)

EMBEDDING_SPACE_FILE = "embedding_space.pth"
PREVIOUS_DATASET_SIZE_FILE = "previous_dataset_size.txt"

# Load the previous dataset size if it exists
previous_dataset_size = 0
if os.path.exists(PREVIOUS_DATASET_SIZE_FILE):
    with open(PREVIOUS_DATASET_SIZE_FILE, "r") as file:
        previous_dataset_size = int(file.read().strip())
        
# Function to calculate the size of a MongoDB collection
def calculate_collection_size(collection):
    # Calculate the total size of all documents in the collection
    total_size = sum(len(doc['image']) for doc in collection.find())
    return total_size

# Function to check for changes in the dataset stored in MongoDB
def check_for_dataset_changes():
    # Get the current size of the MongoDB collections
    current_photo_collection_size = calculate_collection_size(PHOTOS_COLLECTION)

    # Check if there are changes in the dataset
    if (current_photo_collection_size != previous_dataset_size):
        update_embedding_space()
        # Update the previous dataset size
        with open(PREVIOUS_DATASET_SIZE_FILE, "w") as file:
            file.write(str(current_photo_collection_size))



def update_embedding_space():
    global embedding_space
    embedding_space = EmbeddingSpace(net, images_loader, DEVICE)
    torch.save(embedding_space, EMBEDDING_SPACE_FILE)

# Check for changes in the dataset folder based on size
current_size_ds_folder = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(PHOTO_DATASET_PATH) for filename in filenames)
if current_size_ds_folder != previous_dataset_size:
    update_embedding_space()
    # Update the previous dataset size file
    with open(PREVIOUS_DATASET_SIZE_FILE, "w") as file:
        file.write(str(current_size_ds_folder))


# ====================================
#                CODE
# ====================================

# Constants
CANVAS_SIZE = 256  # Size of the square canvas
COLUMNS = 6


def clear_canvas():
    canvas.delete("all")


def get_canvas_image():
    # Get the coordinates of the canvas relative to the screen
    canvas_x = root.winfo_rootx() + canvas.winfo_x()
    canvas_y = root.winfo_rooty() + canvas.winfo_y()

    # Capture the screen image within the canvas region
    image = pyautogui.screenshot(region=(canvas_x, canvas_y, CANVAS_SIZE, CANVAS_SIZE))

    # Convert the image to PIL format
    image_pil = Image.frombytes('RGB', image.size, image.tobytes())

    # Apply transformation to convert the image to a tensor
    transform = transforms.ToTensor()
    tensor = transform(image_pil)

    return tensor


def search_images(event):
    # Get the sketch image from the canvas
    sketch = get_canvas_image()

    # Get the top K images that are most similar to the sketch
    topk_distances, topk_indices = embedding_space.top_k(sketch[None, :].to(DEVICE), K)

    # Clear the previous images
    image_frame.delete("all")

    # Display the top K images and their corresponding distances
    for i, (idx, d) in enumerate(zip(topk_indices, topk_distances)):

        # Retrieve the image from the MongoDB collection based on its index
        idx_int = int(idx.item())  # Convert idx to an integer
        img_bytes = PHOTOS_COLLECTION.find_one(skip=idx_int)["image"]
        img = Image.open(io.BytesIO(img_bytes))

        resized_image = img.resize((256, 256))
        resized_image_tk = ImageTk.PhotoImage(resized_image)

        label = tk.Label(image_frame, image=resized_image_tk)
        label.image = resized_image_tk  # Keep a reference to avoid garbage collection
        label.grid(row=i // COLUMNS, column=i % COLUMNS, padx=10, pady=10)

        # Create label for the distance and add to the frame
        text_label = tk.Label(image_frame, text=str(f'{i + 1}° - {d.item():.4}'))
        text_label.grid(row=i // COLUMNS, column=i % COLUMNS, padx=10, pady=10, sticky='n')


# Load the model and embedding space
net.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))

# Load or create embedding space
check_for_dataset_changes()
if os.path.exists(EMBEDDING_SPACE_FILE):
    embedding_space = torch.load(EMBEDDING_SPACE_FILE)
else:
    update_embedding_space()


# Create the main window
root = tk.Tk()
root.title("Image Search")
root.state('zoomed')

# Create the canvas for drawing
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", highlightbackground="black",
                   highlightthickness=2)
canvas.pack(padx=20, pady=20)

# Bind mouse events to canvas
canvas.bind("<B1-Motion>",
            lambda event: canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="black"))
canvas.bind("<ButtonRelease-1>", search_images)

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Create the buttons
clear_button = tk.Button(button_frame, text="Clear Canvas", command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=10)

# Create a frame for the image display
image_frame = tk.Canvas(root)
image_frame.pack(pady=20)

# Run the main event loop
root.mainloop()

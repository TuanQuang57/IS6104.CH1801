import tkinter as tk
from tkinter import ttk
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
import time

import ctypes

myappid = 'IS6103.CH1801.SketchZoo'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


'''
SCRIPT DESCRIPTION:
This script displays a GUI where you can draw sketches and obtain the corresponding images.
'''
# Define collection names for photos and sketches
PHOTOS_COLLECTION = db['Photos']
SKETCHES_COLLECTION = db['Sketches']

print(PHOTOS_COLLECTION.index_information())

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
WEIGHT_PATH = f"../weights/{DATASET_NAME}-{OUTPUT_EMBEDDING}-contrastive-resnet50.pth"

# Pick a K (for the K-Precision)
#   It is used show k retrieved images
K = 12

# Pick a Batch Size
BATCH_SIZE = 16

# Pick a Backbone
#   The backbone represents the neural network within the siamese network, 
#   after which several linear layers will be applied to produce an embedding of size EMBEDDING_SIZE.
backbone = models.resnet50()
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
    start_time = time.time()  # Record start time
    # Get the sketch image from the canvas
    sketch = get_canvas_image()

    # Get the top K images that are most similar to the sketch
    topk_distances, topk_indices = embedding_space.top_k(sketch[None, :].to(DEVICE), K)


    # Calculate time taken for the query
    query_time = time.time() - start_time
    print(f"Time taken for query: {query_time} seconds")

    # Clear the previous images
    image_frame.delete("all")

    start_time = time.time()  # Record start time
    # Display the top K images and their corresponding distances
    for i, (idx, d) in enumerate(zip(topk_indices, topk_distances)):

        # Record start time for fetching image bytes
        fetch_start_time = time.time()

        # Retrieve the image from the MongoDB collection based on its index
        idx_int = int(idx.item())  # Convert idx to an integer
        img_bytes = PHOTOS_COLLECTION.find_one(skip=idx_int)["image"]

        # Calculate time taken for fetching image bytes
        fetch_time = time.time() - fetch_start_time
        print(f"Time taken for fetching image bytes: {fetch_time} seconds")

        img = Image.open(io.BytesIO(img_bytes))

        resized_image = img.resize((256, 256))
        resized_image_tk = ImageTk.PhotoImage(resized_image)

        label = tk.Label(image_frame, image=resized_image_tk)
        label.image = resized_image_tk  # Keep a reference to avoid garbage collection
        label.grid(row=i // COLUMNS, column=i % COLUMNS, padx=10, pady=10)

        # Create label for the distance and add to the frame
        text_label = tk.Label(image_frame, text=str(f'Top: {i + 1} - Distance: {d.item():.4}'), font=('Arial', 10, 'bold'))
        text_label.grid(row=i // COLUMNS, column=i % COLUMNS, padx=10, pady=10, sticky='n')

        # Calculate time taken for displaying images
    display_time = time.time() - start_time
    print(f"Time taken for displaying images: {display_time} seconds")

    total_time = query_time + display_time
    print(f"Total time taken: {total_time} seconds")


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
root.title("SketchZoo Animal Image Retrieval - IS6104.CH1801")
root.iconbitmap('../icon/Logo_DH_UIT.ico')
root.state('zoomed')

def set_pen_cursor(event):
    if erasing:
        canvas.config(cursor="dotbox")  # Set cursor to none when in erase mode
    else:
        canvas.config(cursor="pencil")  # Set cursor to a pencil shape when in draw mode

def set_default_cursor(event):
    canvas.config(cursor="")  # Set cursor back to the default when leaving the canvas

def toggle_erasing():
    global erasing
    erasing = not erasing  # Toggle the value of the erasing flag

def handle_draw(event):
    if erasing:
        # Get the current mouse position
        x, y = event.x, event.y
        # Find all items that intersect with the mouse position
        overlapping_items = canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2)
        # Remove all overlapping items
        for item in overlapping_items:
            canvas.delete(item)
    else:
        # Draw on the canvas
        canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="black")


# Create the canvas for drawing
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", highlightbackground="black",
                   highlightthickness=2)
canvas.pack(padx=20, pady=20)


# Bind mouse events to canvas
# Bind events to change cursor
canvas.bind("<Enter>", set_pen_cursor)  # When cursor enters canvas
canvas.bind("<Leave>", set_default_cursor)  # When cursor leaves canvas
canvas.bind("<B1-Motion>", handle_draw)
canvas.bind("<ButtonRelease-1>", search_images)

# Create a frame to hold the button
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Create a styled button for toggling erasing mode
erase_button = ttk.Button(button_frame, text="Toggle Draw/Erase Mode", command=toggle_erasing, style='TButton', cursor="hand2")
erase_button.pack(side=tk.LEFT, padx=10)

# Create a styled button
clear_button = ttk.Button(button_frame, text="Clear Canvas", command=clear_canvas, style='TButton', cursor="hand2")
clear_button.pack(side=tk.RIGHT, padx=10)

# Define a custom style for the button
style = ttk.Style()
style.configure('TButton', background='#3498db', foreground='black', font=('Arial', 10, 'bold'), padding=10)

# Create a frame for the image display
image_frame = tk.Canvas(root)
image_frame.pack(pady=20)

# Flag to indicate whether erasing mode is active
erasing = False


# Run the main event loop
root.mainloop()

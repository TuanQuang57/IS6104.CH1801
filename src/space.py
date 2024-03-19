import matplotlib.pyplot as plt
from EmbeddingSpace import *
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from Networks import SiameseNetwork
from mongo_connection import db
from mongo_image_ds import MongoImageDataset

'''
SCRIPT DESCRIPTION:

This script is responsible for plotting the embedding space of a model in a 2D space (where EMBEDDING_SIZE = 2).
'''

# MongoDB collections
PHOTOS_COLLECTION = db['Photos']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# ====================================
#               CONFIG
# ====================================

#Pick a Dataset (you can use the dictionary up here as reference)
DATASET_NAME = 'full'

#Pick an embedding size
#   MUST BE EQUAL TO 2
OUTPUT_EMBEDDING = 16

#Pick a Backbone
#   The backbone represents the neural network within the siamese network, 
#   after which several linear layers will be applied to produce an embedding of size EMBEDDING_SIZE.
backbone = models.resnet34()

#Load a model (the embedding_size MUST be equal to 2)
net = SiameseNetwork(output = OUTPUT_EMBEDDING, backbone = backbone).to(DEVICE)
net.load_state_dict(torch.load("../weights/full-16-contrastive-resnet34.pth", map_location=torch.device('cpu')))

#Pick a Batch Size
BATCH_SIZE = 16

# Create MongoDB dataset
photo_dataset = MongoImageDataset(PHOTOS_COLLECTION, transform=transforms.ToTensor())


# Split dataset
train_size = int(0.8 * len(photo_dataset))
val_size = len(photo_dataset) - train_size
photo_train_dataset, photo_val_dataset = random_split(photo_dataset, [train_size, val_size])

# Data loaders
workers = 0  # Adjust as needed
photo_train_loader = DataLoader(photo_train_dataset, shuffle=False, num_workers=workers, batch_size=BATCH_SIZE)
photo_val_loader = DataLoader(photo_val_dataset, shuffle=False, num_workers=workers, batch_size=BATCH_SIZE)


#Pick the dataset that you want to embed
loader = photo_val_loader

embedding_space = EmbeddingSpace(net, loader, DEVICE)

x = torch.permute(embedding_space.embeddings, (1,0)).tolist()[0]
y = torch.permute(embedding_space.embeddings, (1,0)).tolist()[1]

colours_dict = {0 : 'brown', 1 : 'green', 2 : 'blue', 3: 'yellow', 4 :'black'}
# Convert classes to integers if they are floats
classes = [int(c.item()) if isinstance(c.item(), float) else c.item() for c in embedding_space.classes]

# Assign colors based on classes, with a default color for out-of-range values
colors = [colours_dict.get(c, 'gray') for c in classes]

plt.scatter(x, y, color = colors)
plt.show()
from EmbeddingSpace import *
from torchvision import models
from Metrics import k_precision
from Utils import fix_random
from Networks import SiameseNetwork
from torchvision import transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from mongo_connection import db
from mongo_image_ds import MongoImageDataset

'''
SCRIPT DESCRIPTION:

This script is used to print a chart that compares various k-precision values of different models using a bar graph.
'''
# Define collection names for photos and sketches
PHOTOS_COLLECTION = db['Photos']
SKETCHES_COLLECTION = db['Sketches']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"We're using {DEVICE} in eval.py")
fix_random(42)
generator1 = torch.Generator().manual_seed(42)
workers = 0





# ====================================
#               CONFIG
# ====================================

#Pick a Dataset (you can use the dictionary up here as reference)
DATASET_NAME = 'full'

#Pick a K (for the K-Precision)
#   It is used to represent the k factor for calculating k-accuracy 
K = 12

#Pick a Batch Size
BATCH_SIZE = 16

#Create the dictionary for the final graph
#   For each experiment pick the backbone, the embedding_size and the weight_path
dict={0: {'backbone' : models.resnet18(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-contrastive.pth'},
      1: {'backbone' : models.resnet34(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-contrastive-resnet34.pth'},
      2: {'backbone' : models.resnet50(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-contrastive-resnet50.pth'},
      3: {'backbone' : models.resnet34(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-triplet-resnet34.pth'}}

dict_32={
    0: {'backbone' : models.resnet34(), 'embedding_size' : 32, 'weight_path' : '../weights/full-32-contrastive.pth'},
    1: {'backbone' : models.resnet34(), 'embedding_size' : 32, 'weight_path' : '../weights/full-32-triplet.pth'}}

# EMBEDDING_SPACE_FILE = "eval_embedding_space.pth"



# ====================================
#                CODE
# ====================================

#Images and Sketch
# images_ds = ImageFolder(PHOTO_DATASET_PATH, transform = transforms.ToTensor())
# images_loader = DataLoader(images_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)
# images_train_ds, images_val_ds = random_split(images_ds, (0.8, 0.2), generator = generator1)
# images_train_dl = DataLoader(images_train_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)
# images_val_dl = DataLoader(images_val_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)

# sketches_ds = ImageFolder(SKETCHES_DATASET_PATH, transform = transforms.ToTensor())
# sketches_loader = DataLoader(sketches_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)
# sketches_train_ds, sketches_val_ds, sketches_k_acc = random_split(sketches_ds, (0.8, 0.15, 0.05), generator = generator1)
# small_sketches_loader = DataLoader(sketches_k_acc, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)


def update_embedding_space():
    global embedding_space
    embedding_space = EmbeddingSpace(net, photo_val_loader, DEVICE)
    # torch.save(embedding_space, EMBEDDING_SPACE_FILE)

# Define your data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Load images and sketches from MongoDB
photo_dataset = MongoImageDataset(PHOTOS_COLLECTION, transform=transform)
sketches_dataset = MongoImageDataset(SKETCHES_COLLECTION, transform=transform)

# Split datasets
train_size = int(0.8 * len(photo_dataset))
val_size = len(photo_dataset) - train_size
photo_train_dataset, photo_val_dataset = random_split(photo_dataset, [train_size, val_size])


train_size = int(0.8 * len(sketches_dataset))
val_size = int(0.15 * len(sketches_dataset))
test_size = len(sketches_dataset) - train_size - val_size
sketches_train_dataset, sketches_val_dataset, small_sketches_dataset = random_split(sketches_dataset, [train_size, val_size, test_size])


# Data loaders
workers = 0
photo_train_loader = DataLoader(photo_train_dataset, shuffle=True, num_workers=workers, pin_memory=True, batch_size=BATCH_SIZE)
photo_val_loader = DataLoader(photo_val_dataset, shuffle=False, num_workers=workers, pin_memory=True, batch_size=BATCH_SIZE)

sketches_train_loader = DataLoader(sketches_train_dataset, shuffle=True, num_workers=workers, pin_memory=True, batch_size=BATCH_SIZE)
sketches_val_loader = DataLoader(sketches_val_dataset, shuffle=False, num_workers=workers, pin_memory=True, batch_size=BATCH_SIZE)
small_sketches_loader = DataLoader(small_sketches_dataset, shuffle=False, num_workers=workers, pin_memory=True, batch_size=BATCH_SIZE)


# if os.path.exists(EMBEDDING_SPACE_FILE):
#     embedding_space1 = torch.load(EMBEDDING_SPACE_FILE)
# else:
#     update_embedding_space()

results = []

for k in dict.keys():
    net = SiameseNetwork(output = dict[k]['embedding_size'], backbone = dict[k]['backbone']).to(DEVICE)
    net.load_state_dict(torch.load(dict[k]['weight_path'], map_location=torch.device('cpu')))
    net = net.eval()
    
    with torch.no_grad():
        embedding_space1 = EmbeddingSpace(net, photo_val_loader, DEVICE)
        results.append(k_precision(net, small_sketches_loader, embedding_space1, K, DEVICE))
        print(f'Model {k} has K@{K} = {results[k]:.2f}')


plt.bar(["Resnet18 w Contrastive Loss", "Resnet34 w Contrastive Loss", "Resnet50 w Contrastive Loss", "resnet34 w Triplet Loss"], results)
plt.show()


    
    




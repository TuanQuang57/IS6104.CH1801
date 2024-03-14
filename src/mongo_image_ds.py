from torch.utils.data import Dataset
from PIL import Image
import io

class MongoImageDataset(Dataset):
    def __init__(self, collection, transform=None):
        self.collection = collection
        self.transform = transform
        self.class_to_numeric = {}  # Mapping from string class labels to numeric labels
        self.numeric_to_class = {}  # Mapping from numeric labels to string class labels
        self._create_class_mapping()

    def _create_class_mapping(self):
        # Retrieve all unique class labels from MongoDB collection
        unique_classes = self.collection.distinct("class")

        # Create mapping from string class labels to numeric labels and vice versa
        for idx, class_label in enumerate(unique_classes):
            self.class_to_numeric[class_label] = idx
            self.numeric_to_class[idx] = class_label

    def __len__(self):
        return self.collection.count_documents({})

    def __getitem__(self, idx):
        doc = self.collection.find_one({}, skip=idx)
        img_bytes = doc["image"]  # Assuming image data is stored as "image" field in MongoDB
        img = Image.open(io.BytesIO(img_bytes))
        class_label_str = doc["class"]  # Assuming class label is stored as "class" field in MongoDB
        class_label_numeric = self.class_to_numeric[class_label_str]  # Convert string label to numeric label

        if self.transform:
            img = self.transform(img)
            # class_label = self.transform(class_label)
        return img, class_label_numeric
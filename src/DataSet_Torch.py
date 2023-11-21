import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SceneDataset(Dataset):
    def __init__(self, root_dir, sequence_length=3, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        self.scenes = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.scenes.sort()

        self.image_paths = []
        for scene in self.scenes:
            images = [os.path.join(scene, f) for f in os.listdir(scene) if f.endswith('.jpg')]
            images.sort()
            self.image_paths.extend(images)

    def __len__(self):
        return len(self.image_paths) - self.sequence_length

    def __getitem__(self, idx):
        sequence = []
        for i in range(self.sequence_length):
            img_path = self.image_paths[idx + i]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            sequence.append(img)
        return torch.stack(sequence, dim=0)

root_dir = '/home/vyas/CVIP/project/sync'
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = SceneDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)



# print(type(dataset))
# Author Sai Karthik Vyas
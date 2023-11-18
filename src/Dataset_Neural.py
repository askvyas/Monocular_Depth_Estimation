import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt 
import os


class DatasetNeural(Dataset):
    def __init__(self, root_dir, transform):

        self.root_dir = root_dir
        self.transform=transform
        #list of all image paths both rgb and depth
        self.img_paths=[os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        # mapping of rgb to depth
        self.paired_images = []
        for scene in self.img_paths:
            rgb_images = [f for f in os.listdir(scene) if f.startswith('rgb_') and f.endswith('.jpg')]
            depth_images = [f for f in os.listdir(scene) if f.startswith('sync_depth_') and f.endswith('.png')]

            for rgb_image in rgb_images:
                number = rgb_image.split('_')[1].split('.')[0]
                corresponding_depth = f'sync_depth_{number}.png'
                if corresponding_depth in depth_images:
                    self.paired_images.append((os.path.join(scene, rgb_image), 
                                               os.path.join(scene, corresponding_depth)))


    def __len__(self):
            return len(self.paired_images)

    from PIL import Image

    def __getitem__(self, idx):
        rgb_path, depth_path = self.paired_images[idx]

        # Load the RGB image and apply transformations
        rgb_image = Image.open(rgb_path).convert('L')
        if self.transform:
            rgb_image = self.transform(rgb_image)

        # Load the depth image as is, without converting to RGB
        depth_image = Image.open(depth_path)
        if self.transform:
            depth_image = self.transform(depth_image)

        return rgb_image, depth_image
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

imsize = 512 if torch.cuda.is_available() else 128  

transform = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.ToTensor()])  

root_dir = '/home/vyas/CVIP/project/sync'

dataset = DatasetNeural(root_dir, transform=transform)


def show_images(dataset, num_pairs=3):
    fig, axs = plt.subplots(num_pairs, 2, figsize=(10, num_pairs * 5))

    for i in range(num_pairs):
        rgb_image, depth_image = dataset[i]  # Fetch the i-th pair of images

        # Convert the RGB image tensor to numpy array for plotting
        rgb_image = rgb_image.numpy().transpose(1, 2, 0)
        axs[i, 0].imshow(rgb_image)
        axs[i, 0].set_title(f'RGB Image {i}')
        axs[i, 0].axis('off')

        # Handle depth image
        depth_image = depth_image.squeeze(0).numpy()  # Assuming depth_image is a single-channel image
        # Optionally adjust the range or data type here if necessary
        axs[i, 1].imshow(depth_image, cmap='gray')  # Use a grayscale colormap
        axs[i, 1].set_title(f'Depth Image {i}')
        axs[i, 1].axis('off')

    plt.show()

show_images(dataset)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from  Dataset_Neural import DatasetNeural
from PIL import Image
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader



root_dir = '/home/vyas/CVIP/project/sync'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128  


transform = transforms.Compose([
    transforms.Resize(imsize), 
    transforms.ToTensor()      
])


import torch.nn as nn

# class CNN_Model(nn.Module):
#     def __init__(self) -> None:
#         super(CNN_Model, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  
#         self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.max_pool2d(x, 2)
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.max_pool2d(x, 2)
#         x = nn.functional.relu(self.conv3(x))  
#         x = torch.sigmoid(self.conv4(x))
#         return x
class CNN_Model(nn.Module):
    def __init__(self) -> None:
        super(CNN_Model, self).__init__()
        # Modification : Added batchnormalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.upconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.upconv1(x)))
        x = nn.functional.relu(self.bn5(self.upconv2(x)))
        x = self.upconv3(x)

        return x





model=CNN_Model()
model.to(device)
#new learning 0.0001 trial
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

#might need to change the loss function 
#Tried MSE, L1 smooth loss
criterion=nn.SmoothL1Loss()

num_epochs = 3

dataset = DatasetNeural(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False,num_workers=0)
# print(dataloader)

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0

    for i, (rgb_images, depth_images) in enumerate(dataloader):
        # print(type(rgb_images))
        rgb_images = rgb_images.to(device)
        depth_images = depth_images.to(device, dtype=torch.float)

        

        optimizer.zero_grad()

        outputs = model(rgb_images)
        outputs = F.interpolate(outputs, size=depth_images.shape[2:], mode='bilinear', align_corners=False)


        # print(outputs.shape)
        loss = criterion(outputs, depth_images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99: 
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')



#------------------------------------------------------------------------------- display------------------------------------------------------------------------

from torchvision import transforms
from PIL import Image

image_path = "/home/vyas/CVIP/project/sync/basement_0001a/rgb_00000.jpg"  
transform = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.ToTensor()
])

image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)  

image = image.to(device)

model.eval()
with torch.no_grad():
    prediction = model(image)

predicted_depth = prediction.squeeze().cpu().numpy()
#normalization added (trial)
predicted_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
plt.imshow(predicted_depth, cmap='gray')
plt.title("Depth Image")
plt.axis('off')
plt.show()


torch.save(model.state_dict(), "/home/vyas/CVIP/project/model.pth")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DataSet_Torch import dataloader,dataset

class FramePredictionCNN(nn.Module):
    def __init__(self):
        super(FramePredictionCNN, self).__init__()
        # Assuming input frames are concatenated along the channel dimension
        self.conv1 = nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # Sigmoid to ensure output range [0, 1]
        return x

model = FramePredictionCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        # Get the input sequences and labels
        sequences = data.to(device)
        inputs = sequences[:, :2, :, :].reshape(-1, 6, 128, 128)  # Concatenate first 2 frames
        targets = sequences[:, 2, :, :].reshape(-1, 3, 128, 128)  # Predict the 3rd frame

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')

print('Finished Training')

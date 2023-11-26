#implementing neural style transfer from source: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from Dataset_Neural import DatasetNeural
import copy

from torchvision.transforms import Grayscale
from torchvision.transforms.functional import to_tensor, to_pil_image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

imsize = 512 if torch.cuda.is_available() else 128  

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()      
])




root_dir = '/home/vyas/CVIP/project/sync'
dataset = DatasetNeural(root_dir, transform=loader)

content_img, style_img = dataset[0]
print(f"Content image size: {content_img.size()}")
print(f"Style image size: {style_img.size()}")
content_img = content_img.to(device).unsqueeze(0)
style_img = style_img.to(device).unsqueeze(0)

def convert_to_rgb(image):
    # Remove the batch dimension (BxCxHxW to CxHxW)
    image = image.squeeze(0)
    image_pil = to_pil_image(image)
    image_rgb = image_pil.convert("RGB")
    # Add the batch dimension back (CxHxW to BxCxHxW)
    return to_tensor(image_rgb).unsqueeze(0)


if style_img.size(0) == 1:
    style_img = convert_to_rgb(style_img)


assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


unloader = transforms.ToPILImage()  

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)      
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)  

    G = torch.mm(features, features.t())  

    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()




cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses



input_img = content_img.clone()
plt.figure()
imshow(input_img, title='Input Image')



def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer



def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=100, content_weight=20):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img



output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, content_img.clone())



to_grayscale = transforms.Compose([
    Grayscale(num_output_channels=1),
    transforms.ToPILImage()
])


output_gray = to_grayscale(output.to(device).squeeze(0))

to_tensor = transforms.ToTensor()
output_gray_tensor = to_tensor(output_gray)



image_path = 'results/neuralstyle_op.png'
plt.imsave(image_path, output)

plt.figure()
imshow(output_gray_tensor, title='Output Image')
plt.ioff()
plt.show()



image_path = 'results/midas_op.png'
plt.imsave(image_path, output)

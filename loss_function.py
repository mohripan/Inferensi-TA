import torch
import torch.nn as nn
from torchvision.models import vgg19
from pytorch_pretrained_vit import ViT

# Load pre-trained VGG model for perceptual loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg19(pretrained=True).features[:36].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# Define the combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, vgg, device, lambda_perceptual=0.01, lambda_l1=1):
        super(CombinedLoss, self).__init__()
        self.vgg = vgg
        self.lambda_perceptual = lambda_perceptual
        self.lambda_l1 = lambda_l1
        self.l1_loss = nn.L1Loss()

    def forward(self, sr_images, hr_images):
        sr_features = self.vgg(sr_images)
        hr_features = self.vgg(hr_images)

        perceptual_loss = self.l1_loss(sr_features, hr_features)
        l1_loss = self.l1_loss(sr_images, hr_images)

        loss = self.lambda_perceptual * perceptual_loss + self.lambda_l1 * l1_loss
        return loss
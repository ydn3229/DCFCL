from torch import nn
from torch.nn import functional as F

from FLAlgorithms.additional.ResNet import resnet18_cbam

class S_ConvNet(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size, xa_dim, num_classes = 10):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size
        self.num_classes = num_classes
        
        # layers
        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=4, stride=2, padding=1
        )

        self.conv2 = nn.Conv2d(
            channel_size, channel_size*2,
            kernel_size=4, stride=2, padding=1
        )
        
        self.conv3 = nn.Conv2d(
            channel_size*2, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        
        self.fc1 = nn.Linear((image_size//8)**2 * channel_size*4, xa_dim)

        self.fc2 = nn.Linear(xa_dim, xa_dim)
        
        # aux-classifier fc
        self.fc_classifier = nn.Linear(xa_dim, self.num_classes)
        
        # activation functions:
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xa = self.forward_to_xa(x)
        classes_p, logits = self.forward_from_xa(xa)

        return classes_p, xa, logits  
    
    def forward_to_xa(self, x):
        xa = self.conv1(x)
        xa = self.conv2(xa)
        xa = self.conv3(xa)

        xa = xa.view(xa.shape[0], (self.image_size//8)**2 * self.channel_size*4)

        xa = F.relu(self.fc1(xa))
        return xa

    def forward_from_xa(self, xa):
        xb = F.relu(self.fc2(xa))
        logits = self.fc_classifier(xb)        
        classes_p = self.softmax(logits)

        return classes_p, logits

class Resnet_plus(nn.Module):
    def __init__(self, image_size, xa_dim, num_classes = 10):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        
        # layers
        self.features = resnet18_cbam(False)
        channel_size = 512
        # self.fc1 = nn.Linear(channel_size, xa_dim)
        # self.fc2 = nn.Linear(xa_dim, xa_dim)
        
        # # aux-classifier fc
        # self.fc_classifier = nn.Linear(xa_dim, self.num_classes)
        
        # # activation functions:
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xa = self.forward_to_xa(x)
        # xa = self.softmax(xa)
        classes_p, logits = self.forward_from_xa(xa)

        return classes_p, xa, logits  
    
    def forward_to_xa(self, x):
        xa = x # [N, 512]
        return xa

    def forward_from_xa(self, xa):
        # xa = self.fc1(xa)
        # xb = self.fc2(xa)
        logits = self.features(xa)        
        # classes_p = self.softmax(logits)

        return logits, logits
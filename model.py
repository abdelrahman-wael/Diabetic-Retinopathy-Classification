from torchvision import models
import torch.nn as nn 
import torch.optim as optim
import os
import torch.nn.functional as F

class DR_Classifier(nn.Module):
    def __init__(self):
        super(DR_Classifier, self).__init__()
#         self.densenet = models.densenet161(pretrained = True)
        self.wideresnet = models.wide_resnet101_2(pretrained = True)

        num_ftrs = self.wideresnet.fc.in_features
        self.wideresnet.fc = nn.Linear(num_ftrs, 1000)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 5)
        self.dropout = nn.Dropout()

    def forward(self, imgs):
        res = F.relu(self.wideresnet(imgs))
        res = self.dropout(res)
        res = F.relu(self.fc1(res))
        res = self.dropout(res)
        res = F.relu(self.fc2(res))
        res = self.dropout(res)
        res = self.fc3(res)
        return res


def getModelAndOptimizer(weights = "./weights/",lr = 0.0001 , weight_decay=0 ):
  if not (weights in os.listdir()):
    print("no weights found")
  model = DR_Classifier()
  optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay = weight_decay)

  return model , optimizer

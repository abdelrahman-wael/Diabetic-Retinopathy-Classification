from torchvision import models
import torch.nn as nn 
import torch.optim as optim
import os
import torch.nn.functional as F
import torch
from efficientnet_pytorch import EfficientNet

class DR_Classifier(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=False,
                 aux_logits=False, freeze_features=False):
        super(DR_Classifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.aux_logits = aux_logits

        if self.model_name == 'inception-v3':
            self.model = models.inception_v3(pretrained=pretrained, 
                                             aux_logits=aux_logits)
            num_ftrs = self.model.fc.in_features
            self.model.fc = self.createLayer(num_ftrs)
            if self.aux_logits:
                num_ftrs = self.model.AuxLogits.fc.in_features 
                self.model.AuxLogits.fc = self.createLayer(num_ftrs)
        elif self.model_name == "densenet-161":
            self.model = models.densenet161(pretrained=pretrained)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = self.createLayer(num_ftrs)
        elif self.model_name == "efficientnet-b7":
            self.model = EfficientNet.from_pretrained(model_name)
            num_ftrs = self.model._fc.in_features
            self.model._fc = self.createLayer(num_ftrs)
        elif self.model_name == "efficientnet-b5":
            self.model = EfficientNet.from_pretrained(model_name)
            num_ftrs = self.model._fc.in_features
            self.model._fc = self.createLayer(num_ftrs)
        elif self.model_name == "wideresnet":
          self.model = models.wide_resnet101_2(pretrained = True)
          num_ftrs = self.model.fc.in_features
          self.model.fc = self.createLayer(num_ftrs)
        else:
            print("Invalid model name, exiting...")
            return;


    def classes_number(self):
        return self.num_classes

    def forward(self, imgs):
        res = self.model(imgs)
        return res

    def createLayer(self, num_ftrs):
        big_layer = nn.Sequential(
                    nn.Linear(num_ftrs, 768), 
                    nn.ReLU(), 
                    nn.Dropout(),
                    nn.Linear(768, 256),
                    nn.ReLU(), 
                    nn.Dropout(),
                    nn.Linear(256, self.num_classes))
        return big_layer



def getModelAndOptimizer(device ,hyperparameters):

  availableModels = os.listdir("/content/drive/MyDrive/Visualization/saved Models")
  modelName = hyperparameters["modelName"]
  model=DR_Classifier(modelName)
  learningRate = hyperparameters["learningRate"]
  weightDecay =hyperparameters["weightDecay"]

  if hyperparameters["pretrainedPath"]: 
    weights=torch.load(hyperparameters["pretrainedPath"])
    model.load_state_dict(weights["model_state"])
    model=model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate , weight_decay = weightDecay)
    optimizer.load_state_dict(weights["optimizer_state"])
    print("the weights are loaded")

  else:
    optimizer = optim.Adam(model.parameters(), lr=learningRate , weight_decay = weightDecay)
  return model , optimizer
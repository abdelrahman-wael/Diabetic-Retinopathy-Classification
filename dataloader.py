from torch.utils.data import Dataset , DataLoader
from utils.helpers import *
from torchvision import transforms
import json 

class fundsDataset(Dataset):
  def __init__(self,imagesPath , maksPath  , gradingCsv , transform = None  ):
    super(fundsDataset, self).__init__()
    with open(maksPath, 'r') as fp:
      self.masksPath = json.load(fp)
    self.imagesPath = imagesPath
    self.grade = imagesPath
    self.img_transform = transform

  def __getitem__(self,index):
    image = self.imagesPath[index]
    mask = self.masksPath[image]
    dr_grade = self.class_[image]

    if self.img_transform:
      image = self.img_transform(image)
      dr_grade = self.img_transform(dr_grade)

    sample = {'image': image , 'dr_grade': dr_grade , "mask": mask}
    return sample

  

  def addAllMasks(masks):
    maskShape = imageio.imread(masks[0]).shape 
    overlayMask = np.zeros(maskShape)
    for mask in masks:
      overlayMask += imageio.imread(mask)

    return overlayMask
  
  def __len__(self):
    return len(self.imagesPath)

def getTransformation():
  return transforms.Compose([transforms.ToTensor()]) 


def dataLoader(imagesPath , masksPath , grading):
  trainImagesPath , validImagesPath , testImagesPath = splitData(imagesPath,grading)
  print("trainImagesPath = " , len(trainImagesPath))
  print("validImagesPath = " , len(validImagesPath))
  print("testImagesPath = " , len(testImagesPath))
  transformation=getTransformation()
  trainDataset = fundsDataset(trainImagesPath,masksPath,grading,transformation)
  validDataset = fundsDataset(validImagesPath,masksPath,grading,transformation)
  testDataset = fundsDataset(testImagesPath,masksPath,grading,transformation)

  return trainDataset,validDataset,testDataset

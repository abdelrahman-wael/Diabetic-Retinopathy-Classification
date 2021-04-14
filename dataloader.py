from torch.utils.data import Dataset , DataLoader
from utils.helpers import *
from torchvision import transforms
import json 
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch 
import cv2

class fundsDataset(Dataset):
  def __init__(self,imagesPath , masksPath  , gradingCsv , visualization=False,transform = None):
    super(fundsDataset, self).__init__()
    with open(masksPath, 'r') as fp:
      self.masksPath = json.load(fp)
    self.imagesName = os.listdir()
    self.imagesPath = imagesPath
    self.grade = gradingCsv
    self.img_transform = transform
    self.images = self.imgDic(imagesPath)
    self.visualization = visualization
    
  def imgDic(self,imagesPath):
    images={}
    for imagePath in tqdm(imagesPath , leave = False , position=0):
      image = imageio.imread(imagePath)
      images[imagePath] = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return images

  def __getitem__(self,index):
    imageName = self.imagesPath[index]
    image = self.images[imageName]
    imageName = imageName.split("/")[-1]
    masks = self.masksPath[imageName]
    dr_grade = self.grade[imageName]
    dr_grade = np.array(dr_grade)
    
    if self.img_transform:
      image = self.img_transform(image)
    dr_grade = torch.tensor(dr_grade)

    masks = self.embeddMask(masks)

    if self.visualization:
      overlappedMasks = addAllMasks(masks)
      overlappedMasks = cv2.resize(overlappedMasks, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
      overlappedMasks = self.img_transform(overlappedMasks)
    else:
      overlappedMasks = 0

    return {"imageName":imageName,'image': image , 'dr_grade': dr_grade , "masks":masks , "overlappedMasks":overlappedMasks }

  def embeddMask(self,masks):
    embedding = (7-(len(masks)))*[""]
    return masks + embedding


  def read_csv(self,csv):
    label = pd.read_csv(csv , index_col=0 , header = None)
    dic=label.to_dict()
    dic = dic[1]
    return dic
  
  def __len__(self):
    return len(self.imagesPath)

def getTransformation():
  return transforms.Compose([transforms.ToTensor()]) 



def dataLoader(imagesPath , masksPath , grading ,hyperparameters,num_workers=2, pin_memory=False):
  trainImagesPath , validImagesPath , testImagesPath , visualizationDic = splitData(imagesPath,grading,hyperparameters["testSize"])
  batchSize = hyperparameters["batchSize"]
  print("trainImagesPath = " , len(trainImagesPath))
  print("validImagesPath = " , len(validImagesPath))
  print("testImagesPath = " , len(testImagesPath))
  transformation=getTransformation()
  trainDataset = fundsDataset(trainImagesPath,masksPath,grading,transformation)
  validDataset = fundsDataset(validImagesPath,masksPath,grading,transformation)
  testDataset = fundsDataset(testImagesPath,masksPath,grading,transformation)
  trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,shuffle=True, num_workers=num_workers,
                                             pin_memory=pin_memory)
  validDataLoader = DataLoader(validDataset, batch_size=batchSize,shuffle=False, num_workers=num_workers,
                                             pin_memory=pin_memory)
  testDataLoader =  DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=num_workers,
                                             pin_memory=pin_memory)

  return trainDataLoader,validDataLoader,testDataLoader,visualizationDic
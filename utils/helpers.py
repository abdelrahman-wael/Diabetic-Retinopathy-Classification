
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import imageio
import numpy as np
import torch 

def getMetrics(preds_numpy,labels_numpy):
  currentKappa = cohen_kappa_score(labels_numpy, preds_numpy, weights='quadratic')
  currentAccuracy = accuracy_score(labels_numpy, preds_numpy)
  confusionMatrix = confusion_matrix(labels_numpy,preds_numpy)
  return currentKappa , currentAccuracy , confusionMatrix
  

def IOU(visualization , mask , device = "cpu"):
  print(np.max(visualization))
  print(np.min(visualization))
  print(np.max(mask))
  print(np.min(mask))
  mask = mask.to(device)
  visualization = visualization.to(device)
  intersection = visualization*mask
  union = visualization+mask
  x=torch.ones(1)
  union = torch.where( union <= 1, 
  union, 
  x.double())
  union = torch.sum(union,(2,3))
  intersection = torch.sum(intersection , (2,3))
  iou = intersection / union
  del intersection,union
  return iou

#  get all masks for each corresponding image note lessionDir and imagesDir are folders names 
# and current Dir is the dir of the dataset
def getLessionsPath(lessionsDir,imagesDir):
  imagesWithMask = {}
  currentDir = os.getcwd()
  for image in imagesDir:
    maskPaths = []
    for dir in lessionsDir:
      if( image in os.listdir(dir)):
        maskPaths += [currentDir + "/"+dir + "/" + image]

    imagesWithMask[image] = maskPaths

  return imagesWithMask


# giving all masks absolute paths, return overlapped mask 
def addAllMasks(masks):
  maskShape = imageio.imread(masks[0],as_gray=True).shape 
  overlappedMask = np.zeros(maskShape)
  for mask in masks:
    if not(mask == ""):  
      overlappedMask += imageio.imread(mask,as_gray=True)

  return overlappedMask

# read csv and return dictionary with (image:DRgrade)
def read_csv(gradePath):
  label = pd.read_csv(gradePath , index_col=0 , header = None)
  dic=label.to_dict()
  dic = dic[1]
  return dic



def loadPaths(datasetFolderPath):
  try:
    imagesPath = datasetFolderPath+"/Original_Images"
    masksPath = datasetFolderPath+"/imagesWithMask.json"
    gradingCsv = datasetFolderPath + "/DR_Seg_Grading_Label.csv"
  except:
    print("error occured when trying to read paths from dataset folder ")

  imagesPath = os.listdir(imagesPath)
  imagesPath= [datasetFolderPath+"/Original_Images/"+image for image in imagesPath]
  gradingCsv = read_csv(gradingCsv)

  return imagesPath , masksPath , gradingCsv

# will be tweaked to retrive certain lesions classes "hard exudates, soft exudates" 
def visualizationData(imagesPath,grade,masks=None,testSize=0.1):
  trainImagesPath, validImagesPath, trainGrade , testGrade = train_test_split( imagesPath, grade, test_size=testSize, random_state=42, stratify=grade)
  visualizationDic = {validImagePath:0 for validImagePath in validImagesPath}
  return visualizationDic  

def splitData (imagesPath,grade,testSize = 0.3):
  # retrieve dic values and transform it to a list
  grade=list(grade.values())
  trainImagesPath, testImagesPath, trainGrade , testGrade = train_test_split( imagesPath, grade, test_size=testSize, random_state=42, stratify= grade)
  testImagesPath, validImagesPath, testGrade ,validGrade = train_test_split( testImagesPath, testGrade, test_size=0.5, random_state=42, stratify= testGrade)
  visualizationDic = visualizationData(validImagesPath,validGrade)
  return trainImagesPath, validImagesPath , testImagesPath ,visualizationDic
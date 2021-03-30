
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
import imageio

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

def getMetrics(preds_numpy,labels_numpy):
  currentKappa = cohen_kappa_score(labels_numpy, preds_numpy, weights='quadratic')
  currentAccuracy = accuracy_score(labels_numpy, preds_numpy)
  return currentKappa , currentAccuracy 
  


# giving all masks absolute paths, return overlapped mask 
def addAllMasks(masks):
  maskShape = imageio.imread(masks[0]).shape 
  overlappedMask = np.zeros(maskShape)
  for mask in masks:
    overlappedMask += imageio.imread(mask)

  return overlappedMask


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
  print(imagesPath , masksPath , gradingCsv)

  imagesPath = os.listdir(imagesPath)
  gradingCsv = read_csv(gradingCsv)

  return imagesPath , masksPath , gradingCsv

def splitData (imagesPath,grade,test_size_ = 0.3):
  # retrieve dic values and transform it to a list
  grade=list(grade.values())
  trainImagesPath, testImagesPath, trainGrade , testGrade = train_test_split( imagesPath, grade, test_size=test_size_, random_state=42, stratify= grade)
  testImagesPath, validImagesPath, testGrade ,validGrade = train_test_split( testImagesPath, testGrade, test_size=0.5, random_state=42, stratify= testGrade)
  return trainImagesPath, validImagesPath , testImagesPath
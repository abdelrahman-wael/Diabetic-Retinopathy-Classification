from dataloader import *
from utils.helpers import *



def main():
  datasetFolderPath = "/content/drive/MyDrive/dataset"

  imagesPath , masksPath , gradingCsv = loadPaths(datasetFolderPath)
  trainingLoader , validationLoader , testLoader = dataLoader(imagesPath , masksPath , gradingCsv ) 

  








if __name__=="__main__":
  main()
from dataloader import *
from utils.helpers import *
from comet_ml import Experiment
from comet_ml import ExistingExperiment



def main():
  with open("train_config.txt", "r") as read_file:
    hyperparameters = json.load(read_file) 
  
  datasetFolderPath = "/content/drive/MyDrive/dataset"

  imagesPath , masksPath , gradingCsv = loadPaths(datasetFolderPath)
  trainingLoader , validationLoader , testLoader = dataLoader(imagesPath , masksPath , gradingCsv ) 
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', device)
  model , optimizer = getModelAndOptimizer(device , hyperparameters)








if __name__=="__main__":
  main()
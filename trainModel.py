from time import sleep
from runModel import *

def trainModel(model , optimizer , trainDataloader , validDataloader , device ,weightsDir = "/content/drive/MyDrive/Visualization/weights" , epoch = 20):
  model.to(device)
  for i in range(epoch):
    trainingMetrics = runModel(model , optimizer , trainDataloader , device ,train = True)
    # this prevent tqdm from falling into a deadlock
    # cometML()
    sleep(1)
    validMetrics = runModel(model , optimizer , validDataloader , device ,train = False)

    # cometML()
    
    # save best model

    print("================finished epoch {}===============".format(i))
    sleep(1)


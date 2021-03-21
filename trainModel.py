import torch

def trainModel(model , optimizer , trainDataloader , validDataloader , device , epoch = 20):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', device)
  model.to(device)
  for i in range(epoch):
    trainPred,trainGroundTruth=runModel(model , optimizer , trainDataloader , device ,train = True)
    # print train metric
    # printMetrics(pred,groundTruth)
    
    # trainPred,trainGourndTruth = runModel(model , optimizer , validDataloader , train = False)
    # print validation metric

    # cometml

    # save best model




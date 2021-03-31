from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import torch.nn as nn
import torch 
from utils.helpers import getMetrics
from tqdm import tqdm 
def runModel(model , optimizer , dataloader   , device ,train):

  
  if train:
    model.train()
    mode = "training"
  else:
    model.eval()
    mode = "validation"

  lossfn = nn.CrossEntropyLoss()
  # check if gpu is available  

  # Initialize the prediction and label lists(tensors)
  preds_list=torch.zeros(0,dtype=torch.long, device='cpu')
  labels_list=torch.zeros(0,dtype=torch.long, device='cpu')
  totalLoss = 0
  model.to(device)
  with tqdm(dataloader, unit="batch" , position=0 , leave=True ) as tepoch:
    for batch in tepoch:
      # print(len(batch[0].keys()))
      images = batch["image"]
      labels =batch["dr_grade"]
      images = images.to(device,dtype=torch.float)
      labels = labels.to(device)

      preds=model(images)
      loss = lossfn(preds, labels)
      
      totalLoss += loss.item()
      loss.to(device)
      if train:
        loss.backward()
        optimizer.step()
      
      probabilties,preds =torch.max(preds , 1)

      preds_list=torch.cat([preds_list, preds.view(-1).cpu()])
      labels_list=torch.cat([labels_list, labels.view(-1).cpu()])

      del preds , labels

      preds_numpy = preds_list.numpy()
      labels_numpy = labels_list.numpy()
      
      currentKappa, currentAccuracy = getMetrics(preds_numpy,labels_numpy)
     

      tepoch.set_postfix(loss=totalLoss/len(preds_numpy), mode = mode +" accuracy ={}".format(currentAccuracy) ,kappa =  currentKappa)
  return {"totalLoss":totalLoss/len(preds_numpy),
          "currentKappa":currentKappa,
          "currentAccuracy":currentAccuracy,
          "confusionMatrix":confusion_matrix(labels_numpy,preds_numpy)}
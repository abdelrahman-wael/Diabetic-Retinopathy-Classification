def runModel(model , optimizer , dataloader   , device ,train):

  if train:
    model.train()
  else:
    model.eval()

  lossfn = nn.CrossEntropyLoss() 

  epochPreds = []
  epochGT = []

  with tqdm(dataloader, unit="batch" , position=0 , leave=True) as tepoch:
    for batch in tepoch:
      images = batch["image"]
      labels =batch["dr_grade"]
      images = images.to(device)
      labels = labels.to(device)

      preds=model(images)
      loss = lossfn(preds, labels)
      
      if train:
        loss.backward()
        optimizer.step()

      tepoch.set_postfix(loss=loss.item())

  return pred,gt

from time import sleep


def cometML(Results,experiment,phase,epoch):
  experiment.log_metric("{}_loss".format(phase), Results["totalLoss"], step=epoch)
  experiment.log_metric("{}_accuracy".format(phase), Results["currentAccuracy"], step=epoch)
  if phase == "validation":
    experiment.log_metric("validation_QWK", Results["currentKappa"], step=epoch)
    experiment.log_confusion_matrix(matrix=Results["confusionMatrix"],                                     
                                    title = "Confusion Matrix, Epoch #%d" %(epoch),
                                    file_name = "confusion-matrix-%03d.json" %(epoch),
                                    max_examples_per_cell = 10000, step=epoch)

  

def trainModel(model , optimizer , trainDataloader , validDataloader , device ,experiment, epochs = 20):
  model.to(device)
  for epoch in range(epochs):
    trainingMetrics = runModel(model , optimizer , trainDataloader , device ,train = True)
    # this prevent tqdm from falling into a deadlock
    cometML(trainingMetrics ,experiment , "training" , epoch )
    sleep(1)
    validMetrics = runModel(model , optimizer , validDataloader , device ,train = False)
    cometML(validMetrics ,experiment , "validation" , epoch )
    
    
    # print validation metric

    # cometml

    # save best model

    print("================finished epoch {}===============".format(epoch))
    sleep(1)




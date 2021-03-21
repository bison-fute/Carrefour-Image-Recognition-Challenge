import torch
import numpy as np
import time
import csv


def get_validation_loss(test_loader, model, criterion):
    validation_loss = []
    for i, (x, y) in enumerate(test_loader):
        print("batch test ... ", i)
        print(x.get_device(), next(model.parameters()).is_cuda)
        with torch.no_grad():
            output = model(x)
        loss = criterion(output, y)
        validation_loss.append(loss)
    validation_loss_avg = np.average(validation_loss)
    return validation_loss_avg


def print_save_accuracy(train_loader, test_loader, model, accuracy_score, device, train=True, test=True):
    if train:
        prediction_train = []
        target_train = []
        for i, (x, y) in enumerate(train_loader):
            if i>0:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = model(x)
            softmax = torch.exp(output).cpu()
            prob = list(softmax.numpy())
            predictions = np.argmax(prob, axis=1)
            prediction_train.append(predictions)
            target_train.append(y)
        # validation accuracy
        accuracy_val = []
        for i in range(len(prediction_train)):
            accuracy_val.append(accuracy_score(target_train[i], prediction_train[i]))
        print('train accuracy: \t', np.average(accuracy_val))

    if test:
        prediction_val = []
        target_val = []
        for i, (x, y) in enumerate(test_loader):
            if i>0:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = model(x)
            softmax = torch.exp(output).cpu()
            prob = list(softmax.numpy())
            predictions = np.argmax(prob, axis=1)
            prediction_val.append(predictions)
            target_val.append(y)
        # validation accuracy
        accuracy_val = []
        for i in range(len(prediction_val)):
            accuracy_val.append(accuracy_score(target_val[i], prediction_val[i]))
        print('test accuracy: \t', np.average(accuracy_val))

    dict_accuracy = {"train_acc":np.average(accuracy_val),"test_acc":np.average(accuracy_val)}
    with open('accuracy_{}.csv'.format(int(time.time())), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_accuracy.items():
            writer.writerow([key, value])

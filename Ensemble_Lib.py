#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from keras.models import load_model

def ModelsPrediction(X_test, model_dict, num_NN):
    # model prediction for each test set image
    prediction = []
    for i in range(num_NN):
        prediction.append(model_dict[i].predict(X_test))
    return prediction




def EnsemblePrediction(X_test, prediction, num_NN):
    # ensemble prediction obtained by summing the probabilities of belonging to each class associated by each network 
    # with respect to each image of the test set
    ensemble_prediction = {}
    for imagine in range(X_test.shape[0]):
        imagine_pred = np.zeros(10)
        for i in range(num_NN):
            imagine_pred += prediction[i][imagine]
        ensemble_prediction[imagine] = imagine_pred
    return ensemble_prediction




def SingleModelAccuracy(X_test, y_test, prediction, num_NN):
    models_accuracy = np.zeros(num_NN)
    for n in range(num_NN):
        correct = 0
        for imagine in range(X_test.shape[0]):
            if (prediction[n][imagine].argmax() == y_test[imagine]):
                correct += 1
        models_accuracy[n] = correct / X_test.shape[0]
    return models_accuracy.mean(), models_accuracy.std()





def EnsembleAccuracy(X_test, y_test, ensemble_prediction):
    correct = 0
    for imagine in range(X_test.shape[0]):
        if (ensemble_prediction[imagine].argmax() == y_test[imagine]):
            correct += 1
    ensemble_accuracy = correct / X_test.shape[0]
    return ensemble_accuracy





def DegreeOfDisagreement(X_test, prediction, num_NN):
    # I calculate the degree of disagreement (DoD) between networks by comparing the predictions of pairs of networks and counting on how 
    # many predictions are in disagreement.
    # In error I take into account how many times an image is found to be classified in a discordant way
    
    DoD = np.zeros((num_NN,num_NN))
    error = {}
    for imagine in range(X_test.shape[0]):
        error[imagine] = 0
        
    for i in range(num_NN):
        for j in range(i+1,num_NN):
            for imagine in range(X_test.shape[0]):
                if prediction[i][imagine].argmax() != prediction[j][imagine].argmax():
                    DoD[i][j] += 1
                    error[imagine] += 1
    DoD = DoD/X_test.shape[0]
    return DoD, error




def DoD_lista(DoD, num_NN):
    DoD_list = []
    for i in range(num_NN):
        for j in range(i+1,num_NN):
            DoD_list.append(DoD[i][j])
    return DoD_list


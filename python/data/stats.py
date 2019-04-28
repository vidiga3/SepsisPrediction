import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt

""" Model for storing mlModel with label """
class Model:
    def __init__(self,model,label):
        self.model = model
        self.label = label
        
    def __str__(self):
        return self.label

def rocCurves(testResultsL,modelLabelsL):
    plt.figure(figsize=(8, 6))
    for i,test_results in enumerate(testResultsL):
        y_test,y_pred,y_prob = zip(*test_results)
        model = modelLabelsL[i]
        logit_roc_auc = roc_auc_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label= model + ' (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC.jpg')
    plt.show()

def statScores(y_test,y_pred,y_prob,modelLabel):
    print('Accuracy of classifier on '+ modelLabel + 'test set: {:.2f}'.format(accuracy_score(y_pred, y_test)))
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='Model (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('*********************************************')


def func(X_train, y_train,X_test,y_test,mlModel):
    print('*******************',mlModel,'***************')
    model = mlModel.model
    model.fit(X_train, y_train)
    if('xgboost'==mlModel.label):
        plot_importance(model,max_num_features=15)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
#     statScores(y_test,y_pred,y_prob,mlModel.label)
#     print('output',zip(y_test,y_pred,y_prob))
    return y_test,y_pred,y_prob
    
def splitNRunModel(useDF,mlModels):
#     print('cols:',useDF.columns)
    print('.......splitting.......')
    train, test = train_test_split(useDF, test_size=0.3)
    X_train,y_train = train.loc[:, train.columns != 'angus'],train.angus
    X_test,y_test = test.loc[:, test.columns != 'angus'],test.angus
    output = []
    for mlModel in mlModels:
        output.append(func(X_train, y_train,X_test,y_test,mlModel))
    print('......... done .........')

def runNewModel(train,test,mlModels):
    X_train,y_train = train.loc[:, train.columns != 'angus'],train.angus
    X_test,y_test = test.loc[:, test.columns != 'angus'],test.angus
    output = []
    for mlModel in mlModels:
        output.append(func(X_train, y_train,X_test,y_test,mlModel))
    return output

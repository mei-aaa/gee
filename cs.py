# coding=utf-8

import pandas as pd
import json
import scipy.stats
import random
import numpy as np
import matplotlib.pyplot as plt

train_people_file='/home/mei/Documents/gt-kaggle/train_data/train_people/feature_355808/part-00000'
train_robot_file='/home/mei/Documents/gt-kaggle/train_data/train_robot/feature_93900/part-00000'
test_people_file='/home/mei/Documents/gt-kaggle/test_data/test_people/feature_43955/part-00000'
test_robot_c_file='/home/mei/Documents/gt-kaggle/test_data/test_robot_c/feature_37702/part-00000'
test_robot_cl_file='/home/mei/Documents/gt-kaggle/test_data/test_robot_Cl/feature_17139/part-00000'

##1 Read data
def readdata(filename):
    with open(filename) as f:
        data=[]
        for line in f.readlines():
            data.append(json.loads(line))
        data=pd.DataFrame(data)
    return data
train_people=readdata(train_people_file)
train_robot=readdata(train_robot_file)
test_people=readdata(test_people_file)
test_robot_c=readdata(test_robot_c_file)
test_robot_cl=readdata(test_robot_cl_file)

##1.2 train data x ;train data y :people 1 ,robot 0
train_y=np.concatenate((np.ones((train_people.shape[0],1)),np.zeros((train_robot.shape[0],1))))
train_x=train_people.append(train_robot,ignore_index=True)
train_y=pd.DataFrame(train_y)
train_y.columns=['label']
train_data=pd.concat([train_x,train_y],axis=1)

## test data x; test data y
test_x=test_people.append(test_robot_c,ignore_index=True).append(test_robot_cl,ignore_index=True)
test_y=np.concatenate((np.ones((test_people.shape[0],1)),
                       np.zeros((test_robot_c.shape[0]+test_robot_cl.shape[0],1))))
test_y=pd.DataFrame(test_y)
test_y.columns=['label']


###undersampling
sam_train_people=train_people.sample(frac=0.25,replace=True,random_state=111)
sam_train_x=sam_train_people.append(train_robot,ignore_index=True)
sam_train_y=np.concatenate((np.ones((sam_train_people.shape[0],1)),np.zeros((train_robot.shape[0],1))))
sam_train_y=pd.DataFrame(sam_train_y)


###SMOTE
'''
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
train_x_res, train_y_res = sm.fit_sample(train_x,train_y)
'''

##1.3 data distribution
plt.figure(figsize=(18,12))
KL = np.zeros((train_people.shape[1]))
for i in range(train_people.shape[1]):
    peo_p = np.histogram(sam_train_people[i],50)
    rob_p = np.histogram(train_robot[i],peo_p[1])
    peo = peo_p[0]/peo_p[0].sum()
    rob = rob_p[0]/rob_p[0].sum() + 1e-12
    KL[i] = scipy.stats.entropy(peo,rob)
    plt.subplot(4, 4, i+1)
    plt.title("KL: %.3f" % KL[i])
    plt.hist(sam_train_people[i], peo_p[1], color='green', alpha=0.5)
    plt.hist(train_robot[i], peo_p[1], color='red', alpha=0.5)
plt.show()

'''
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(train_x)
Xs_new=pca.transform(train_x)
p1 = plt.scatter(Xs_new[0:train_people.shape[0],1], \
    Xs_new[0:train_people.shape[0]:,0], marker = '.', color = 'g',alpha=0.3)
p2 = plt.scatter(Xs_new[train_people.shape[0]:train_x.shape[0],1], \
    Xs_new[train_people.shape[0]:train_x.shape[0],0], marker = '.', color = 'r',alpha=0.3)


people=train_people.sample(frac=0.003,replace=False)
robot=train_robot.sample(frac=0.01,replace=False)
np.vstack((people,robot))

from sklearn.manifold import TSNE
tsne=TSNE(n_components=3)
f_tsne=tsne.fit_transform(np.vstack((people,robot)))
p_tsne=f_tsne[:1067]
r_tsne=f_tsne[1067:]
p1 = plt.scatter(p_tsne[:,0],p_tsne[:,1], marker = '.', color = 'g',alpha=0.3)
p2 = plt.scatter(r_tsne[:,0], r_tsne[:,1], marker = '.', color = 'r',alpha=0.3)
plt.show()
'''


from sklearn.ensemble import RandomForestClassifier  
from sklearn.grid_search import GridSearchCV  
from sklearn import cross_validation, metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression 
import xgboost as xgb

###LR
LR = LogisticRegression()
LR.fit(train_x, train_y.values.ravel())
train_y_predprob=LR.predict_proba(train_x)[:,1]
train_y_pred=LR.predict(train_x)
test_y_pred = LR.predict(test_x)
test_y_predprob = LR.predict_proba(test_x)[:,1]
print("LR Accuracy (Train): %3f" %metrics.accuracy_score(train_y.values,train_y_pred))
print("LR AUC Score(Train): %3f" %metrics.roc_auc_score(train_y,train_y_predprob))
print("LR Accuracy Score(Test): %3f" %metrics.accuracy_score(test_y.values,test_y_pred))
print("LR AUC Score(Test): %3f" %metrics.roc_auc_score(test_y,test_y_predprob))
print(confusion_matrix(test_y,LR.predict(test_x)))
print(classification_report(test_y,LR.predict(test_x)))

def LR_search(clf,param,trainx,trainy,testx,testy):
    LR_search=GridSearchCV(clf,param_grid=param,cv=5,scoring='roc_auc')
    LR_search.fit(trainx,trainy.values.ravel())
    print(LR_search.best_params_, LR_search.best_score_)
    train_y_pred = LR_search.predict(trainx)
    train_y_predprob = LR_search.predict_proba(trainx)[:,1]
    test_y_pred = LR_search.predict(testx)
    test_y_predprob = LR_search.predict_proba(testx)[:,1]
    print("LR Accuracy (Train): %3f" %metrics.accuracy_score(trainy.values,train_y_pred))
    print("LR AUC Score(Train): %3f" %metrics.roc_auc_score(trainy,train_y_predprob))
    print("LR Accuracy Score(Test): %3f" %metrics.accuracy_score(testy.values,test_y_pred))
    print("LR AUC Score(Test): %3f" %metrics.roc_auc_score(testy,test_y_predprob))
    print(confusion_matrix(testy,LR_search.predict(testx)))
    print(classification_report(testy,LR_search.predict(testx)))

param_lr1={'C':[0.1,0.2,0.5,1,2,5,10],'class_weight':['balanced', None]}
LR_search(clf=LR,param=param_lr1,trainx=train_x,trainy=train_y,testx=test_x,testy=test_y)

LR_search(clf=LR,param=param_lr1,trainx=sam_train_x,trainy=sam_train_y,testx=test_x,testy=test_y)

###RF              
clf=RandomForestClassifier(random_state=10)
param_test={'n_estimators':[30],
            'max_depth':np.arange(3,13,3),
            'max_features':np.arange(3,10,2),
            'min_samples_split':np.arange(10,31,5)}

def RF_search(clf,param,trainx,trainy,testx,testy):
    RF_search=GridSearchCV(clf,param_grid=param,cv=5,scoring='roc_auc')
    RF_search.fit(trainx,trainy.values.ravel())
    print(RF_search.best_params_, RF_search.best_score_)
    train_y_pred = RF_search.predict(trainx)
    train_y_predprob = RF_search.predict_proba(trainx)[:,1]
    test_y_pred = RF_search.predict(testx)
    test_y_predprob = RF_search.predict_proba(testx)[:,1]
    print("RF Accuracy (Train): %3f" %metrics.accuracy_score(trainy.values,train_y_pred))
    print("RF AUC Score(Train): %3f" %metrics.roc_auc_score(trainy,train_y_predprob))
    print("RF Accuracy Score(Test): %3f" %metrics.accuracy_score(testy.values,test_y_pred))
    print("RF AUC Score(Test): %3f" %metrics.roc_auc_score(testy,test_y_predprob))
    print(confusion_matrix(testy,RF_search.predict(testx)))
    print(classification_report(testy,RF_search.predict(testx)))
'''
param_test={'n_estimators':[30],
            'max_depth':np.arange(3,13,3),
            'max_features':np.arange(3,10,2),
            'min_samples_split':np.arange(10,31,5)}

'''
param_test={'n_estimators':[30],
            'max_depth':[12],
            'max_features':[9],
            'min_samples_split':[10]}
RF_search(clf=RandomForestClassifier(random_state=10),param=param_test,trainx=train_x,trainy=train_y,testx=test_x,testy=test_y)

RF_search(clf,param=param_test,trainx=sam_train_x,trainy=sam_train_y,testx=test_x,testy=test_y)

###xgboost
def Xgb_search(clf,param,trainx,trainy,testx,testy):
    xgb_search=GridSearchCV(clf,param_grid=param,cv=5,scoring='roc_auc')
    xgb_search.fit(trainx,trainy.values.ravel())
    print(xgb_search.best_params_, xgb_search.best_score_)
    train_y_pred = xgb_search.predict(trainx)
    train_y_predprob = xgb_search.predict_proba(trainx)[:,1]
    test_y_pred = xgb_search.predict(testx)
    test_y_predprob = xgb_search.predict_proba(testx)[:,1]
    print("Xgboost Accuracy (Train): %3f" %metrics.accuracy_score(trainy.values,train_y_pred))
    print("Xgboost AUC Score(Train): %3f" %metrics.roc_auc_score(trainy,train_y_predprob))
    print("Xgboost Accuracy Score(Test): %3f" %metrics.accuracy_score(testy.values,test_y_pred))
    print("Xgboost AUC Score(Test): %3f" %metrics.roc_auc_score(testy,test_y_predprob))
    print(confusion_matrix(testy,xgb_search.predict(testx)))
    print(classification_report(testy,xgb_search.predict(testx)))


xgb_model=xgb.XGBClassifier()
xgb_param={'learning_rate':[0.01,0.1,0.2],
           'max_depth':[3,5,7,9],
           'n_estimators':[100]}

Xgb_search(xgb_model,param=xgb_param,trainx=train_x,trainy=train_y,testx=test_x,testy=test_y)
Xgb_search(xgb_model,param=xgb_param,trainx=sam_train_x,trainy=sam_train_y,testx=test_x,testy=test_y)
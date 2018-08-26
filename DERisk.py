# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 12:25:04 2018

@author:  Kalyani Jandhyala
"""

import csv
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix , coo_matrix, hstack
import scipy
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm ,grid_search
from sklearn.svm import LinearSVC , SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score,accuracy_score, classification_report, f1_score ,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier , NearestCentroid
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import FeatureUnion, Parallel, Pipeline, make_union, make_pipeline
import warnings

def main():  
    warnings.simplefilter('ignore')
#reading required csv's #####:
    train_data = pd.read_csv("file:///E:/Rkmveri/SEMISTER 4/Depression_CLEF/agg_train.csv")  ##train data
    test_data =  pd.read_csv("file:///E:/Rkmveri/SEMISTER 4/Depression_CLEF/agg_test_new.csv")   ## test data
######################################
    train = list(train_data['Text'])
    test = list(test_data['Text'])
    y_train = list(train_data['Label'])
    tst_id = pd.DataFrame(test_data['ID'])
###########tf-idf:
    vectorizer = TfidfVectorizer(min_df =1,stop_words='english',use_idf=True,analyzer='word',ngram_range=(1,1),max_features=15000)
    x_train = vectorizer.fit_transform(train)
    x_test  = vectorizer.transform(test)
    print(" 1.SVM \n 2.Naive-base \n 3.Logistic Regression \n 4.multilayer perceptron\n 5.Random Forest \n 6.Knn \n 7.aDboost")
    choice = input(" Enter your Choice : ")
    usr_choice(choice,x_train,x_test,y_train,tst_id) # for only rawvectorized  data
    return
def OUTPUT(predicted,tst_id):
        
    file = open("E:/Rkmveri/SEMISTER 4/Preedicted_output.txt",'w')
    for i in range(0,820):
        file.write(str(tst_id['ID'].values[i])+"\t\t"+str(predicted[0].values[i])+'\n')
    file.close()
    return ; 
 
def GridSearch(x_train,x_test,y_train,tst_id,parameters,clf): #for only raw vectorized data

    pipeline2 = Pipeline([   
       # ('feature_selection', SelectKBest(chi2,k=1000)),    
        ('clf', clf),
    ])
    grid = grid_search.GridSearchCV(pipeline2,parameters,cv=10)          
    grid.fit(x_train,y_train)    #for only raw  vectorized data
    clf = grid.best_estimator_                   # Best grid
    print('\n The best grid is as follows: \n')
    print(grid.best_estimator_)
    # Classification of the test samples
    predicted = clf.predict(x_test)  #for only raw vectorized data   
    predicted = pd.DataFrame(predicted)
    predicted[predicted[0]==0]=2 #  changing all 0 values to 2 
    
    OUTPUT(predicted,tst_id)
    return;

def usr_choice(choice,x_train,x_test,y_train,tst_id): ## for raw vectorized text data,,
    if choice=="1":
        print("SVM is running...")
        clf = svm.LinearSVC(class_weight='balanced',verbose=0, random_state=None,max_iter=1000)  
        parameters = {
        'clf__C':(0.99,0.98,0.93,0.999,0.91,1,10,2.5,5,8,0.001,0.1),
        }
    elif choice=="2":  
        print("NaiveBayes is running...")
        clf = BernoulliNB(fit_prior=None)  
        parameters = {
       'clf__alpha':[1],
        }       
    elif choice=="3":
        print("Logistic Regression is running...")
        clf = LogisticRegression(solver='liblinear',class_weight='balanced',random_state=5,tol=0.001,max_iter=1000) 
        parameters = {
        'clf__C':(10,10.2,9.99,9.95,10.9,10.6),
        }
    elif choice=="4":
        print("MLP is running...")
        clf = MLPClassifier(alpha=0.0001,hidden_layer_sizes=(5, 2), random_state=1)
        parameters = {
        'clf__solver':('lbfgs','sgd','adam'),        
        'clf__activation':('identity', 'logistic', 'tanh', 'relu'),
        'clf__random_state':(0,1,4,7,9,10),
        }
    elif choice=="5":
        print("Random Forest is running...")
        clf = RandomForestClassifier(criterion='entropy',max_features=None,class_weight='balanced')
        parameters = {
        'clf__n_estimators':(10,60,50,70,100),
        }
    elif choice=="6":
        print("KNeighborsClassifier is running...")
        clf = KNeighborsClassifier( algorithm='brute')
        parameters = {
            'clf__n_neighbors':(3,10,13,100),
        }
    elif choice=="7":
        print("AdaBoostClassifier is running...")
        clf = AdaBoostClassifier(algorithm='SAMME.R',learning_rate=0.01)
        parameters = {
        'clf__n_estimators':(5,1,10,20,50,70,100),
#        'clf__max_depth':[20],
#        'clf__min_samples_split':[761],
        }
    
        
    else:
        print("Wrong choice. Try Again Later")
    GridSearch(x_train,x_test,y_train,tst_id, parameters, clf) #for only raw vectorized data
    return;
if __name__ == "__main__":
    main()
    
    
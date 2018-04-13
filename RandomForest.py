# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:29:25 2018

@author: JimSmit

codes to share @ GitHub

RandomForest - automated Alzheimer's disease diagnosis / prognosis based on MRI-scan derived features
"""

#Import useful libraries
import numpy as np
import pandas as pd 
import sys
import time


#Import data file, preferably Excel document 
X = pd.read_excel('file_name.xlsx')


#normalize volume features by Intracranial volumes (ICV)
#specify a and b for start and end of columns filled with volume-related features
X.ix[a,b]=X.iloc[:,a:b].div(X.ix[:,c], axis=0)

#log scale white matter lesions
#specify a and b for start and end of columns filled with WML features
X.ix[:,a:b] = np.log(X.iloc[:,a:b])

#drop irrelevant columns
#specify cols for the column numbers to be removed
X.drop(X.columns[cols],axis=1,inplace=True)

#define label vector 
y = X['name_label_column']

#drop label column from feature matrix
X = X.drop('name_label_column',1)

#solve for non-available measurements in feature matrix
#unknown value replaced by mean of the distribution
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X=imp.transform(X)


            
#define function for classification with RandomForest 
 
def Randomforest(features, classes):
    #import libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import roc_auc_score
    import imblearn
    from imblearn.over_sampling import SMOTE, ADASYN
    from scipy import stats
    from sklearn import svm
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold

   
    #define empty arrays for results
    acc= []
    AUC = []
    
    # Optional: if using RFE for feature selection
    #define empty arrays for number of features and column numbers of selected features
    n_features=[]
    selected_features = np.zeros((len(X[1,:])))

    
    #define k for number of iterations (cross validation
    k = 50
    #start K-fold loop     
    for i in range(0,k):
        print([i],) # To print every iteration to make progress durig running visible
        sys.stdout.flush() # To print the previous line on the screen immeidately. Without this, it is stored in a buffer and printed later.
        
        # Train-test split, percentage of test group tuned by 'test_size'
        # random-state=i makes sure every iteration used a unique subset as testing group
        X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.1, random_state=i)
        
        # Optional for unbalanced classes: resample training set by SMOTE
        # Make sure that amount of subjects in all classes are equal, makes use of synthetic subjects
        X_train_r, y_train_r = SMOTE().fit_sample(X_train, y_train)

        

        # Feature selection: Do t test for p < 0.05

        #name the 2 classes in the training set
        class_1 = X_train[y_train == 1]
        class_2 = X_train[y_train == 0]
        
        h,p = stats.ttest_ind( class_1,class_2,equal_var = False,nan_policy='omit')     
        treshold = p < 0.05   # set treshold for P < 0.05
        p[treshold] = 0  # All low values set to 0
        mask = p == 0    # define mask
        X_train = X_train[:,mask]  
        X_test = X_test[:,mask]
              
        #standarization of training and testing data
        X_train_scaled = preprocessing.scale(X_train)
        X_test_scaled = preprocessing.scale(X_test)
                 
        # Optional: 
        # Feature selection: Using Recursive feature extraction (RFE)    
        svc = SVC(kernel="linear") # selects classifier to provide information about feature importance
        
        selector = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10)) # set umber of features to remove at each iteration, set number of iterations in corss validation
        
        selector = selector.fit(X_train_scaled, y_train) #fit the RFE to the training set
        
        X_train_FS = selector.transform(X_train_scaled) #extract selected features from the training set
        X_test_FS = X_test_scaled[:,selector.support_]  #extract selected features from the testing set, according to outcome of RFE executed in the training set
        n_features.append(selector.n_features_)       #fill in number of selected features per iteration 
        selected_features[mask] = selected_features[mask]+selector.support_ # fill in which features are selected per iteration
    
        # RandomForest classification
        t=100 # define number of trees
        clf = RandomForestClassifier(n_estimators=t) # define classifier
        clf = clf.fit(X_train_scaled, y_train) # fit classifier to training and testing data
        score = clf.score(X_test_scaled, y_test) # define accuracy score
        acc.append(score)  # fill in accuracy to accuracy-array 
        score_AUC = clf.predict_proba(X_test_scaled) # define Area under the ROC curve (AUC) score
        score_AUC = score_AUC[:,1]
        ROC_AUC = roc_auc_score(y_test, score_AUC)
        AUC.append(ROC_AUC)  #  fill in AUC to AUC-array
        
    # Print statements    

    print('accuracies: \n',acc)
    print('accuracies by a k fold CV Random Forest: '+ str(np.mean(acc)) + ' ( std : ' + str(np.std(acc)) + ' )' )
    print('AUCs: \n',AUC)
    print('AUC by a k fold CV Random Forest: '+ str(np.mean(AUC)) + ' ( std : ' + str(np.std(AUC)) + ' )' )
    
    # Define dataframe with performance scores
    df = pd.DataFrame({'acc_coarse': scores,'AUC_coarse': AUC})
    
    # export outcomes to a csv_file on local map               
    df.to_csv('name_file.csv', encoding='utf-8', index=False)

 
# Run the function with feature matrix (X) and label vector (y)
Randomforest (X, y)

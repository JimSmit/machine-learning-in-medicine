# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:32:35 2018

@author: JimSmit

codes to share @ GitHub

Support Vector Machine - automated Alzheimer's disease diagnosis / prognosis based on MRI-scan derived features

"""

#Import useful libraries
import numpy as np
import pandas as pd 
import sys

#Import data file, preferably Excel document 
X = pd.read_excel('file_name.xlsx')


#normalize volume-based features by Intracranial volumes (ICV)
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


# define function for SVM classification, optimizing hyperparameters (C and gamma) by a double Gridsearch

def Classification_comparator_linearSVM (features, classes):
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.svm import SVC
    import math
    from scipy import stats
    from sklearn import preprocessing
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import roc_auc_score
    import imblearn
    from imblearn.over_sampling import SMOTE, ADASYN


    # Define range for the C value Gridsearch (logaritic range recommented)
    C_range_coarse= np.logspace(-5,5,10,base=2)
    
    # Define empty array for selected C values
    C_values_coarse =[]
    
    # Optional: Define range for the gamma value Gridsearch (only when using Gaussian kernels)
    gamma_range_coarse= np.logspace(-5,5,10,base=2)
    
    # Define empty array for selected gamma values
    gamma_values_coarse =[]
    
    # Define empty arrays for performance measures in coarse gridsearch
    acc_coarse = []
    AUC_coarse = []
    
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
        # strtification by classes to make sure train and test data contain same ratio of classes
        X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.1, stratify=classes)
        
        # Optional for unbalanced classes: resample training set by SMOTE
        # Make sure that amount of subjects in all classes are equal, makes use of synthetic subjects
        X_train, y_train = SMOTE().fit_sample(X_train, y_train)
               
        # Feature selection: Do t test for p < 0.05       
        class_1 = X_train[y_train == 1] # name the 2 classes in the training set
        class_2 = X_train[y_train == 0]
        
        h,p = stats.ttest_ind( class_1,class_2,equal_var = False,nan_policy='omit')     
        treshold = p < 0.05   # set treshold for P < 0.05
        p[treshold] = 0  # All low values set to 0
        mask = p == 0    # define mask
        X_train = X_train[:,mask]  
        X_test = X_test[:,mask]
              
        #standarization of training and testing data
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
        
        # Optional: 
        # Feature selection: Using Recursive feature extraction (RFE)    
        svc = SVC(kernel="linear") # selects classifier to provide information about feature importance
        
        selector = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10)) # set umber of features to remove at each iteration, set number of iterations in corss validation
        
        selector = selector.fit(X_train, y_train) #fit the RFE to the training set
        
        X_train = selector.transform(X_train) #extract selected features from the training set
        X_test = X_test[:,selector.support_]  #extract selected features from the testing set, according to outcome of RFE executed in the training set
        n_features.append(selector.n_features_)       #fill in number of selected features per iteration 
        selected_features[mask] = selected_features[mask]+selector.support_ # fill in which features are selected per iteration
    
        #find optimal C_value and train classifier with optimal C_value
        svr = svm.SVC(kernel='kernel', class_weight = 'balanced') # Define classification algorithm, kernel: choose from: 
                                                                  # {linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, class-weight balanced implies weighted SVMs
        param_grid = {'C':C_range_coarse,
                     'gamma':gamma_range_coarse}  #Define grid for parameters
            
        n = 10 # Define number of iterations for inner loop: gridsearch     
        clf = GridSearchCV(svr, param_grid, cv=10, scoring='roc_auc') # Define classifier, cross-validated gridsearch
    
        clf.fit(X_train, y_train)   # By default also trains classifier on entire train set with optimal C_value (refit option)
        C_values_coarse.append(clf.best_params_['C']) # Fill array with chosen C values
        gamma_values_coarse.append(clf.best_params_['gamma']) # Fill array with chosen gamma values
      
        # apply classifier to test set
        score_overall = clf.score(X_test, y_test)    # define accuracy
        acc_coarse.append(score_overall)  # fill array with accuracies
        score_AUC = clf.predict(X_test) # define area under the ROC Curve (AUC)
        ROC_AUC = roc_auc_score(y_test, score_AUC)
        AUC_coarse.append(ROC_AUC) # fill array with AUCs
       
    # print statements
  
    print('selected features by t test: ', mask)      
    print('chosen C values in coarse gridsearch: \n',C_values_coarse)
    print('chosen gamma values in coarse gridsearch: \n',gamma_values_coarse)
    import collections
    counter=collections.Counter(C_values_coarse)
    print('count of chosen C values in coarse gridsearch: \n',counter)
    counter=collections.Counter(gamma_values_coarse)
    print('count of chosen gamma values in coarse gridsearch: \n',counter)
    print('accuracies in coarse gridsearch: \n',acc_coarse)
    print('Mean accuracy in coarse gridsearch: '+ str(np.mean(acc_coarse)) + ' ( std : ' + str(np.std(acc_coarse)) + ' )' )
    print('AUCs in coarse gridsearch: \n',AUC_coarse)
    print('Mean AUC in coarse gridsearch: '+ str(np.mean(AUC_coarse)) + ' ( std : ' + str(np.std(AUC_coarse)) + ' )' ) 
    
# Codes for 2nd, fine gridsearch    
    
    # Define empty array for selected C values
    C_values_fine =[]
    
    # Define empty array for selected gamma values
    gamma_values_fine =[]
    
    # Define empty arrays for performance measures in coarse gridsearch
    acc_fine = []
    AUC_fine = []

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
        
        # create fine range, centred around the best found parameter in the first grodsearch
        a = C_values_coarse[i]
        C_range_fine = np.logspace((np.log2(a))-1,(np.log2(a))+1,16,base=2)
        b = gamma_values_coarse[i]
        gamma_range_fine = np.logspace((np.log2(b))-1,(np.log2(b))+1,16,base=2)

        # Train-test split, percentage of test group tuned by 'test_size'
        # random-state=i makes sure every iteration used a unique subset as testing group
        # strtification by classes to make sure train and test data contain same ratio of classes
        X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.1, stratify=classes)
        
        # Optional for unbalanced classes: resample training set by SMOTE
        # Make sure that amount of subjects in all classes are equal, makes use of synthetic subjects
        X_train, y_train = SMOTE().fit_sample(X_train, y_train)
               
        # Feature selection: Do t test for p < 0.05       
        class_1 = X_train[y_train == 1] # name the 2 classes in the training set
        class_2 = X_train[y_train == 0]
        
        h,p = stats.ttest_ind( class_1,class_2,equal_var = False,nan_policy='omit')     
        treshold = p < 0.05   # set treshold for P < 0.05
        p[treshold] = 0  # All low values set to 0
        mask = p == 0    # define mask
        X_train = X_train[:,mask]  
        X_test = X_test[:,mask]
              
        #standarization of training and testing data
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
        
        # Optional: 
        # Feature selection: Using Recursive feature extraction (RFE)    
        svc = SVC(kernel="linear") # selects classifier to provide information about feature importance
        
        selector = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10)) # set umber of features to remove at each iteration, set number of iterations in corss validation
        
        selector = selector.fit(X_train, y_train) #fit the RFE to the training set
        
        X_train = selector.transform(X_train) #extract selected features from the training set
        X_test = X_test[:,selector.support_]  #extract selected features from the testing set, according to outcome of RFE executed in the training set
        n_features.append(selector.n_features_)       #fill in number of selected features per iteration 
        selected_features[mask] = selected_features[mask]+selector.support_ # fill in which features are selected per iteration
        
        
        #find optimal C_value and train classifier with optimal C_value              
        svr = svm.SVC(kernel='kernel', class_weight = 'balanced')   # Define classification algorithm, kernel: choose from: 
                                                                  # {linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, class-weight balanced implies weighted SVMs    
        param_grid = {'C':C_range_fine,
                     'gamma':gamma_range_fine} # define grid for fine gridsearch
                     
        n = 10 # Define number of iterations for inner loop: gridsearch     
        clf = GridSearchCV(svr, param_grid, cv=n, scoring='roc_auc') # Define classifier, cross-validated gridsearch
    
        clf.fit(X_train, y_train)   # By default also trains classifier on entire train set with optimal C_value (refit option)
        C_values_fine.append(clf.best_params_['C']) # Fill array with chosen C values
        gamma_values_fine.append(clf.best_params_['gamma']) # Fill array with chosen gamma values
      
        # apply classifier to test set
        score_overall = clf.score(X_test, y_test)    # define accuracy
        acc_fine.append(score_overall)  # fill array with accuracies
        score_AUC = clf.predict(X_test) # define area under the ROC Curve (AUC)
        ROC_AUC = roc_auc_score(y_test, score_AUC)
        AUC_coarse.append(ROC_AUC) # fill array with AUCs
    
    # print statements  
    print('selected features by t test: ', mask)     
    print('chosen C values in fine gridsearch: \n',C_values_fine)
    print('chosen gamma values in fine gridsearch: \n',gamma_values_fine)
    import collections
    counter=collections.Counter(C_values_fine)
    print('count of chosen C values in fine gridsearch: \n',counter)
    counter=collections.Counter(gamma_values_fine)
    print('count of chosen gamma values in fine gridsearch: \n',counter)
    print('accuracies in fine gridsearch: \n',acc_fine)
    print('Mean accuracy in fine gridsearch: '+ str(np.mean(acc_fine)) + ' ( std : ' + str(np.std(acc_fine)) + ' )' )
    print('AUCs in fine gridsearch: \n',AUC_fine)
    print('Mean AUC in fine gridsearch: '+ str(np.mean(AUC_fine)) + ' ( std : ' + str(np.std(AUC_fine)) + ' )' )
    
    # Define dataframe with performance scores first and second gridsearch
    df = pd.DataFrame({'acc_coarse': acc_coarse,'AUC_coarse': AUC_coarse,'acc_fine':acc_fine,'AUC_fine': AUC_fine})
    
    # export outcomes to a csv_file on local map 
    df.to_csv('name_file.csv', encoding='utf-8', index=False)

# Run the function with feature matrix (X) and label vector (y)
Classification_comparator_linearSVM (X, y)    
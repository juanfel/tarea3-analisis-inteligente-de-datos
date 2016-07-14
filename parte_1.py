# -*- coding: utf-8 -*-
"""
Created on Sat Jul 09 12:30:55 2016

@author: gotba_000
"""


# Parte 1

# A)

import urllib
import pandas as pd
train_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.train"
test_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.test"
train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
train_df = pd.DataFrame.from_csv('train_data.csv',header=0,index_col=0)
test_df = pd.DataFrame.from_csv('test_data.csv',header=0,index_col=0)
train_df.head()
test_df.tail()


# B)

from sklearn.preprocessing import StandardScaler
X = train_df.ix[:,'x.1':'x.10'].values
y = train_df.ix[:,'y'].values
X_std = StandardScaler().fit_transform(X)

#y_test = test_df.ix[:,'y'].values

# C)

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
#import seaborn as sns
import numpy as np
sklearn_pca = PCA(n_components=2)
Xred_pca = sklearn_pca.fit_transform(X_std)
cmap = plt.cm.get_cmap('spectral')
mclasses=(1,2,3,4,5,6,7,8,9,10,11)
mcolors = [cmap(i) for i in np.linspace(0,1,11)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses,mcolors):
    plt.scatter(Xred_pca[y==lab, 0],Xred_pca[y==lab, 1],label=lab,c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.show()


# D)

from sklearn.lda import LDA
sklearn_lda = LDA(n_components=2)
Xred_lda = sklearn_lda.fit_transform(X_std,y)
cmap = plt.cm.get_cmap('spectral')
mclasses=(1,2,3,4,5,6,7,8,9,10,11)
mcolors = [cmap(i) for i in np.linspace(0,1,11)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses,mcolors):
    plt.scatter(Xred_lda[y==lab, 0],Xred_lda[y==lab, 1],label=lab,c=col)
plt.xlabel('LDA/Fisher Direction 1')
plt.ylabel('LDA/Fisher Direction 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.show()


# F)

clasificador = np.zeros(11)
for clase in range(0,11):
    clasificador[clase] = float(y.tolist().count(clase+1)) / y.size
    
    
# G)
    
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
Xtest = test_df.ix[:,'x.1':'x.10'].values
ytest = test_df.ix[:,'y'].values
X_std_test = StandardScaler().fit_transform(Xtest)

lda_model = LDA()
lda_model.fit(X_std,y)
print lda_model.score(X_std,y)
print lda_model.score(X_std_test,ytest)

qda_model = QDA()
qda_model.fit(X_std,y)
print qda_model.score(X_std,y)
print qda_model.score(X_std_test,ytest)

knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_std,y)
print knn_model.score(X_std,y)
print knn_model.score(X_std_test,ytest)

plt.figure(figsize=(12, 8))
train_scores = []
test_scores = []
for k in range(1,21):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_std,y)
    train_scores += [knn_model.score(X_std,y)]
    test_scores += [knn_model.score(X_std_test,ytest)]
plt.plot(range(1,21), train_scores, label="Train score")
plt.plot(range(1,21), test_scores, label="Test score")
plt.xlabel("Number of neighbors")
plt.ylabel("Score")
plt.legend(loc='upper right', fancybox=True)
plt.show()


# H)

lda_train_scores = []
lda_test_scores = []
qda_train_scores = []
qda_test_scores = []
knn_train_scores = []
knn_test_scores = []

for k in range(1,11):
    sklearn_pca = PCA(n_components=k)
    Xred_pca = sklearn_pca.fit_transform(X_std)
    Xred_pca_test = sklearn_pca.fit_transform(X_std_test)    
    #LDA
    lda_model = LDA()
    lda_model.fit(Xred_pca,y)
    lda_train_scores += [1-lda_model.score(Xred_pca,y)]
    lda_test_scores += [1-lda_model.score(Xred_pca_test,ytest)]
    #QDA
    qda_model = QDA()
    qda_model.fit(Xred_pca,y)
    qda_train_scores += [1-qda_model.score(Xred_pca,y)]
    qda_test_scores += [1-qda_model.score(Xred_pca_test,ytest)]
    #KNN
    knn_model = KNeighborsClassifier(n_neighbors=8)
    knn_model.fit(Xred_pca,y)
    knn_train_scores += [1-knn_model.score(Xred_pca,y)]
    knn_test_scores += [1-knn_model.score(Xred_pca_test,ytest)]

plt.figure(figsize=(12, 8))
plt.plot(range(1,11), lda_train_scores, label="LDA train score", c = "blue")
plt.plot(range(1,11), lda_test_scores, label="LDA test score", c = "cyan")
plt.plot(range(1,11), qda_train_scores, label="QDA train score", c = "yellow")
plt.plot(range(1,11), qda_test_scores, label="QDA test score", c = "green")
plt.plot(range(1,11), knn_train_scores, label="8-neighbors clas. train score", c = "red")
plt.plot(range(1,11), knn_test_scores, label="8-neighbors clas. test score", c = "magenta")
plt.xlabel("Dimension (reduced from PCA)")
plt.ylabel("Error")
plt.legend(loc='right', fancybox=True)
plt.show()


# I)

lda_train_scores = []
lda_test_scores = []
qda_train_scores = []
qda_test_scores = []
knn_train_scores = []
knn_test_scores = []

for k in range(1,11):
    sklearn_lda = LDA(n_components=k)
    Xred_lda = sklearn_lda.fit_transform(X_std,y)   
    Xred_lda_test = sklearn_lda.fit_transform(X_std_test,ytest)     
    #LDA
    lda_model = LDA()
    lda_model.fit(Xred_lda,y)
    lda_train_scores += [1-lda_model.score(Xred_lda,y)]
    lda_test_scores += [1-lda_model.score(Xred_lda_test,ytest)]
    #QDA
    qda_model = QDA()
    qda_model.fit(Xred_lda,y)
    qda_train_scores += [1-qda_model.score(Xred_lda,y)]
    qda_test_scores += [1-qda_model.score(Xred_lda_test,ytest)]
    #KNN
    knn_model = KNeighborsClassifier(n_neighbors=8)
    knn_model.fit(Xred_lda,y)
    knn_train_scores += [1-knn_model.score(Xred_lda,y)]
    knn_test_scores += [1-knn_model.score(Xred_lda_test,ytest)]

plt.figure(figsize=(12, 8))
plt.plot(range(1,11), lda_train_scores, label="LDA train score", c = "blue")
plt.plot(range(1,11), lda_test_scores, label="LDA test score", c = "cyan")
plt.plot(range(1,11), qda_train_scores, label="QDA train score", c = "yellow")
plt.plot(range(1,11), qda_test_scores, label="QDA test score", c = "green")
plt.plot(range(1,11), knn_train_scores, label="8-neighbors clas. train score", c = "red")
plt.plot(range(1,11), knn_test_scores, label="8-neighbors clas. test score", c = "magenta")
plt.xlabel("Dimension (reduced from LDA)")
plt.ylabel("Error")
plt.legend(loc='right', fancybox=True)
plt.show()
import numpy as np
from . import Base
from math import inf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean,canberra,chebyshev,minkowski
import math

class PAD(Base):      
    def __init__(self, Z, pseudoy, k, X=None, lr=0.1, iterations=2000, plot=False): # Z is initial predict_prob
        super().__init__(Z, X, lr, iterations, plot)
        self.k = k
        self.labels = pseudoy
        self.scores = self.get_scores(self.labels)
        #self.labels = np.ones(Z.shape[0])
        
    # =====================================
    def get_labels(self, scores, u, yu): #########
        labels = self.labels
        labels[u]= yu
        dist = []
        neigh = []
        for ele in self.X:
                #dist.append(euclidean(self.X[u],ele))
                #dist.append(canberra(self.X[u],ele))
                #dist.append(chebyshev(self.X[u],ele))
                dist.append(minkowski(self.X[u],ele))
                
        temp_score = np.argsort(dist) 
        #n = self.X.shape[0]
        #k = int(math.sqrt(n))
        k = self.k
        n_id = np.setdiff1d(temp_score[:k], self.queried_ids) # ids for unqueried neighbours
        for i in n_id:
            if (yu == 1 and scores[i] >= 0.5) or (yu == -1 and scores[i] < 0.5): # or (yu == -1 and scores[i] < 0.5)
                neigh.append(i)
       
        labels[list(neigh)] = yu  # k neighbours
        return labels
    
    # =====================================
    def get_scores(self, labels): # change in update function
        clf = RandomForestClassifier(n_estimators=50, random_state=1).fit(self.X, labels) 
        #"class_weight={-1:1,1:100}"......class_weight='balanced_subsample'
        #clf = OneClassSVM(gamma='auto')
        scores = clf.predict_proba(self.X)[:, 1]
        scores = np.asmatrix(scores)
        scores = np.transpose(scores)
        return scores
    
    # =====================================
    '''def weights(self, labels): # change in update function
        
        return weights

    # =====================================
    def get_neighbours(self, X, u,yu): # change in update function
        dist = []
        
        for ele in self.X:
                dist.append(euclidean(self.X[u],ele))
            temp_score = np.argsort(dist)
            labels[list(temp_score[:15])] = yu
        return '''
    # =====================================
    '''def query(self, Z, X, clf, y, budget):
        dist = []
        #labels = self.get_labels()##########
        anomalies = 0
        nominals = 0
        for _ in range(budget):
            #scores = clf.fit(self.X, np.ravel(labels,order='C')).predict_proba(self.X)[:, 1]
            scores = self.get_scores()
            i = np.argmax(scores)
            feedback = y[i]  # this is the expert feedback
            if feedback == 1:
                anomalies = anomalies + 1
            else:
                nominals = nominals + 1
            labels[i] = feedback # update true label
            scores[i] = -np.inf
            for ele in self.X:
                #dist.append(np.sum(np.power(np.subtract(self.X[i],ele),2))) 
                dist.append(euclidean(self.X[i],ele))
            temp_score = np.argsort(dist)
            labels[list(temp_score[:15])] = feedback###############
        return labels, anomalies, nominals  '''
      
    # =====================================
    def update(self, u, yu):  # u == i,yu == feedback(true label)
        super().update(u, yu)
        self.labels = self.get_labels(self.scores,u, yu)
        self.scores = self.get_scores(self.labels)
        
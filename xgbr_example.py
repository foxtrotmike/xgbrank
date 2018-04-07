#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:02:25 2018

@author: fayyaz
"""

import numpy as np
from xgbranker import XGBRanker


import matplotlib.pyplot as plt
import itertools

def plotit(X,Y=None,clf=None, markers = ('s','o'), hold = False, transform = None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf (2D), taken from: https://github.com/foxtrotmike/svmtutorial/blob/master/svmtutorial.ipynb
    """
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    if clf is not None:
        npts = 100
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf.decision_function(t)
        z = np.reshape(z,(npts,npts))
        
        extent = [minx,maxx,miny,maxy]
        plt.imshow(z,vmin = -2, vmax = +2)    
        plt.contour(z,[-1,0,1],linewidths = [2],colors=('b','k','r'),extent=extent, label='f(x)=0')
        plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.axis([minx,maxx,miny,maxy])   
    if Y is not None:
        plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
        plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')        
         
    else:
        plt.scatter(X[:,0],X[:,1],marker = '.', c = 'k', s = 5)
    if not hold:
        plt.grid()
        plt.show()
            
if __name__=='__main__':
    
    Xp = 1+np.random.randn(50,2)
    Xn = -1-np.random.randn(50,2)
    X = np.vstack((Xp,Xn))
    Y = np.array([1]*Xp.shape[0]+[-1]*Xn.shape[0])
    print 'The data dimensions are',X.shape, Y.shape


    Xpt = 1+np.random.randn(50,2)
    Xnt = -1-np.random.randn(50,2)
    Xt = np.vstack((Xpt,Xnt))
    Yt = np.array([1]*Xpt.shape[0]+[-1]*Xnt.shape[0])    
    
    G = np.random.randint(0,5,len(Y))+1
    rs = XGBRanker(n_estimators=150, learning_rate=0.1, subsample=0.9)#, objective='rank:pairwise')
    rs.decision_function = rs.predict
# 100 samples, 4 groups, 25/group, predicted_values can be used to sort x in their own group

    rs.fit(X,Y,G)
       
    yp =  rs.decision_function(X)
    from sklearn.metrics import roc_auc_score
    print "ROC",roc_auc_score(Yt,yp)   
    
    yp =  rs.decision_function(Xt)
    print "ROC",roc_auc_score(Yt,yp)  
    plotit(X,Y, rs)
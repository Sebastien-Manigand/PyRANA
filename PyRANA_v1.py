# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:14:48 2021

@author: sebas
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import csv
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'





class PyRANA:
    
    def __init__(self):
        self.filename = ""
        self.X = None
        self.Y = None
        self.errX = None
        self.errY = None
        self.quickFitRes = {}
        self.dataname = "data"
        self.datanames = []
        self.fits_groups = []
        return
    
    
    
    
    
    def poly1(self, X, a, b):
        return a + b*X
    
    def poly2(self, X, a, b, c):
        return a + b*X + c*X**2
    
    def gauss(self, X, a, b, c):
        return a * np.exp(-(X - b)**2 / (2*c**2)) 
    
    def cauchy(self, X, a, b, c):
        return a / (1 + ((X - b) / c)**2 ) * self.gauss(X, 1, b, 5*c)
    
    def assymCauchy(self, X, I, x0, s, a):
        g = 2*s / (1 + np.exp(a*(X-x0)))
        return I / (1 + ((X - x0) / g)**2 ) * self.gauss(X, 1, x0, 5*s)
    
    def assymGauss(self, X, I, x0, s, a):
        g = 2*s / (1 + np.exp(a*(X-x0)))
        return I * np.exp(-(X - x0)**2 / (2*g**2)) 
        
    def bwfGraphene(self, X, I, x0, s, a):
        l = (X - x0) / s
        return I * ( a**2 + (1 - a**2 + l*a) / (1 + l**2) )

    
    
    def fctWrapper2(self, X, fctdef, a, b):
        if fctdef == 'poly1':
            return self.poly1(X, a, b)
        else:
            print("fctWrapper2 ERROR: fctdef '{0}' not defined here...".format(fctdef))
            sys.exit()
    
    def fctWrapper3(self, X, fctdef, a, b, c):
        if fctdef == 'poly2':
            return self.poly2(X, a, b, c)
        elif fctdef == 'gauss':
            return self.gauss(X, a, b, c)
        elif fctdef == 'cauchy':
            return self.cauchy(X, a, b, c)
        else:
            print("fctWrapper3 ERROR: fctdef '{0}' not defined here...".format(fctdef))
            sys.exit()
    
     
    def fctWrapper4(self, X, fctdef, a, b, c, d):
        if  fctdef == 'assymGauss':
            return self.assymGauss(X, a, b, c, d)
        elif  fctdef == 'assymCauchy':
            return self.assymCauchy(X, a, b, c, d)
        elif  fctdef == 'BWFGraphene':
            return self.bwfGraphene(X, a, b, c, d)
        else:
            print("fctWrapper4 ERROR: fctdef '{0}' not defined here...".format(fctdef))
            sys.exit()
    
    
    
    
    def modelWrapper(self, X, modeldef, bound_x0=20, bound_s=20, bound_amin=-1, bound_amax=1, comment=False):
        numpar = []
        p0 = []
        fct_list = ['poly1', 'poly2', 'gauss', 'cauchy', 'assymGauss', 'assymCauchy', 'BWFGraphene']
        for i in range(len(modeldef)):
            if not modeldef[i][0] in fct_list:
                print("modelWrapper ERROR: {0} does not exist, please review your code...".format(modeldef[i][0]))
                sys.exit()
            numpar.append(len(modeldef[i])-1)
            
        for i in range(1, len(numpar)):
            for j in range(1, len(numpar)-1):
                if numpar[j] < numpar[j+1]:
                    num = numpar[j]
                    numpar[j] = numpar[j+1]
                    numpar[j+1] = num
                    mold = modeldef[j]
                    modeldef[j] = modeldef[j+1]
                    modeldef[j+1] = mold
            
        for i in range(len(modeldef)):
            for j in range(1, len(modeldef[i])):
                p0.append(modeldef[i][j])
            
        #totpar = np.sum(numpar)
        bounds = [[], []]
        for i in range(numpar[0]):
            bounds[0].append(-np.inf)
            bounds[1].append(np.inf)
        for i in range(1, len(numpar)):
            bounds[0].append(0) # intensity
            bounds[1].append(np.inf)
            bounds[0].append(modeldef[i][2]-bound_x0) # position
            bounds[1].append(modeldef[i][2]+bound_x0) 
            bounds[0].append(max(modeldef[i][3]-bound_s, 0)) # width
            bounds[1].append(modeldef[i][3]+bound_s) 
            if numpar[i] == 4:
                bounds[0].append(bound_amin) # assymmetric factor
                bounds[1].append(bound_amax)
            
        
        if numpar == [2]:
            if comment: print("detected parameters: [2]")
            return lambda X, a, b: self.fctWrapper2(X, modeldef[0][0], a, b), p0, bounds
            
        elif numpar == [3]:
            if comment: print("detected parameters: [3]")
            return lambda X, a, b, c: self.fctWrapper3(X, modeldef[0][0], a, b, c), p0, bounds
            
        
        
        elif numpar == [2, 3]:
            if comment: print("detected parameters: [2, 3]")
            return lambda X, a, b, c, d, e: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper3(X, modeldef[1][0], c, d, e), p0, bounds
            
        elif numpar == [2, 4]:
            if comment: print("detected parameters: [2, 4]")
            return lambda X, a, b, c, d, e, f: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f), p0, bounds
            
        elif numpar == [3, 3]:
            if comment: print("detected parameters: [3, 3]")
            return lambda X, a, b, c, d, e, f: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper3(X, modeldef[1][0], d, e, f), p0, bounds
            
        elif numpar == [3, 4]:
            if comment: print("detected parameters: [3, 4]")
            return lambda X, a, b, c, d, e, f, g: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g), p0, bounds
            
        
        
        elif numpar == [2, 3, 3]:
            if comment: print("detected parameters: [2, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper3(X, modeldef[1][0], c, d, e) + self.fctWrapper3(X, modeldef[2][0], f, g, h), p0, bounds
        
        elif numpar == [2, 4, 3]:
            if comment: print("detected parameters: [2, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i), p0, bounds
        
        elif numpar == [2, 4, 4]:
            if comment: print("detected parameters: [2, 3, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j), p0, bounds
        
        elif numpar == [3, 3, 3]:
            if comment: print("detected parameters: [3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper3(X, modeldef[1][0], d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i), p0, bounds
        
        elif numpar == [3, 4, 3]:
            if comment: print("detected parameters: [3, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper3(X, modeldef[2][0], h, i, j), p0, bounds
        
        elif numpar == [3, 4, 4]:
            if comment: print("detected parameters: [3, 4, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k), p0, bounds
        
        
        
        elif numpar == [2, 3, 3, 3]:
            if comment: print("detected parameters: [2, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper3(X, modeldef[1][0], c, d, e) + self.fctWrapper3(X, modeldef[2][0], f, g, h) + self.fctWrapper3(X, modeldef[3][0], i, j, k), p0, bounds
        
        elif numpar == [2, 4, 3, 3]:
            if comment: print("detected parameters: [2, 4, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i) + self.fctWrapper3(X, modeldef[3][0], j, k, l), p0, bounds
        
        elif numpar == [2, 4, 4, 3]:
            if comment: print("detected parameters: [2, 4, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper3(X, modeldef[3][0], k, l, m), p0, bounds
        
        elif numpar == [2, 4, 4, 4]:
            if comment: print("detected parameters: [2, 4, 4, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper4(X, modeldef[3][0], k, l, m, n), p0, bounds
        
        elif numpar == [3, 3, 3, 3]:
            if comment: print("detected parameters: [3, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper3(X, modeldef[1][0], d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i) + self.fctWrapper3(X, modeldef[3][0], j, k, l), p0, bounds
        
        elif numpar == [3, 4, 3, 3]:
            if comment: print("detected parameters: [3, 4, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper3(X, modeldef[2][0], h, i, j) + self.fctWrapper3(X, modeldef[3][0], k, l, m), p0, bounds
        
        elif numpar == [3, 4, 4, 3]:
            if comment: print("detected parameters: [3, 4, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper3(X, modeldef[3][0], l, m, n), p0, bounds
        
        elif numpar == [3, 4, 4, 4]:
            if comment: print("detected parameters: [3, 4, 4, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper4(X, modeldef[3][0], l, m, n, o), p0, bounds
        
        
        elif numpar == [2, 3, 3, 3, 3]:
            if comment: print("detected parameters: [2, 3, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper3(X, modeldef[1][0], c, d, e) + self.fctWrapper3(X, modeldef[2][0], f, g, h) + self.fctWrapper3(X, modeldef[3][0], i, j, k) + self.fctWrapper3(X, modeldef[4][0], l, m, n), p0, bounds
        
        elif numpar == [2, 4, 3, 3, 3]:
            if comment: print("detected parameters: [2, 4, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i) + self.fctWrapper3(X, modeldef[3][0], j, k, l) + self.fctWrapper3(X, modeldef[4][0], m, n, o), p0, bounds
        
        elif numpar == [2, 4, 4, 3, 3]:
            if comment: print("detected parameters: [2, 4, 4, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper3(X, modeldef[3][0], k, l, m) + self.fctWrapper3(X, modeldef[4][0], n, o, p), p0, bounds
        
        elif numpar == [2, 4, 4, 4, 3]:
            if comment: print("detected parameters: [2, 4, 4, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper4(X, modeldef[3][0], k, l, m, n) + self.fctWrapper3(X, modeldef[4][0], o, p, q), p0, bounds
        
        elif numpar == [2, 4, 4, 4, 4]:
            if comment: print("detected parameters: [2, 4, 4, 4, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper4(X, modeldef[3][0], k, l, m, n) + self.fctWrapper4(X, modeldef[4][0], o, p, q, r), p0, bounds
        
        elif numpar == [3, 3, 3, 3, 3]:
            if comment: print("detected parameters: [3, 3, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper3(X, modeldef[1][0], d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i) + self.fctWrapper3(X, modeldef[3][0], j, k, l) + self.fctWrapper3(X, modeldef[4][0], m, n, o), p0, bounds
        
        elif numpar == [3, 4, 3, 3, 3]:
            if comment: print("detected parameters: [3, 4, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper3(X, modeldef[2][0], h, i, j) + self.fctWrapper3(X, modeldef[3][0], k, l, m) + self.fctWrapper3(X, modeldef[4][0], n, o, p), p0, bounds
        
        elif numpar == [3, 4, 4, 3, 3]:
            if comment: print("detected parameters: [3, 4, 4, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper3(X, modeldef[3][0], l, m, n) + self.fctWrapper3(X, modeldef[4][0], o, p, q), p0, bounds
        
        elif numpar == [3, 4, 4, 4, 3]:
            if comment: print("detected parameters: [3, 4, 4, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper4(X, modeldef[3][0], l, m, n, o) + self.fctWrapper3(X, modeldef[4][0], p, q, r), p0, bounds
        
        elif numpar == [3, 4, 4, 4, 4]:
            if comment: print("detected parameters: [3, 4, 4, 4, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper4(X, modeldef[3][0], l, m, n, o) + self.fctWrapper4(X, modeldef[4][0], p, q, r, s), p0, bounds
        
        
        elif numpar == [2, 3, 3, 3, 3, 3]:
            if comment: print("detected parameters: [2, 3, 3, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper3(X, modeldef[1][0], c, d, e) + self.fctWrapper3(X, modeldef[2][0], f, g, h) + self.fctWrapper3(X, modeldef[3][0], i, j, k) + self.fctWrapper3(X, modeldef[4][0], l, m, n) + self.fctWrapper3(X, modeldef[5][0], o, p, q), p0, bounds
        
        elif numpar == [2, 4, 3, 3, 3, 3]:
            if comment: print("detected parameters: [2, 4, 3, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i) + self.fctWrapper3(X, modeldef[3][0], j, k, l) + self.fctWrapper3(X, modeldef[4][0], m, n, o) + self.fctWrapper3(X, modeldef[5][0], p, q, r), p0, bounds
        
        elif numpar == [2, 4, 4, 3, 3, 3]:
            if comment: print("detected parameters: [2, 4, 4, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper3(X, modeldef[3][0], k, l, m) + self.fctWrapper3(X, modeldef[4][0], n, o, p) + self.fctWrapper3(X, modeldef[5][0], q, r, s), p0, bounds
        
        elif numpar == [2, 4, 4, 4, 3, 3]:
            if comment: print("detected parameters: [2, 4, 4, 4, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper4(X, modeldef[3][0], k, l, m, n) + self.fctWrapper3(X, modeldef[4][0], o, p, q) + self.fctWrapper3(X, modeldef[5][0], r, s, t), p0, bounds
        
        elif numpar == [2, 4, 4, 4, 4, 3]:
            if comment: print("detected parameters: [2, 4, 4, 4, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper4(X, modeldef[3][0], k, l, m, n) + self.fctWrapper4(X, modeldef[4][0], o, p, q, r) + self.fctWrapper3(X, modeldef[5][0], s, t, u), p0, bounds
        
        elif numpar == [2, 4, 4, 4, 4, 4]:
            if comment: print("detected parameters: [2, 4, 4, 4, 4, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v: self.fctWrapper2(X, modeldef[0][0], a, b) + self.fctWrapper4(X, modeldef[1][0], c, d, e, f) + self.fctWrapper4(X, modeldef[2][0], g, h, i, j) + self.fctWrapper4(X, modeldef[3][0], k, l, m, n) + self.fctWrapper4(X, modeldef[4][0], o, p, q, r) + self.fctWrapper3(X, modeldef[5][0], s, t, u, v), p0, bounds
        
        elif numpar == [3, 3, 3, 3, 3, 3]:
            if comment: print("detected parameters: [3, 3, 3, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper3(X, modeldef[1][0], d, e, f) + self.fctWrapper3(X, modeldef[2][0], g, h, i) + self.fctWrapper3(X, modeldef[3][0], j, k, l) + self.fctWrapper3(X, modeldef[4][0], m, n, o) + self.fctWrapper3(X, modeldef[4][0], p, q, r), p0, bounds
        
        elif numpar == [3, 4, 3, 3, 3, 3]:
            if comment: print("detected parameters: [3, 4, 3, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper3(X, modeldef[2][0], h, i, j) + self.fctWrapper3(X, modeldef[3][0], k, l, m) + self.fctWrapper3(X, modeldef[4][0], n, o, p) + self.fctWrapper3(X, modeldef[4][0], q, r, s), p0, bounds
        
        elif numpar == [3, 4, 4, 3, 3, 3]:
            if comment: print("detected parameters: [3, 4, 4, 3, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper3(X, modeldef[3][0], l, m, n) + self.fctWrapper3(X, modeldef[4][0], o, p, q) + self.fctWrapper3(X, modeldef[4][0], r, s, t), p0, bounds
        
        elif numpar == [3, 4, 4, 4, 3, 3]:
            if comment: print("detected parameters: [3, 4, 4, 4, 3, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper4(X, modeldef[3][0], l, m, n, o) + self.fctWrapper3(X, modeldef[4][0], p, q, r) + self.fctWrapper3(X, modeldef[4][0], s, t, u), p0, bounds
        
        elif numpar == [3, 4, 4, 4, 4, 3]:
            if comment: print("detected parameters: [3, 4, 4, 4, 4, 3]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper4(X, modeldef[3][0], l, m, n, o) + self.fctWrapper4(X, modeldef[4][0], p, q, r, s) + self.fctWrapper3(X, modeldef[4][0], t, u, v), p0, bounds
        
        elif numpar == [3, 4, 4, 4, 4, 4]:
            if comment: print("detected parameters: [3, 4, 4, 4, 4, 4]")
            return lambda X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w: self.fctWrapper3(X, modeldef[0][0], a, b, c) + self.fctWrapper4(X, modeldef[1][0], d, e, f, g) + self.fctWrapper4(X, modeldef[2][0], h, i, j, k) + self.fctWrapper4(X, modeldef[3][0], l, m, n, o) + self.fctWrapper4(X, modeldef[4][0], p, q, r, s) + self.fctWrapper4(X, modeldef[4][0], t, u, v, w), p0, bounds
        
        
        
        
        else:
            if comment: print("modelWrapper ERROR: modeldef {0} is not defined in the wrapper, please implement it...".format(numpar))
            sys.exit()
        
        
        
    def loadfile_SALSA(self, fname):
        X = []
        Y = []
        try:
            f = open(fname, 'r') 
            lines = list(csv.reader(f, delimiter=','))
            self.dataname = fname.split('.')[0]
            self.datanames.append(self.dataname)
            self.fits_groups.append([])    
        except:
            print("failed to read the file \"{0}\"".format(fname))
            return -1
        inData = False
        for i in range(len(lines)):
            if i > 0 and inData:
                X.append(float(lines[i][0]))
                Y.append(float(lines[i][1]))
            else:
                if lines[i][0] == 'DATA':
                    inData = True
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        return 0
    
    
    
    
    def loadfile_METAFILE(self, fname):
        X = []
        Y = []
        try:
            f = open(fname, 'r') 
            lines = f.readlines()#list(csv.reader(f, delimiter='\t'))
            self.dataname = fname.split('.')[0]
            self.datanames.append(self.dataname)
            self.fits_groups.append([])    
        except:
            print("failed to read the file \"{0}\"".format(fname))
            return -1
        inData = False
        for i in range(len(lines)):
            if lines[i][0] != '#':
                if i > 0 and inData:
                    bf = lines[i].strip().split(' ')
                    while('' in bf):
                        bf.remove('')
                    X.append(float(bf[0]))
                    Y.append(float(bf[1]))
                else:
                    if lines[i].strip() == 'DATA':
                        print("DATA flag detected")
                        inData = True
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        print("{0} data points".format(len(self.X)))
        return 0
    
    
    
    
    def loadfile_ASCII(self, fname):
        X = []
        Y = []
        Yerr = []
        try:
            f = open(fname, 'r') 
            lines = f.readlines()#list(csv.reader(f, delimiter='\t'))
            self.dataname = fname.split('.')[0]
            self.datanames.append(self.dataname)
            self.fits_groups.append([])    
        except:
            print("failed to read the file \"{0}\"".format(fname))
            return -1
        for i in range(len(lines)):
            if lines[i][0] != '#':
                bf = lines[i].strip().split('\t')
                while('' in bf):
                    bf.remove('')
                X.append(float(bf[0]))
                Y.append(float(bf[1]))
                if len(bf) > 2:
                    Yerr.append(float(bf[2]))
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        if len(Yerr) > 0:
            self.Yerr = np.asarray(Yerr)
        print("{0} data points".format(len(self.X)))
        return 0
    
    
    
    
    def getRMS(self, wn = None, spec = None, method='fft', _DEBUG = False):
        '''
        Returns the rms and the processed arrays.
    
        Parameters
        ----------
        wn : numpy.array or list
            x-array.
        spec : numpy.array or list
            y-array.
        method : str, optional
            Flag corresponding to the method used to derive the rms. The default is 'fft'.
            Available methods:
                - 'fft': This method uses the Fast Fourier Transform (FFT) of the
                        spectrum. In the transformed space, the short frequencies
                        corresponds to the continuum and the raman peaks (relatively
                        broad spectral features) while the high frequencies corresponds
                        to the fast fluctuations. Assuming the noise is the major
                        component of the fast fluctuations, the width of the 
                        distribution of the fast fluctuation gives a direct measure
                        of the rms.
                        The algorithm follows the following steps:
                            - The FFT of the spectrum is computed,
                            - The short frequencies, with period lower than 3 pixels, 
                            are set to zero,
                            - The spectrum is recovered by inversing the Fourier tranform,
                            - The std of the filtered spectrum  is calculated and used
                            to scale the histogram.
                            - The histogram is fitted to a Gaussian function, where the
                            width sigma = rms / (sqrt(2*log(2)))
                            
        Returns
        -------
        rms: float
            The rms of the filtered FFT. Default/error is 0.
            
        _DEBUG = True
        -------
        returns freq_fft, filtered spec, fit_x0, rms, fit_ampl
    
        '''
        
        if wn is None:
            if self.X is None:
                print("Error--getRMS: wn and self.X are both None...")
                return -1
            else:
                wn = self.X
        if spec is None:
            if self.Y is None:
                print("Error--getRMS: spec and self.Y are both None...")
                return -1
            else:
                spec = self.Y
        
        
        
        if method == 'fft':
            #print(abs(np.average(wn[1:]-wn[:-1])))
            spec_fft = np.fft.fft(spec)
            filtered = np.fft.fft(spec)
            freq_fft = np.fft.fftfreq(len(wn), d=abs(np.average(wn[1:]-wn[:-1])) )
            for i in range(len(spec_fft)):
                if abs(freq_fft[i]) < 1.0/(4*abs(np.average(wn[1:]-wn[:-1]))):
                    filtered[i] *= 0
                    #filtered[i].img = 0
            spec_ = np.fft.ifft(filtered)
            spec_ = np.abs(spec_) * np.real(spec_) / np.abs(np.real(spec_))
            std = np.std(spec_)
            #print(std)
            b, e = np.histogram(spec_, bins=100, range=(-3*std, 3*std))
            e = (e[1:]+e[:-1])/2.0
            par, cov = curve_fit(self.gauss, e, b, p0=[0, 20.0, std], bounds=[[-std/2, 10, std/2], [std/2, 100, 2*std]], maxfev=10000)
            if _DEBUG:
                return b, e, par[0], par[2]*(np.sqrt(2*np.log(2))), par[1] #np.mean(spec_), np.std(spec_)
            else:
                return par[2]*(np.sqrt(2*np.log(2)))
            
        else:
            print("Warning: getRMS >> '{0}' method not recognized, please refer to the documentation.")
            return 0.0
        
        return None
    
    
    
    
    def cleanSpectrum(self, rms=None, comment=False, ignore=None, forced=None):
        if rms is None:
            rms = self.getRMS()
            if comment: print("rms: {0:.2f}".format(rms))
    
        dx = 100 # range of the sub-windows (in the X-axis unit)
        deadPix = []
        deadPix_width = []
        
        for i in range(int((max(self.X)-min(self.X)) / dx)):
            X_ = self.X[abs(self.X - (min(self.X)+(int(i) + 0.5)*dx)) <= dx/2]
            Y_ = self.Y[abs(self.X - (min(self.X)+(int(i) + 0.5)*dx)) <= dx/2]
            
            res = np.polynomial.polynomial.polyfit(X_, Y_, deg=10)
            Yfit = np.polyval(np.flip(res), X_)
            
            rms_ = 0.5*np.std(Y_-Yfit)
        
            X_filt = X_[abs(Y_-Yfit) < 3*max(rms, rms_)]
            Y_filt = Y_[abs(Y_-Yfit) < 3*max(rms, rms_)]
            X_dead = X_[(abs(Y_-Yfit) >= 3*max(rms, rms_))]
            
            if(len(X_dead) > 0):
                #print(len(X_dead))
                res = np.polynomial.polynomial.polyfit(X_filt, Y_filt, deg=10)
                Y_filt_fit = np.polyval(np.flip(res), X_)
                Y_pix = np.zeros_like(X_)
                
                for x in X_dead:
                    deadPix.append(x)
                    try:
                        res, cov = curve_fit(self.gauss, X_, Y_-Y_filt_fit, p0=[100, deadPix[-1], 0.5], maxfev=10000)
                        Y_pix = Y_pix + self.gauss(X_, *res)
                        deadPix_width.append(abs(res[2]))
                    except:
                        deadPix_width.append(0)
                
        deadPix_valid = []
        for x, s in zip(deadPix, deadPix_width):
            
            X_window = self.X[abs(self.X - x) <= dx/2]
            
            if(s > np.mean(X_window[1:]-X_window[:-1])):
                deadPix_valid.append(False)
            else:
                deadPix_valid.append(True)
        
        idToPop = []
        for i in range(len(deadPix)):
            if not deadPix_valid[i]:
                idToPop.append(i)
                
        for i in np.flip(idToPop):
            deadPix.pop(i)
            deadPix_width.pop(i)
            deadPix_valid.pop(i)
        
        deadPix = np.array([deadPix, deadPix_width]).T
        
        if not forced is None:
            for i in range(len(forced)):    
                deadPix = np.append(deadPix, [[forced[i], 0.5]], axis=0)
        
        if comment: print(deadPix)
        
        for i in range(len(deadPix)):
            if not ignore is None:
                if min(np.abs(np.asarray(ignore) - deadPix[i][0]) ) < 2:
                    if comment: print("ignoring p={0:.2f} cm-1".format(deadPix[i][0]))
                else:
                    if comment: print("correcting p={0:.2f} cm-1".format(deadPix[i][0]))
                    d = np.abs(self.X - deadPix[i][0])
                    dmin = min(d)
                    id_pix = list(d).index(dmin)
                    self.Y[id_pix] = np.mean([self.Y[id_pix-2], self.Y[id_pix-1], self.Y[id_pix+1], self.Y[id_pix+2]])
            else:
                if comment: print("correcting p={0:.2f} cm-1".format(deadPix[i][0]))
                d = np.abs(self.X - deadPix[i][0])
                dmin = min(d)
                id_pix = list(d).index(dmin)
                self.Y[id_pix] = np.mean([self.Y[id_pix-2], self.Y[id_pix-1], self.Y[id_pix+1], self.Y[id_pix+2]])
            
    
    
    
    
    
    def quickfit(self, X=None, Y=None, errY_=None, continuum = 'linear', peaks = [{'profile': 'cauchy', 'x0': None, 'I': None, 's': None}],
                 Xmin = None, Xmax = None,
                 use_diffXcontrib=True):
    
        bound_x0 = 20
        bound_s0 = 20
        bound_a0 = 1
        errY = None
        
        if X is None:
            if self.X is None:
                print("Error--quickFit: X and self.X are both None...")
                return -1
            else:
                X = self.X
        if Y is None:
            if self.Y is None:
                print("Error--quickFit: Y and self.Y are both None...")
                return -1
            else:
                Y = self.Y
        if errY_ is None:
            if errY_ is None:
                #print("Warning--quickFit: errY and self.errY are both None, replacing by rms from getRMS()")
                errY = self.getRMS(wn=X, spec=Y) * np.ones_like(X)
            else:
                errY = errY_
           
        errX = np.interp(X, (X[1:]+X[:-1])/2, (X[1:]-X[:-1]))
            
            
            
        if Xmin is None:
            Xmin = min(X)
        if Xmax is None:
            Xmax = max(X)
        
        Y = Y[(X >= Xmin) & (X <= Xmax)]
        errY = errY[(X >= Xmin) & (X <= Xmax)]
        errX = errX[(X >= Xmin) & (X <= Xmax)]
        X = X[(X >= Xmin) & (X <= Xmax)]
        
        
        peakdef = []
        
        if continuum is None or continuum == 'linear':
            peakdef.append(['poly1', min(Y), 0.1])
        elif continuum == 'poly2':
            peakdef.append(['poly2', min(Y), 0.1, 0.01])
            
        for i in range(len(peaks)):
            if peaks[i]['profile'] in ['gauss', 'assymGauss', 'cauchy', 'assymCauchy', 'BWFGraphene']:
                
                peakdef.append([peaks[i]['profile']])
                
                if 'I' in peaks[i].keys():
                    if not peaks[i]['I'] is None:
                        peakdef[-1].append(peaks[i]['I'])
                    else:
                        peakdef[-1].append(1000)
                else:
                    peakdef[-1].append(1000)
                if 'x0' in peaks[i].keys():
                    if not peaks[i]['x0'] is None:
                        peakdef[-1].append(peaks[i]['x0'])
                    else:
                        peakdef[-1].append((Xmax-Xmin)/2)
                else:
                    peakdef[-1].append((Xmax-Xmin)/2)
                if 's' in peaks[i].keys():
                    if not peaks[i]['s'] is None:
                        peakdef[-1].append(peaks[i]['s'])
                    else:
                        peakdef[-1].append(20)
                else:
                    peakdef[-1].append(20)
                
            if peaks[i]['profile'] in ['assymGauss', 'assymCauchy', 'BWFGraphene']:
                if 'a' in peaks[i].keys():
                    if not peaks[i]['a'] is None:
                        peakdef[-1].append(peaks[i]['a'])
                    else:
                        peakdef[-1].append(0.1)
                else:
                    peakdef[-1].append(0.1)
                    
                    
        model, p0, bounds = self.modelWrapper(X, peakdef)
        
                
        
        res, cov = curve_fit(model, X, Y, p0=p0, sigma=errY, absolute_sigma=True, bounds=bounds, maxfev=10000)
        fitY = model(X, *res)
        
        if use_diffXcontrib:
            dYdX = np.diff(fitY)/np.diff(X)
            res, cov = curve_fit(model, X, Y, p0=p0, sigma=np.sqrt(np.power(errX*np.interp(X, (X[1:]+X[:-1])/2, dYdX), 2) + errY*errY), absolute_sigma=True, bounds=bounds, maxfev=10000)
    
        err = [np.sqrt(2*np.log(2) * cov[i][i]) for i in range(len(cov))]
        
        bestfit = [[peakdef[i][0]] for i in range(len(peakdef))]# peakdef.copy()
        besterr = [[peakdef[i][0]] for i in range(len(peakdef))]# peakdef.copy()
        index = 0
        for i in range(len(peakdef)):
            if peakdef[i][0] == 'poly1':
                bestfit[i].append(res[index])
                bestfit[i].append(res[index + 1])
                besterr[i].append(err[index])
                besterr[i].append(err[index + 1])
            elif peakdef[i][0] == 'poly2' or peakdef[i][0] == 'gauss' or peakdef[i][0] == 'cauchy':
                bestfit[i].append(res[index])
                bestfit[i].append(res[index + 1])
                bestfit[i].append(res[index + 2])
                besterr[i].append(err[index])
                besterr[i].append(err[index + 1])
                besterr[i].append(err[index + 2])
            # elif peakdef[i][0] == 'gauss':
            #     bestfit[i].append(res[index])
            #     bestfit[i].append(res[index + 1])
            #     bestfit[i].append(res[index + 2])
            #     besterr[i].append(err[index])
            #     besterr[i].append(err[index + 1])
            #     besterr[i].append(err[index + 2])
            # elif peakdef[i][0] == 'cauchy':
            #     bestfit[i].append(res[index])
            #     bestfit[i].append(res[index + 1])
            #     bestfit[i].append(res[index + 2])
            #     besterr[i].append(err[index])
            #     besterr[i].append(err[index + 1])
            #     besterr[i].append(err[index + 2])
            elif peakdef[i][0] == 'assymGauss' or peakdef[i][0] == 'assymCauchy' or peakdef[i][0] == 'BWFGraphene':
                bestfit[i].append(res[index])
                bestfit[i].append(res[index + 1])
                bestfit[i].append(res[index + 2])
                bestfit[i].append(res[index + 3])
                besterr[i].append(err[index])
                besterr[i].append(err[index + 1])
                besterr[i].append(err[index + 2])
                besterr[i].append(err[index + 3])
            # elif peakdef[i][0] == 'assymCauchy':
            #     bestfit[i].append(res[index])
            #     bestfit[i].append(res[index + 1])
            #     bestfit[i].append(res[index + 2])
            #     bestfit[i].append(res[index + 3])
            #     besterr[i].append(err[index])
            #     besterr[i].append(err[index + 1])
            #     besterr[i].append(err[index + 2])
            #     besterr[i].append(err[index + 3])
            index += len(peakdef[i]) - 1
        
        fit = { 'bestfit': bestfit,
                'errors': besterr,
                'dataX': np.asarray(X),
                'dataY': np.asarray(Y),
                'dataYerr': np.sqrt(np.power(np.interp(X, (X[1:]+X[:-1])/2, dYdX), 2) + errY*errY) if use_diffXcontrib else np.asarray(errY),
                'model': fitY
                 }
        self.fits_groups[-1].append(fit)
        
        return fit
    
    
    
    
    def print_peaks(self, bestfit, besterr):
        print("\n====={   BEST-FIT RESULTS   }=====")
        for i in range(len(bestfit)):
            if bestfit[i][0] == 'gauss':
                print("| Gaussian peak")
            elif bestfit[i][0] == 'cauchy':
                print("| Cauchy peak")
            elif bestfit[i][0] == 'assymGauss':
                print("| Assymmetric Gaussian peak")
            elif bestfit[i][0] == 'assymCauchy':
                print("| Assymmetric Cauchy peak")
            elif bestfit[i][0] == 'BWFGraphene':
                print("| Breit-Wigner-Fano peak")
            
            if bestfit[i][0] in ['gauss', 'cauchy', 'assymGauss', 'assymCauchy', 'BWFGraphene']:
                print("|    x0 = {0:.2f} +- {1:.2f} cm-1".format(bestfit[i][2], besterr[i][2]))
                print("|    I  = {0:.0f} +- {1:.0f} counts".format(bestfit[i][1], besterr[i][1]))
                print("|    s  = {0:.2f} +- {1:.2f} cm-1".format(bestfit[i][3], besterr[i][3]))
            if bestfit[i][0] in ['assymGauss', 'assymCauchy', 'BWFGraphene']:
                print("|    a  = {0:.3f} +- {1:.3f} ".format(bestfit[i][4], besterr[i][4]))
            
            # if bestfit[i][0] != 'poly1' and bestfit[i][0] != 'poly2':
            #     print("")
        print("==================================")
    
    
    def chi2_(self, Y, Yerr, Yfit):
        return np.sum((Y - Yfit)**2/(Yerr*Yerr))
    
    
    def str_peaks(self, fits, typetxt='ascii'):
        peaks = []
        pos = []
        posErr = []
        Int = []
        IntErr = []
        sig = []
        sigErr = []
        asy = []
        asyErr = []
        chi2 = []
        N = []
        p = []
        
        for j in range(len(fits)):
            bestfit = fits[j]['bestfit']
            besterr = fits[j]['errors']
            n = len(fits[j]['dataX'])
            c = self.chi2_(fits[j]['dataY'], fits[j]['dataYerr'], fits[j]['model'])
            pr = stats.chi2.sf(x=c, df=int(n-np.sum([len(l) for l in bestfit[1:]])))
            for i in range(1, len(bestfit)):
                if bestfit[i][0] == 'gauss':
                    peaks.append("Gaussian")
                elif bestfit[i][0] == 'cauchy':
                    peaks.append("Cauchy")
                elif bestfit[i][0] == 'assymGauss':
                    peaks.append("assymmetric Gaussian")
                elif bestfit[i][0] == 'assymCauchy':
                    peaks.append("assymmetric Cauchy")
                elif bestfit[i][0] == 'BWFGraphene':
                    peaks.append("Breit-Wigner-Fano")
                
                if bestfit[i][0] in ['gauss', 'cauchy', 'assymGauss', 'assymCauchy', 'BWFGraphene']:
                    pos.append(bestfit[i][2])
                    Int.append(bestfit[i][1])
                    sig.append(bestfit[i][3])
                    posErr.append(besterr[i][2])
                    IntErr.append(besterr[i][1])
                    sigErr.append(besterr[i][3])
                    chi2.append(c)
                    N.append(n)
                    p.append(pr)
                if bestfit[i][0] in ['assymGauss', 'assymCauchy', 'BWFGraphene']:
                    asy.append(bestfit[i][4])
                    asyErr.append(besterr[i][4])
                else:
                    asy.append(None)
                    asyErr.append(None)
        
        s = ""
        if typetxt == 'ascii':
            for i in range(len(peaks)):
                s += "{0}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.3f}\t{6:.3f}\t".format(
                    peaks[i], pos[i], posErr[i], Int[i], IntErr[i], sig[i], sigErr[i])
                if asy[i] is None:
                    s += "\t\t{0:.3e}\t{1}\t{2:.3e}".format(chi2[i], N[i], p[i])
                else:
                    s += "{0:.4f}\t{1:.4f}\t{2:.3e}\t{3}\t{4:.3e}".format(asy[i], asyErr[i], chi2[i], N[i], p[i])
                if i < len(peaks) - 1:
                    s += "\n"
            return s
        
        
    def log(self, output="pyranaLog.txt"):
        f = open(output, 'w')
        f.write("########################################\n")
        f.write("#  FIT REPORT: {0}\n".format(""))
        f.write("########################################\n")
        f.write("#\n")
        f.write("# Profile name\tcenter\terror\tintensity\terror\twidth\terror\tassymmetry\terror\tchi2\tN\tprob")
        for i in range(len(self.datanames)):
            f.write("\n#\n# {0}\n".format(self.datanames[i]))
            f.write(self.str_peaks(self.fits_groups[i], typetxt='ascii'))
        f.close()
    
    
    
    
    def plot_peaks(self, X, Y, bestfit, output=None, dpi=96, ascii_output=None):
        fig, ax = plt.subplots(figsize=(8,6))
        model, p0, b = self.modelWrapper(X, bestfit)
        Yfit_tot = model(X, *p0)
        ax.plot(X, Y, linewidth=1.5, drawstyle='steps-mid', color='blue', label='data')
        ax.plot(X, Yfit_tot, linewidth=1.5, drawstyle='steps-mid', color='red', label='model')
        
        Xpeaks = np.linspace(min(X), max(X), num=1000)
        model, p0, b = self.modelWrapper(Xpeaks, [bestfit[0]])
        Ycont = model(Xpeaks, *p0)
        ax.plot(Xpeaks, Ycont, linewidth=1, color='#aa00aa', label='continuum')
        for i in range(1, len(bestfit)):
            model, p0, b = self.modelWrapper(Xpeaks, [bestfit[0], bestfit[i]])
            Yfit = model(Xpeaks, *p0)
            if i == 1: ax.plot(Xpeaks, Yfit, linewidth=1, color='#00bb00', label='components')
            else: ax.plot(Xpeaks, Yfit, linewidth=1, color='#00bb00')
        
        ax.set_xlim((min(X), max(X)))
        ax.set_xlabel("Raman shift ($\Delta$ cm$^{-1}$)", fontsize=12)
        ax.set_ylabel("Intensity (counts)", fontsize=12)
        ax.legend(fontsize=12)
        fig.tight_layout()
        if not output is None:
            fig.savefig(output, dpi=dpi)
            
        if not ascii_output is None:
            f = open(ascii_output, 'w')
            f.write("########################################\n")
            f.write("#  PLOT PEAKS: {0}\n".format(self.dataname))
            f.write("########################################\n")
            f.write("#\n")
            f.write("# Xmin={0:.2f}  Xmax={1:.2f}\n".format(min(X), max(X)))
            f.write("# fctdef = {0}\n".format(bestfit[0]))
            for i in range(1, len(bestfit)): f.write("# fctdef = {0}\n".format(bestfit[i]))
            f.write("#\n")
            f.write("# X (cm-1)\tY\tYfit")
            for i in range(len(X)):
                f.write("\n{0:.2f}\t{1:.2f}\t{2:.2f}".format(X[i], Y[i], Yfit_tot[i]))
            f.close()
        
        
    def plot_fit(self, fit, output=None, dpi=96, ascii_output=None):
        self.plot_peaks(fit['dataX'], fit['dataY'], fit['bestfit'], output=output, dpi=dpi, ascii_output=ascii_output)
    
    
    def plot_all(self, fits=None, output=None, dpi=96):
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(self.X, self.Y, linewidth=1, drawstyle='steps-mid', color='blue', label='data')
        if not fits is None:
            for i in range(len(fits)):
                model, p0, b = self.modelWrapper(fits[i]['dataX'], fits[i]['bestfit'])
                Yfit = model(fits[i]['dataX'], *p0)
                if i == 0: 
                    ax.plot(fits[i]['dataX'], Yfit, linewidth=1, drawstyle='steps-mid', color='red', label='combined models')
                else: 
                    ax.plot(fits[i]['dataX'], Yfit, linewidth=1, drawstyle='steps-mid', color='red')
        
        ax.set_xlim((min(self.X), max(self.X)))
        ax.set_xlabel("Raman shift ($\Delta$ cm$^{-1}$)", fontsize=12)
        ax.set_ylabel("Intensity (counts)", fontsize=12)
        ax.legend(fontsize=12)
        fig.tight_layout()
        if not output is None:
            fig.savefig(output, dpi=dpi)
        
        
        
    
    
if __name__ == '__main__':
    
    salsa = PyRANA()
    name = "210212_003_532_PYX110_spot3"
    salsa.loadfile_METAFILE("example/{0}.txtr.meta".format(name))
    
    peak_230 = salsa.quickfit(continuum='poly2', Xmin=170, Xmax=280,
                     peaks = [{'profile': 'assymCauchy', 'x0': 230}])

    salsa.plot_peaks(peak_230['dataX'], peak_230['dataY'], peak_230['bestfit'])
    
    peak_350 = salsa.quickfit(continuum='linear', Xmin=250, Xmax=480,
                         peaks = [{'profile': 'assymCauchy', 'x0': 334},
                                  {'profile': 'cauchy', 'x0': 378},
                                  {'profile': 'cauchy', 'x0': 397}
                                  ])
    
    salsa.plot_peaks(peak_350['dataX'], peak_350['dataY'], peak_350['bestfit'])
    
    peak_670 = salsa.quickfit(Xmin=500, Xmax=800, continuum='linear',
                            peaks=[ {'profile': 'cauchy', 'x0': 730, 's':20, 'I':100},
                                   {'profile': 'cauchy', 'x0': 660},
                                   {'profile': 'assymCauchy', 'x0': 680},
                                    ])
    
    salsa.plot_peaks(peak_670['dataX'], peak_670['dataY'], peak_670['bestfit'])
    
    peak_1000 = salsa.quickfit(Xmin=850, Xmax=1100, continuum='linear',
                            peaks=[ {'profile': 'cauchy', 'x0': 930, 's':20},
                                   {'profile': 'cauchy', 'x0': 1010},
                                   {'profile': 'cauchy', 'x0': 1030},
                                    ])
    
    salsa.plot_peaks(peak_1000['dataX'], peak_1000['dataY'], peak_1000['bestfit'])
    
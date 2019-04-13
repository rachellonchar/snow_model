import numpy as np
import matplotlib.pyplot as plt
import sys    
import os
import xlwt
import xlrd
import pandas as pd
import sklearn.linear_model
regr = sklearn.linear_model.LinearRegression()
from scipy.optimize import curve_fit
import operator

#-------------------------------------------------------------------------
#MODEL FUNCTIONS:
def avg_of_funcs(x,funcs):
    return sum([fun(x) for fun in funcs])/len(funcs)

#exponential and logistic
def func_exp(x,a,b):
    return a * np.exp(b * x) 
def func_logistic(x,c,a,b,d):
    return c/(1+a*np.exp(b*x)) + d

import numpy, scipy.optimize
#polynomial approximations
def func_poly(x, *args):
    n, poly = 0,0
    for a in args:
        poly += a*(x**n)
        n += 1
    return poly
def func_linear(x,a0,a1):
    return func_poly(x,a0,a1)
def func_poly2(x,a0,a1,a2):
    return func_poly(x,a0,a1,a2)
def func_poly3(x,a0,a1,a2,a3):
    return func_poly(x,a0,a1,a2,a3)

def func_poly2_fixed_endpoints(xy1,xy2):
    x1,y1 = xy1[0],xy1[1]
    x2,y2 = xy2[0],xy2[1]
    def polyA(x,a):
        b = (y1-a*x1**2-y2+a*x2**2)/(x1-x2)
        c = y1-a*x1**2-b*x1
        return c+b*x+a**2
    return polyA

#-----------------------------------------------------------------------------
#PLOTTING FUNCTIONS
def model_type(funct):
    classA = [func_linear, func_poly,func_poly2,func_poly3]
    if funct in classA:
        return 'poly'
    else:
        return funct

def pre_plot(tupl,fit_type='poly'):#,y_axis=None):
    if fit_type=='poly':
        strg = 'f(x) = %5.3f + %5.3fx'
        for i in range(2, len(tupl)):
            strg += ' %5.3fx^'+str(i)
        return strg % tuple(tupl)
    #elif fit_type=='linear'
    elif fit_type==func_exp:
        return 'f(x) = %5.3fexp(%5.3fx)' % tuple(tupl)
    elif fit_type==func_logistic:
        return ' %5.3f/{1 + %5.3fexp(-%5.3fx)}+%5.3f' % tuple(tupl)

def fit_2sets(X_series,Y_series, fit_func=func_linear, mask=None):
            
    X,Y = X_series, Y_series
    if type(mask)!=type(None):
        Xm = np.ma.masked_array(X,mask=mask)
        Ym = np.ma.masked_array(Y,mask=mask)
        X,Y = Xm.compressed(), Ym.compressed()
    popt, pcov = curve_fit(fit_func, X, Y)#,sigma=sigma)
    def newf(x): return fit_func(x,*popt)
    labe = pre_plot(tuple(popt),model_type(fit_func))
    dic1={}
    dic1.update({'function':newf})
    dic1.update({'parameters':popt})
    dic1.update({'print function':labe})
    return dic1

def lin_fit(X_series,Y_series, mask=None,type_return='slope'):
    X,Y = X_series, Y_series
    if type(mask)!=type(None):
        Xm = np.ma.masked_array(X,mask=mask)
        Ym = np.ma.masked_array(Y,mask=mask)
        X,Y = Xm.compressed(), Ym.compressed()
    #print(X,Y)
    slope, intercept, rvalue, pvalue, stderr = linregress(X,Y)
    if type_return=='slope':
        return slope
    elif type_return=='function':
        def newf(x): return intercept+slope*x
        return newf
    elif type_return=='function and print':
        def newf(x): return intercept+slope*x
        printf = pre_plot(tuple([intercept,slope]))
        return newf, printf

def array_span(Xob, function,dense=0,specify_points=0):
    if dense==0 and specify_points==0:
        Xspan = sorted(Xob)
        Yexp = [function(x) for x in Xob]
        return Xspan, Yexp
    else:
        pts = len(Xob) if specify_points==0 else specify_points
        x0,xN = min(Xob),max(Xob)
        Xspan = np.linspace(x0,xN,pts)
        Yexp = [function(x) for x in Xspan]
        return Xspan, Yexp

import scipy
from scipy.stats import chisquare, linregress
#def include_chi2(X_series,Y_series, fit_func=func_linear, mask=None):
    #dic1 = fit_2sets(X_series,Y_series, fit_func=fit_func, mask=mask)
    #print(linregress(X_series,Y_series))
    #popt = dic1['parameters']
    #def fun_test(xx): return popt[0]+1*popt[1]*xx
    #Y_exp = [fun_test(xx) for xx in X_series]
    #Yscip_exp = scipy.array(Y_exp)
    #X_ob,Y_ob = scipy.array(X_series), scipy.array(Y_series)
    #chi2 = 0
    #for ii in range(0,len(Y_exp)):
        #chi2 += ((Y_series[ii]-Y_exp[ii])**2)/Y_exp[ii]
    #print(chi2)
    #print(chisquare(Y_ob, f_exp=Yscip_exp))

#def r_squared(X,Y):
    #linregress(X,Y)

def distance_2pts(p1,p2):
    x1,y1 = p1[0],p1[1]
    x2,y2 = p2[0],p2[1]
    D2 = (x2-x1)**2 + (y2-y1)**2
    D = D2**(1/2)
    return D
#print(distance_2pts([0,0],[1,3]))
def line_2pts(p1,p2):
    x1,y1 = p1[0],p1[1]
    x2,y2 = p2[0],p2[1]
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m,b
def line_mpt(m,pt):
    x1,y1 = pt[0],pt[1]
    b = y1 - m*x1
    return m,b

import math
def plot_mxb(*m_b_pairs,xwindow=[0,10]):
    xar = np.linspace(xwindow[0],xwindow[1],20)
    ct = 0
    #phis = []
    for mb in m_b_pairs:
        if type(mb[0])!=type([]) and type(mb[1])!=type([]):
            m,b = mb[0],mb[1]
        else:
            if type(mb[0])!=type([]):
                m,b = line_mpt(mb[0],mb[1])
            else:
                m,b = line_2pts(mb[0],mb[1])
        yar = [m*x+b for x in xar]
        plt.plot(xar,yar,label='line '+str(ct+1))
        phi0 = math.degrees(math.atan(m))
        phi = phi0 if phi0>=0 else 360+phi0
        #print(phi)
        if ct>0:
            #print('angles between line '+str(ct)+' and line '+str(ct+1)+':')
            tot_ang = abs(phi-phi_prev)
            ang1 = tot_ang if tot_ang<=180 else tot_ang-180
            ang2 = 180-ang1
            #print(ang1,ang2)
            #print(' ')
        phi_prev = phi
        ct += 1
    plt.xlim(xmin=xwindow[0],xmax=xwindow[1])
    plt.ylim(ymin=xwindow[0],ymax=xwindow[1])
    plt.legend()
    plt.show()
        
#plot_mxb([3,5],[-1/3,4])
            

def fit_stats(X_series,Y_series, fit_func=func_linear, mask=None):
    X,Y = X_series,Y_series
    dic1 = fit_2sets(X,Y, fit_func=fit_func, mask=mask)
    #print(dic1['parameters'])
    if type(mask)!=type(None):
        Xm = np.ma.masked_array(X,mask=mask)
        Ym = np.ma.masked_array(Y,mask=mask)
        X,Y = Xm.compressed(), Ym.compressed()
    Yexp = [dic1['function'](xx) for xx in X]
    residuals = [Y[i]-Yexp[i] for i in range(0,len(X))]
    ybar = sum(Y)/len(Y)
    R2_n = [(Yexp[i]-ybar)**2 for i in range(0,len(X))]
    R2_d = [(Y[i]-ybar)**2 for i in range(0,len(X))]
    R2 = sum(R2_n)/sum(R2_d)
    #print(R2)

#observed_values=scipy.array([18,21,16,7,15])
#expected_values=scipy.array([22,19,44,8,16])

#scipy.stats.chisquare(observed_values, f_exp=expected_values)

        

#m = [1,0,0,1,1,1,0,0,1,1,0]
#x = [1,2,5,8,10,2,3,4,4,5,3]
#y = [3,4,-1,9,0,1,1,2,3,4,5]

#m = [0,0,0,0]
#x = [1,2,3,4]
#y = [1,2,3,7]
#lin_fit(x,y)

#fit_stats(x,y,fit_func=func_exp)
#print(sum(x)/4,sum(y)/4)

# with mask m
#dicc = fit_2sets(x,y,fit_func=func_linear,mask=m)
#fun = dicc['function']
#xs, yexp = array_span(x,fun,specify_points=20)
#Xm = np.ma.masked_array(x,mask=m)
#Ym = np.ma.masked_array(y,mask=m)
#plt.plot(Xm,Ym,'bo')
#plt.plot(xs,yexp,'r',label=dicc['print function'])
#plt.show()

## with mask of all 1's (no values masked)
#dicc = fit_2sets(x,y,fit_func=func_linear,mask=[0,0,0,0,0,0,0,0,0,0,0])
#fun = dicc['function']
#Xm = np.ma.masked_array(x,mask=[0,0,0,0,0,0,0,0,0,0,0])
#Ym = np.ma.masked_array(y,mask=[0,0,0,0,0,0,0,0,0,0,0])
#xs, yexp = array_span(x,fun,specify_points=20)
#plt.plot(Xm,Ym,'g.')
#plt.plot(xs,yexp,'y--',label=dicc['print function'])

## with mask=None
#dicc = fit_2sets(x,y,fit_func=func_linear)
#fun = dicc['function']
#xs, yexp = array_span(x,fun,specify_points=20)
#plt.plot(x,y,'g.')
#plt.plot(xs,yexp,'y.',label=dicc['print function'])
##plot_fit(x,y,func_linear)
#plt.legend()
#plt.show()

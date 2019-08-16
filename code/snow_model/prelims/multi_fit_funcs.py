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

#from trig_funcs import *
from scipy.stats import beta

def slope_2pts(xy1,xy2):
    x1,y1 = xy1[0],xy1[1]
    x2,y2 = xy2[0],xy2[1]
    m = (y2-y1)/(x2-x1)
    return m
def line_2pts(xy1,xy2,mb='no'):
    m = slope_2pts(xy1,xy2)
    b = xy1[1]-m*xy1[0]
    def linmxb(x): return m*x+b
    if mb=='yes':
        return m,b
    else:
        return linmxb

def interpolate_fill(x1,y1,x2):
    xys = [[x1[idx],y1[idx]] for idx in range(0,len(x1))]
    xys_sorted = sorted(xys , key=lambda k: [k[0], k[1]])
    x1,y1 = [xy[0] for xy in xys_sorted],[xy[1] for xy in xys_sorted]
    xmin,xmax = min(x1),max(x1)
    if min(x1)!=min(x2):
        xmin = max(min(x1), min(x2))
    if max(x1)!=max(x2):
        xmax = min(max(x1), max(x2))

    x11,x22 = [x for x in x1 if x>=xmin and x<=xmax],[x for x in x2 if x>=xmin and x<=xmax]
    x21 = [x for x in x22 if x not in x11]
    xflush_ = np.append(x11,x21)
    xflush = sorted(xflush_)
    yflush = np.zeros_like(xflush)
    x1prev_idx = 0
    idx = 0
    for xval in xflush:
        if xval in x1:
            xidx = x1.index(xval)
            yflush[idx] = y1[xidx]
            x1prev_idx = xidx
        else:
            #need to interpolate value
            idx1,idx2 = x1prev_idx, x1prev_idx+1
            t1,t2 = x1[idx1],x1[idx2]
            ft1,ft2 = y1[idx1],y1[idx2]
            if ft1==ft2:
                yintp = ft1
            else:
                fun = line_2pts([t1,ft1],[t2,ft2])
                yintp = fun(xval)
            yflush[idx] = yintp
        idx += 1
    return xflush,yflush
            

import shapely
from shapely.geometry import Polygon
def curve_btw(y1,y2,x=None,x1=None,x2=None,top=0,sharex=0):
    if type(x)==type(None) and type(x1)==type(None):
        x1,x2 = range(0,len(y1)),range(0,len(y1))
    if type(x1)==type(None):
        x1 = x
    if type(x2)==type(None):
        x2 = x1
    #print(type(y1),type(y2))
    if type(y1)==int or type(y1)==float:
        yfill1 = [y1 for xx in x1]
        y1 = yfill1
    if type(y2)==int or type(y2)==float:
        yfill2 = [y2 for xx in x2]
        y2 = yfill2
        
    xflush,yflush2 = interpolate_fill(x2,y2,x1)
    xflush,yflush1 = interpolate_fill(x1,y1,x2)
    if top==1: #only area where curve 1 is on top
        ynew2 = [(yflush2[idx] if yflush2[idx]<=yflush1[idx] else yflush1[idx]) for idx in range(0,len(xflush))]
        x2,y2 = xflush,ynew2
        if sharex==1:
            x1,y1 = xflush,yflush1
    elif top==2: #only area where curve 2 is on top
        ynew1 = [(yflush1[idx] if yflush1[idx]<=yflush2[idx] else yflush2[idx]) for idx in range(0,len(xflush))]
        x1,y1 = xflush,ynew1
        if sharex==1:
            x2,y2 = xflush,yflush2
    else:
        if sharex==1:
            x1,y1 = xflush,yflush1
            x2,y2 = xflush,yflush2
    return x1,x2,y1,y2
    
    
def area_between(y1,y2,x=None,x1=None,x2=None,area_type=0):
    x1,x2,y1,y2 = curve_btw(y1,y2,x=x,x1=x1,x2=x2,top=area_type)
    x_y_curve1 = [[x1[idx],y1[idx]] for idx in range(0,len(x1))]
    x_y_curve2 = [[x2[idx],y2[idx]] for idx in range(0,len(x2))]
    polygon_points = [] #creates a empty list where we will append the points to create the polygon
    for xyvalue in x_y_curve1:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1
    for xyvalue in x_y_curve2[::-1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)
    for xyvalue in x_y_curve1[0:1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon
    polygon = Polygon(polygon_points)
    area = polygon.area
    return area

def period_zero(xvalue,period_length,phase_shift=0,lower_bound='default'):#,upper_bound='def'):
    #if upper_bound=='def':
    #xvalue = phase_shift
    if lower_bound=='default':
        lower_bound = phase_shift
    upper_bound = lower_bound + (2*phase_shift + period_length/2)
    if xvalue<=lower_bound:
        iteer = period_length/2 + 2*(phase_shift)
        keep_on = True
    elif xvalue>upper_bound:
        iteer = -1*(period_length/2 + 2*(phase_shift))
        keep_on = True
    else:
        iteer = 0
        keep_on = False
    #print(iteer)
    while keep_on:
        if xvalue>upper_bound and iteer<0:
            xvalue += iteer
            if xvalue<=upper_bound and xvalue>lower_bound:
                keep_on = False
        elif xvalue<=lower_bound and iteer>0:
            xvalue += iteer
            if xvalue<=upper_bound and xvalue>lower_bound:
                keep_on = False
        else:
            print('error: final xval ',xvalue)
            keep_on = False
        #print(xvalue)
    #print(' ')
    return xvalue

def area_arrays(y1,y2,x=None,x1=None,x2=None,top=0):
    x1,x2,y1,y2 = curve_btw(y1,y2,x=x,x1=x1,x2=x2,top=top,sharex=1)
    return x1,y1,y2


def fitFunc(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[0]*x[1]

def fitFunc2(x, a, b, c):
    return a + b*x[0] + c*x[1] 

def fitFunc3(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[2]

def fitFunc4(X, a, b, c, d,e,f,g,h,i):
    x,y = X[0],X[1]
    f1 = a + b*x + c*y + d*x*y + e*x**2 + f*y**2 + g*x**2*y + h*x*y**2 + i*x**2*y**2
    return f1

def fitFunc_stoch(X, a, b, c, d,e,f,g,h,i,j):
    x,y,z = X[0],X[1],X[2]
    f1 = a + b*x + c*y + d*x*y + e*x**2 + f*y**2 + g*x**2*y + h*z**2*y + i*x**2*y**2 + j*z
    return f1

def fitFunc5(X, a, b, c, d,f,i):
    x,y,z = X[0],X[1],X[2]
    f1 = a + b*x + c*y + d*x*y + f*y**2 + i*x**2*y**2 + j*z
    return f1

def fitFunc6(X, a, b, c, d,f,i,j,k):
    x,y,z = X[0],X[1],X[2]
    f1 = a + b*x + c*y + d*x*y + f*y**2 + i*x**2*y**2 + j*z + k*x*z
    return f1

def fit_DifFrac(X, a, b, c):
    x,y,z = X[0],X[1],X[2]#,X[3]
    f1 = a + b*((x - c*y)/z) #+d*z2 + e*z2**2
    return f1

def fit_multi(*Xs,Z,fitter=fitFunc,xl=r'$x$',yl=r'$y$',zl=r'$z$',fl=r'$f$'):
    Xray = []
    idxs = []
    cull = False
    for ii in range(0,len(Z)):
        for X in Xs:
            if np.isnan(X[ii])==True or np.isinf(X[ii]):
                cull = True
        if np.isnan(Z[ii])==True or np.isinf(Z[ii]):
            cull = True
        if cull==False:
            idxs.append(ii)
        cull = False

    for X in Xs:
        Xray.append([X[ii] for ii in idxs])
    Xin,Yin = np.array(Xray),np.array([Z[ii] for ii in idxs])
    params,covs = curve_fit(fitter, Xin, Yin)
    
    a,b,c = params[0],params[1],params[2]
    try:
        d = params[3]
    except:
        None
    if fitter==fitFunc:
        def newf(xval): return fitter(xval,a,b,c,d)
        st1 = fl+'('+xl+','+yl+') = '
        tupl = ['',xl,yl,xl+yl]
        for tt in tupl:
            st1+= '%5.3f'+tt
            if tt!=tupl[-1]:
                st1 += ' + '
        string = st1 % tuple(params)
    elif fitter==fitFunc2:
        def newf(xval): return fitter(xval,a,b,c)
        st1 = fl+'('+xl+','+yl+') = '
        tupl = ['',xl,yl]
        for tt in tupl:
            st1+= '%5.3f'+tt
            if tt!=tupl[-1]:
                st1 += ' + '
        string = st1 % tuple(params)
    elif fitter==fitFunc3:
        def newf(xval): return fitter(xval,a,b,c,d)
        st1 = fl+'('+xl+','+yl+','+zl+') = '
        tupl = ['',xl,yl,zl]
        for tt in tupl:
            st1+= '%5.3f'+tt
            if tt!=tupl[-1]:
                st1 += ' + '
        string = st1 % tuple(params)
    elif fitter==fitFunc4:
        e,f,g,h,i = params[4],params[5],params[6],params[7],params[8]
        def newf(xval): return fitter(xval,a,b,c,d,e,f,g,h,i)
        st1 = fl+'('+xl+','+yl+') = '
        tupl = ['',xl,yl,xl+yl, xl+r'$^2$',yl+r'$^2$',xl+r'$^2$'+yl,xl+yl+r'$^2$',xl+r'$^2$'+yl+r'$^2$']
        for tt in tupl:
            st1+= '%5.3f'+tt
            if tt==tupl[3]:
                st1 += '\n'
            if tt!=tupl[-1]:
                st1 += ' + '
        string = st1 % tuple(params)
    elif fitter==fitFunc6:
        f,i,j,k = params[4],params[5],params[6],params[7]#,params[8],params[9]
        def newf(xval): return fitter(xval,a,b,c,d,f,i,j,k)
        st1 = fl+'('+xl+','+yl+','+zl+') = '
        #tupl = ['',xl,yl,xl+yl, xl+r'$^2$',yl+r'$^2$',xl+r'$^2$'+yl,yl+r'$^2$'+zl,xl+r'$^2$'+yl+r'$^2$',zl]
        tupl = ['',xl,yl,xl+yl,yl+'2',xl+'2'+yl+'2',zl,xl+yl+zl]
        for tt in tupl:
            st1+= '%5.3f'+tt
            if tt==tupl[3]:
                st1 += '\n'
            if tt!=tupl[-1]:
                st1 += ' + '
        string = st1 % tuple(params)
    elif fitter==fitFunc_stoch:
        e,f,g,h,i,j = params[4],params[5],params[6],params[7],params[8],params[9]
        def newf(xval): return fitter(xval,a,b,c,d,e,f,g,h,i,j)
        st1 = fl+'('+xl+','+yl+','+zl+') = '
        #tupl = ['',xl,yl,xl+yl, xl+r'$^2$',yl+r'$^2$',xl+r'$^2$'+yl,yl+r'$^2$'+zl,xl+r'$^2$'+yl+r'$^2$',zl]
        tupl = ['',xl,yl,xl+yl, xl+'2',yl+'2',xl+'2'+yl,zl+'2'+yl,xl+'2'+yl+'2',zl]
        for tt in tupl:
            st1+= '%5.3f'+tt
            if tt==tupl[3]:
                st1 += '\n'
            if tt!=tupl[-1]:
                st1 += ' + '
        string = st1 % tuple(params)
    elif fitter==fit_DifFrac:
        #e = params[4] #,params[5],params[6],params[7],params[8]
        def newf(xval): return fitter(xval,a,b,c)
        st1 = fl+'('+xl+','+yl+') = '
        string_ = st1 + '%5.3f + '+'%5.3f('+xl+'%5.3f'+yl+')/'+zl #+' + %5.3%fv + %5.3f'+'v^2'#+r'v$^2$' 
        string = ''#string_ % tuple(params)
    dic = {}
    dic.update({'function':newf})
    dic.update({'print function':string})
    dic.update({'parameters':params})
    return dic
    
    #print(params)
    #fitParams, fitCovariances = curve_fit(fitFunc, x_3d, x_3d[1,:], p0)
    

#X1,X2,Z = [1,2,3,0,1],[6,7,8,1,0],[22,35,50,3,4]
#fit_multi(X1,X2,Z=Z)
#print(' fit coefficients:\n', fitParams)

















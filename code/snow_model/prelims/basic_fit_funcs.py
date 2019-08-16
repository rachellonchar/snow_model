from multi_fit_funcs import *

#-------------------------------------------------------------------------
#MODEL FUNCTIONS:
def avg_of_funcs(x,funcs):
    return sum([fun(x) for fun in funcs])/len(funcs)

#exponential and logistic
def func_exp(x,a,b):
    return a * np.exp(b * x) 
def func_log(x,a,b):
    return a * np.log(b * x) 
def func_logistic(x,c,a,b,d):
    return c/(1+a*np.exp(b*x)) + d

def better_logistic(x,a,x50,c):
    return c/(1 + np.exp(a*(x-x50)))
def CC_logistic_set(cc=1):
    def fun(x,a,x50): return 1/(1 + np.exp(a*(x-x50)))
    return fun

def beta_pdf(x,a,b):
    return x**(a)*(1-x)**(b)
    #return c*beta.pdf(x, a, b)
def func_logger(x,a,b,c):
    return a*np.log(x+c)/np.log(b)

def func_exp_spf(a):
    def newb(x,b): return func_exp(x,a,b)
    return newb

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
    
def func_linear_0intercept(x,m):
    return x*m
    
def func_poly2(x,a0,a1,a2):
    return func_poly(x,a0,a1,a2)
def func_poly3(x,a0,a1,a2,a3):
    return func_poly(x,a0,a1,a2,a3)

def func_quad1(p1,p2):
    t1,l1 = p1[0],p1[1]
    t2,l2 = p2[0],p2[1]
    def quad(x,a):
        b = (l2-l1-a*(t2**2-t1**2))/(t2-t1)
        c = l1-a*t1**2-b*t1
        return a*x**2+b*x+c
    return quad
    
def func_quad2(p1):
    t1,l1 = p1[0],p1[1]
    def quad(x,b,a):
        c = l1-a*t1**2-b*t1
        return a*x**2+b*x+c
    return quad

def func_lin1(p1):
    t1,l1 = p1[0],p1[1]
    def quad(x,m):
        b = l1 - m*t1
        return b + m*x
    return quad

def func_poly2_EX(x_fix):
    def funn(x,a,c): return a+c*(-2*x_fix*x + x**2)
    return funn

def func_poly2_fixed_endpoints(xy1,xy2):
    x1,y1 = xy1[0],xy1[1]
    x2,y2 = xy2[0],xy2[1]
    def polyA(x,a):
        b = (y1-a*x1**2-y2+a*x2**2)/(x1-x2)
        c = y1-a*x1**2-b*x1
        return c+b*x+a**2
    return polyA

def func_sin(x,a,b,c,d):
    return a*np.sin(b*x + c) + d

def lin_3var(X,a,b,c,d):
    x,y,z = X[0],X[1],X[2]
    return a*x + b*y + c*z + d

def lin_2var(X,a,b,d):
    x,y = X[0],X[1]
    return a*x + b*y + d

def func_slope(b):
    def ff(x,m): return m*x+b
    return ff




#-----------------------------------------------------------------------------
#PLOTTING FUNCTIONS
def model_type(funct):
    classA = [func_linear, func_poly,func_poly2,func_poly3,func_quad1,func_quad2]
    #classB = [func_linear_0intercept, func_exp, func_log, func_logistic, func_sin]
    if funct in classA:
        return 'poly'
    else:
        return funct

def pre_plot(tupl,fit_type='poly',x_fix=None):#,y_axis=None):
    if fit_type=='poly':
        strg = r'$f(x)$ = %5.3f + %5.3f'+r'$x$'
        for i in range(2, len(tupl)):
            strg += r'+%5.5f$x^'+str(i)+'$'
        return strg % tuple(tupl)
    elif fit_type==func_linear_0intercept:
        return 'f(x) = %5.3fx' % tuple(tupl)
    elif fit_type==func_exp:
        return 'f(x) = %5.3fexp(%5.3fx)' % tuple(tupl)
    elif fit_type==func_log:
        return 'f(x) = %5.3flog(%5.3fx)' % tuple(tupl)
    elif fit_type==func_logistic:
        return ' %5.3f/{1 + %5.3fexp(-%5.3fx)}+%5.3f' % tuple(tupl)
    elif fit_type==better_logistic:
        return ' 1/{1 + exp(%5.3f(x-%5.3f))}*%5.3f' % tuple(tupl)
    elif fit_type==func_sin:
        return 'f(x) = %5.3fsin(%5.3fx + %5.3f) + %5.3f' % tuple(tupl)
    elif fit_type==lin_3var:
        return 'f(x,y,z) = %5.3fx + %5.3fy + %5.3fz + %5.3f' % tuple(tupl)
    elif fit_type==func_logger:
        return ' %5.3f/{1 + %5.3fexp(-%5.3fx)}' % tuple(tupl)
        
    else:
        prints = ''
        for t in tupl:
            sp = ', ' if t!=tupl[-1] else ''
            prints += str(round(t,7))+sp
        return prints
        #if x_fix==None:
            #x_fix = 'xEX'
        #p1 = 'f(x) = %5.3f + \n%5.5f(-2(' % tuple(tupl)
        #return p1 + str(x_fix)+')x + x^2'

def mult_masks(*masks):
    #inc_mask = np.zeros_like(masks[0])

    inc_mask = None
    for mask in masks:
        if type(mask)!=None:
            inc_mask = np.zeros(len(mask))
    if type(inc_mask)==type(None):
        None
    else:
        for mask in masks:
            for idx in range(0,len(inc_mask)):
                if mask[idx]==1:
                    inc_mask[idx] = 1
    #print(inc_mask)
    return inc_mask

def series_cleanup(X_series,Y_series,mask=None,inf_threshold=100000,inf_concern=1,
    mask_nan_return=0):
    it=inf_threshold
    indices = np.logical_not(np.logical_or(np.isnan(X_series), np.isnan(Y_series)))
    #print(indices)
    Xnan = [(X_series[idx] if indices[idx]==True else 0) for idx in range(0,len(indices)) ]
    Ynan = [(Y_series[idx] if indices[idx]==True else 0) for idx in range(0,len(indices))  ]
    idx_not_nan = [idx for idx in range(0,len(indices)) if indices[idx]==True ]

    if inf_concern==1:
        good_idxs = [idx for idx in range(0,len(indices)) if Xnan[idx]>-it and Xnan[idx]<it and Ynan[idx]>-it and Ynan[idx]<it ]
        inf_mask = [(0 if idx in good_idxs else 1) for idx in range(0,len(indices))]
        X = [(Xnan[idx] if inf_mask[idx]==0 else 0) for idx in range(0,len(indices)) ]
        Y = [(Ynan[idx] if inf_mask[idx]==0 else 0) for idx in range(0,len(indices))  ]
    else:
        X,Y = Xnan,Ynan
        inf_mask = [0 for idx in range(0,len(indices))]
    
    nan_mask = [(0 if (inf_mask[idx]==0 and indices[idx]==True) else 1) for idx in range(0,len(indices))]
    #Ynan = [Y_series[idx] for idx in range(0,len(indices)) if indices[idx]==True ]
    #print(mask)
    if type(mask)!=type(None):
        mas = [mask[idx] for idx in range(0,len(indices)) if indices[idx]==True ]
        Xm = np.ma.masked_array(X,mask=mas)
        Ym = np.ma.masked_array(Y,mask=mas)
        X,Y = Xm.compressed(), Ym.compressed()
    if mask_nan_return==1:
        return nan_mask
    return X,Y

#series_cleanup([1,2,3,4],[float('inf'),0,0,1])

def lin_fit(X_series,Y_series, mask=None,type_return='slope'):

    X,Y = series_cleanup(X_series,Y_series,mask=mask)
    if type(mask)!=type(None):
        mas = [mask[idx] for idx in range(0,len(indices)) if indices[idx]==True ]
        Xm = np.ma.masked_array(X,mask=mas)
        Ym = np.ma.masked_array(Y,mask=mas)
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

def res(Ydata,Yexpected):
    res = [(Ydata[idx]-Yexpected[idx])**2 for idx in range(0,len(Ydata)) ]
    avg = sum(Ydata)/len(Ydata)
    tot = [(Ydata[idx]-avg)**2 for idx in range(0,len(Ydata)) ]
    R2 = 1 - sum(res)/sum(tot)
    return R2

def fit_2sets(X_series,Y_series, fit_func=func_linear,func_type=None,labe_func='default',sp_val=None, 
    mask=None,clipped=0,pvalue=0,
    Amp=None,period=None,phase_shift=None,y_shift=None,point=None):
    
    #print(len(Y_series))
    if fit_func==lin_3var:
        #print('enter')
        mask1 = series_cleanup(Y_series,X_series[0],mask=mask,mask_nan_return=1)
        #print('exit')
        mask2 = series_cleanup(Y_series,X_series[1],mask=mask,mask_nan_return=1)
        mask3 = series_cleanup(Y_series,X_series[2],mask=mask,mask_nan_return=1)
        #print('3 too babe')
        nan_mask = mult_masks(mask1,mask2,mask3)
        Ym = np.ma.masked_array(Y_series,mask=nan_mask)
        Xm1 = np.ma.masked_array(X_series[0],mask=nan_mask)
        Xm2 = np.ma.masked_array(X_series[1],mask=nan_mask)
        Xm3 = np.ma.masked_array(X_series[2],mask=nan_mask)
        Y,X1,X2,X3 = Ym.compressed(), Xm1.compressed(), Xm2.compressed(), Xm3.compressed()
        #X = [[X1[idx],X2[idx],X3[idx]] for idx in range(0,len(X1))]
        #print(len(X1),len(X2),len(X3),len(Y),len(X))
        X = [X1,X2,X3]
    elif fit_func==lin_2var:
        #print('enter')
        mask1 = series_cleanup(Y_series,X_series[0],mask=mask,mask_nan_return=1)
        mask2 = series_cleanup(Y_series,X_series[1],mask=mask,mask_nan_return=1)
        #mask3 = series_cleanup(Y_series,X_series[2],mask=mask,mask_nan_return=1)
        #print('3 too babe')
        nan_mask = mult_masks(mask1,mask2)
        Ym = np.ma.masked_array(Y_series,mask=nan_mask)
        Xm1 = np.ma.masked_array(X_series[0],mask=nan_mask)
        Xm2 = np.ma.masked_array(X_series[1],mask=nan_mask)
        #Xm3 = np.ma.masked_array(X_series[2],mask=nan_mask)
        Y,X1,X2 = Ym.compressed(), Xm1.compressed(), Xm2.compressed(), Xm3.compressed()
        #X = [[X1[idx],X2[idx],X3[idx]] for idx in range(0,len(X1))]
        #print(len(X1),len(X2),len(X3),len(Y),len(X))
        X = [X1,X2,X3]
    else:
        #print('ERRRROOORRRRRRRRRRR')
        X,Y = series_cleanup(X_series,Y_series,mask=mask)
        #print('   ',len(X),len(Y))
    fix_pop = 0
    if fit_func==func_sin_def or fit_func==func_sin:
        fit_func = func_sin_def(Amp=Amp,period=period,phase_shift=phase_shift,y_shift=y_shift,point=point)
        fix_pop = 1
    #print(len(X),len(Y))
    popt, pcov = curve_fit(fit_func, X, Y)#,sigma=sigma)
    if fix_pop==1:
        popt = sin_popper(popt,Amp=Amp,period=period,phase_shift=phase_shift,y_shift=y_shift,point=point)
        fit_func = func_sin
    def newf(x): return fit_func(x,*popt)
    #if labe_func=='default' and type(sp_val)==type(None):
        #labe = pre_plot(tuple(popt),model_type(fit_func))
        
    if func_type=='fixed end quad':
        if len(popt)==2:
            c = newf(0)
            ppop = [c,popt[0],popt[1]]
            md_t = 'poly'
        elif len(popt)==1:
            c = newf(0)
            yy = newf(1)
            b = (yy - c - popt[0])
            ppop = [c,b,popt[0]]
            md_t = 'poly'
        else:
            ppop = popt
            md_t = model_type(fit_func)
    elif func_type=='fixed end lin':
        b = newf(0)
        ppop = [b,popt[0]]
        md_t = 'poly'
    else:
        ppop = popt
        md_t = model_type(fit_func)
        
    labe = pre_plot(tuple(ppop),md_t)
    #else:
        #if type(sp_val)==type(None):
            #newpop = popt
        #else:
            #newpop = np.append([sp_val],popt)
        #labe = pre_plot(tuple(newpop),model_type(labe_func))
    dic1={}
    dic1.update({'function':newf})
    dic1.update({'parameters':ppop})
    #print(popt)
    dic1.update({'print function':labe})
    if clipped==1:
        dic1.update({'X':X})
        dic1.update({'Y':Y})
        if fit_func==lin_3var:
            dic1.update({'Yexp':[newf([X[0][idx],X[1][idx],X[2][idx]]) for idx in range(0,len(Y))]})
        else:
            dic1.update({'Yexp':[newf(xx) for xx in X]})
        #dic1.update({'Yexp':[newf(xx) for xx in X]})
    else:
        dic1.update({'X':X_series})
        dic1.update({'Y':Y_series})
        if fit_func==lin_3var:
            dic1.update({'Yexp':[newf([X_series[0][idx],X_series[1][idx],X_series[2][idx]]) for idx in range(0,len(Y_series))]})
        else:
            dic1.update({'Yexp':[newf(xx) for xx in X_series]})
        #dic1.update({'Yexp':[newf(xx) for xx in X_series]})
        #dic1.update({'Yexp':[newf(xx) for xx in X_series]})
    r2 = res(dic1['Y'],dic1['Yexp'])
    #print('R2',len(dic1['Y']),len(dic1['Yexp']))
    dic1.update({'R2':r2})
    #print(len(X),len(Y))
    if pvalue==1:
        #try:
            #logXs = [np.log(xx) for xx in X_series]
            #logX,Y = series_cleanup(logXs,Y_series,mask=mask,inf_concern=1)
            #slope, intercept, rvalue, pvalue, stderr = linregress(logX,Y)
            #dic1.update({'log pvalue':pvalue})
        #except:
            #None
        #print(len(X),len(Y))
        slope, intercept, rvalue, pvalue, stderr = linregress(X,Y)
        dic1.update({'pvalue':pvalue})
    return dic1

def array_span(Xob, function,pts=30):
    #pts = len(Xob) if specify_points==0 else specify_points
    x0,xN = min(Xob),max(Xob)
    Xspan = np.linspace(x0,xN,pts)
    Yexp = [function(x) for x in Xspan]
    return Xspan, Yexp

#def expected_Y(X,Y,fit_type=func_linear,mask=None,clipped=0):
    
    #fic = fit_2sets(X,Y, fit_func=fit_type, mask=mask,clipped=clipped)
    #dicT = fic
    #if fit_type == func_exp:
        #popt = fic['parameters']
        #lin_labe = pre_plot(popt) #default model is polynomial
        #def lin_fun(xx): return np.log(popt[0])+popt[1]*xx
        #lin_popt = tuple((np.log(popt[0]),popt[1]))
        #dicT.update({'log function':lin_fun})
        #dicT.update({'print log function':lin_labe})
        #dicT.update({'log parameters':lin_popt})
        #Yln_exp = [lin_fun(xx) for xx in fic['X']]
        #dicT.update({'log Yexp':Yln_exp})
        
        #Xsp,log_Ysp = array_span(dicT['X'],dicT['log function'])
        #dicT.update({'log Ysexp':log_Ysp})
        #lnY = [np.log(y) for y in dicT['Y']]
        #dicT.update({'log Y':lnY})
    #Xsp,Ysp = array_span(dicT['X'],dicT['function'])
    #dicT.update({'Xs':Xsp})
    #dicT.update({'Ysexp':Ysp})
    #return dicT

#def mask_array(array,mask):
    #array1 = np.ma.masked_array(array,mask=mask)
    #return array1.compressed()

#from scipy import stats
#def bin_stat(Xseries,Yseries,mask=None,stat_type='mean',bins=15):
    #if type(mask)==type(None):
        #X,Y = Xseries,Yseries
    #else:
        #X = mask_array(Xseries,mask)
        #Y = mask_array(Yseries,mask)
    #bin_means, bin_edges, binnumber = stats.binned_statistic(X,
        #Y, statistic=stat_type, bins=bins)
    #return bin_means, bin_edges, binnumber
##plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
    ##label='binned by '+stat_type)
 
#def deviations_from_fit(X,Y,fit_type=func_linear,mask=None,clipped=0,clipped_devs=0):
    #fit_dic = expected_Y(X=X,Y=Y,fit_type=fit_type,mask=mask,clipped=clipped)
    #if fit_type==func_exp:
        #Y, lnYex = fit_dic['Y'],fit_dic['log Yexp']
        #dev = [Y[ii] - np.exp(lnYex[ii]) for ii in range(0,len(Y))]
        ##lin_dev = [np.log(Y[ii] - np.exp(lnYex[ii])) for ii in range(0,len(Y))]
    #else:
        #Y, Yex = fit_dic['Y'],fit_dic['Yexp']
        #dev = [Y[ii] - Yex[ii] for ii in range(0,len(Y))]
        
    #if clipped_devs==1 and clipped==1:
        #devs = mask_array(dev,mask)
    #else:
        #devs = dev
    #fit_dic.update({'deviations':devs})
    #if fit_type==func_exp:
        #fit_dic.update({'linear deviations':[np.log(dd) for dd in devs]})
    #return fit_dic

#import scipy
#from scipy.stats import chisquare, linregress
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

#def distance_2pts(p1,p2):
    #x1,y1 = p1[0],p1[1]
    #x2,y2 = p2[0],p2[1]
    #D2 = (x2-x1)**2 + (y2-y1)**2
    #D = D2**(1/2)
    #return D
##print(distance_2pts([0,0],[1,3]))
#def line_2pts(p1,p2):
    #x1,y1 = p1[0],p1[1]
    #x2,y2 = p2[0],p2[1]
    #m = (y2-y1)/(x2-x1)
    #b = y1 - m*x1
    #return m,b
#def line_mpt(m,pt):
    #x1,y1 = pt[0],pt[1]
    #b = y1 - m*x1
    #return m,b

#import math
#def plot_mxb(*m_b_pairs,xwindow=[0,10]):
    #xar = np.linspace(xwindow[0],xwindow[1],20)
    #ct = 0
    ##phis = []
    #for mb in m_b_pairs:
        #if type(mb[0])!=type([]) and type(mb[1])!=type([]):
            #m,b = mb[0],mb[1]
        #else:
            #if type(mb[0])!=type([]):
                #m,b = line_mpt(mb[0],mb[1])
            #else:
                #m,b = line_2pts(mb[0],mb[1])
        #yar = [m*x+b for x in xar]
        #plt.plot(xar,yar,label='line '+str(ct+1))
        #phi0 = math.degrees(math.atan(m))
        #phi = phi0 if phi0>=0 else 360+phi0
        ##print(phi)
        #if ct>0:
            ##print('angles between line '+str(ct)+' and line '+str(ct+1)+':')
            #tot_ang = abs(phi-phi_prev)
            #ang1 = tot_ang if tot_ang<=180 else tot_ang-180
            #ang2 = 180-ang1
            ##print(ang1,ang2)
            ##print(' ')
        #phi_prev = phi
        #ct += 1
    #plt.xlim(xmin=xwindow[0],xmax=xwindow[1])
    #plt.ylim(ymin=xwindow[0],ymax=xwindow[1])
    #plt.legend()
    #plt.show()
        
##plot_mxb([3,5],[-1/3,4])
            

#def fit_stats(X_series,Y_series, fit_func=func_linear, mask=None):
    #X,Y = X_series,Y_series
    #dic1 = fit_2sets(X,Y, fit_func=fit_func, mask=mask)
    ##print(dic1['parameters'])
    #if type(mask)!=type(None):
        #Xm = np.ma.masked_array(X,mask=mask)
        #Ym = np.ma.masked_array(Y,mask=mask)
        #X,Y = Xm.compressed(), Ym.compressed()
    #Yexp = [dic1['function'](xx) for xx in X]
    #residuals = [Y[i]-Yexp[i] for i in range(0,len(X))]
    #ybar = sum(Y)/len(Y)
    #R2_n = [(Yexp[i]-ybar)**2 for i in range(0,len(X))]
    #R2_d = [(Y[i]-ybar)**2 for i in range(0,len(X))]
    #R2 = sum(R2_n)/sum(R2_d)
    ##print(R2)







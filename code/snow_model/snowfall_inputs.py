
#regular code directory setup:
#import sys, os, os.path
#cwd = os.getcwd()
##main_dirc = cwd.split('peat_project', 1)[0]
##cwd_code = main_dirc + 'peat_project/code'
#sys.path.insert(0, cwd+'/prelims')
#from save_funcs import *
#import basic_fit_funcs as btf
#main_dirc = cwd.split('peat_project', 1)[0]+'peat_project/' #change variable back
import sys, os, os.path
cwd = os.getcwd()
main_dirc = cwd.split('snow_main', 1)[0]
cwd_code = main_dirc + 'snow_main/code/snow_model'
sys.path.insert(0, cwd_code+'/prelims')
from save_funcs import *
import basic_fit_funcs as btf

def slope_2pts(xy1,xy2):
    x1,y1 = xy1[0],xy1[1]
    x2,y2 = xy2[0],xy2[1]
    m = (y2-y1)/(x2-x1)
    return m
def line_2pts(xy1,xy2,mb='no'):
    m = slope_2pts(xy1,xy2)
    b = xy1[1]-m*xy1[0]
    #print(m,b)
    def linmxb(x): return m*x+b
    if mb=='yes':
        return m,b
    else:
        return linmxb
#----------------------------------------------------------------------------------------
def bulb_temp(RH,Tair):
    #empirical relationship --> Roland Stull, 2011
    l1 = Tair*np.arctan(0.151977*(RH+8.313659)**(1/2)) + np.arctan(Tair+RH)
    l2 = -np.arctan(RH-1.676331) + 0.00391838*RH**(3/2)*np.arctan(0.023101*RH) - 4.686035
    Tbulb = l1+l2
    return Tbulb

def snow_fraction(Tw):
    if Tw<1.1:
        sf = 1 - 0.5*np.exp(-2.2*(1.1-Tw)**(1.3))
    else:
        sf = 0.5*np.exp(-2.2*(Tw-1.1)**(1.3))
    return sf

def rain_snow_amount(X,Tair,RH):
    Tb = bulb_temp(RH,Tair)
    if X>0:
        SnowFrac = snow_fraction(Tb)
        Xs = SnowFrac*X
        if Xs<.0000001:
            Xs = 0
        Xr = X - Xs
    else:
        Xs,Xr = 0,0
    return Xs,Xr,Tb
#----------------------------------------------------------------------------------------

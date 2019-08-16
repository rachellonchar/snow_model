
from snowfall_inputs import *

# Sensible and latent heat
#--------------------
def resistance_of_heat_transfer(u):
    zu,zT,d = 2,1,0
    zm,zh,k = .001,.0002,.41
    rh = np.log((zu - d + zm)/zm)*np.log((zT - d + zh)/zh)/(k**2*u)*1/86400
    return rh
def wind_speed_2m(uh,h):
    u2 = uh*(4.87/(np.log(67.8*h-5.42))) #m/s
    return u2
    
def fun_H(uh,h,Ts,Ta):
    Ca = 1.29 # heat capacity of air
    u = wind_speed_2m(uh,h)
    rh = resistance_of_heat_transfer(u)
    H = Ca*(Ts - Ta)/rh
    return H

def fun_LE(uh,h,Ts,Tmin):
    R = 0.4615
    lam_v = 2500
    u = wind_speed_2m(uh,h)
    rv = resistance_of_heat_transfer(u)
    
    # saturation vapor density
    def rho0(T): 
        p0 = np.exp((16.78*T-116.8)/(T+273.3))*1/((273.15+T)*R)
        return p0
    rho_s = rho0(Ts)
    rho_a = rho0(Tmin)
    
    E = lam_v*(rho_s-rho_a)/rv
    return E

# Solar Radiation
#--------------------
def S_not(a,doy):
    # solar declination
    g = 0.4102*math.sin(2*np.pi/365 * (doy-80))
    Sprime = 117.5*10**3 # kJ/(m2 day)

    s1d = math.acos(-math.tan(g)*math.tan(a))
    s1 = s1d * math.sin(g)*math.sin(a)
    
    s2d = math.acos(-math.tan(g)*math.tan(a))
    s2 = math.cos(g)*math.cos(a) *math.sin(s2d)
    
    So = Sprime/np.pi *(s1 + s2)
    return So

def atmospheric_transmissivity(a,Tmin,Tmax,RH,doy):
    Tb = bulb_temp(RH,Tmin)
    if Tb<1.3: # (winter albedo)
        B = 0.170*a**(-0.979)
    else:
        B = 0.282*a**(-0.431)
        
    doy_ = doy-30
    if doy_<1:
        doy_ += 365
    So30 = S_not(a,doy_)
    #So30 = extraterrestrial_radiation(a,doy_)
    
    Tt = 0.75*(1+ np.exp(-B/So30 *(Tmax-Tmin)**2))
    return Tt

def albedo(X,A_i,Ta,RH,fixed_albedo=False):
    Xs,Xr,Tb = rain_snow_amount(X,Ta,RH)
    Ax = 0.95 #max albedo
    
    # albedo after snowfall
    if Xs>0:
        rho_star = 50 + 3.4*(Ta+15)
        rho_sn = rho_star if rho_star<50 else 50 # density of snow
        A = Ax - (Ax - A_i)*np.exp(-(4*Xs*rho_sn)/0.12)
    else:
        if fixed_albedo==False:
            if A_i>0.35:
                A = 0.35 - (0.35 - Ax)*np.exp(-(0.177 + np.log((Ax-0.35)/(A_i-0.35))**(2.16)))**(0.46)
            else:
                A = A_i
        else:
            A = fixed_albedo
    return A

def fun_Rs(X,A_i,Ta,RH,doy,Tmin,Tmax,
    fixed_albedo=False,lattitude_loc=47.50505):#[47,32,0]): 
        
    if type(lattitude_loc)==list:
        ll = lattitude_loc
        loc = ll[0] + ll[1]/60 + ll[2]/3600
    else:
        loc = lattitude_loc
    a = np.deg2rad(loc)
    
    #if fixed_albedo:
        #A = A_i
    #else:
    A = albedo(X,A_i,Ta,RH,fixed_albedo=fixed_albedo)
    Tt = atmospheric_transmissivity(a,Tmin,Tmax,RH,doy)
    So = S_not(a,doy)
    #So = extraterrestrial_radiation(a,doy)
    #print(A,Tt,So)
    S = (1-A)*Tt*So
    return A,S,So

# Longwave Radiation
#--------------------
def sat_vapor_pressure(T):
    eT = 0.6108*np.exp(17.27*T/(T+237.3)) # IS THIS WRONG?? 
    return eT
    
def vapor_pressure(RH,Tmin,Tmax):
    eTmin = sat_vapor_pressure(Tmin)
    eTmax = sat_vapor_pressure(Tmax)
    ea = RH/100*((eTmin + eTmax)/2)
    return ea

def extraterrestrial_radiation(a,doy):
    
    Gsc = 0.0820
    dr = 1 + 0.33*math.cos(2*np.pi/365 *doy)
    g = 0.409*math.sin(2*np.pi/365 *doy -1.39)
    ws = math.acos(-math.tan(a)*math.tan(g))
    
    Ra = 24*60/np.pi *Gsc*dr*((ws*math.sin(a)*math.sin(g)) + (math.sin(ws)*math.cos(a)*math.cos(g)))
    return Ra*1000 # kJ/(m2 day)
    
def fun_Ln(X,A_i,Ta,RH,doy,Tmin,Tmax,Ts,
    fixed_albedo=False,lattitude_loc=47.50505,
    elevation=422): # elev in (m)
    
    A,S,So = fun_Rs(X,A_i,Ta,RH,doy,Tmin,Tmax,
        fixed_albedo=fixed_albedo,lattitude_loc=lattitude_loc)
    Rs = S/(1-A) # kJ 
    
    ##----
    #if type(lattitude_loc)==list:
        #ll = lattitude_loc
        #loc = ll[0] + ll[1]/60 + ll[2]/3600
    #else:
        #loc = lattitude_loc
    #a = np.deg2rad(loc)
    ##print(a,loc*np.pi/180)
    #Ra = So#extraterrestrial_radiation(a,doy)
    ##print(So,Ra)
    #Rso = (0.75 + 2*10**(-5)*elevation)*Ra
    
    #Boltz = 4.89*10**(-11) # kJ/(m2 K4)
    #ea = vapor_pressure(RH,Tmin,Tmax)
    #p1 = ((Tmax+273.16)**4 + (Tmin+273.16)**4)/2
    #p2 = 0.34 - 0.14*ea**(1/2)
    #p3 = 1.35*Rs/Rso - 0.35
    ## (step 18)
    #Rnl = Boltz*p1*p2*p3 # kJ/(m2 day)
    
    TaK,TsK = Ta+273.15, Ts+273.15 # convert to Kelvin
    sb_constant = 4.89*10**(-11) # kJ/(m2 K4)
    # fraction cloud coverage 
    if X > 0:
        fcc = 1
    else:
        fcc = 0
        
    eps_a = (0.72 + 0.005*Ta)*(1 - 0.84*fcc) + 0.84*fcc
    eps_t = 0.97
    
    La = eps_a*sb_constant*TaK**4
    if Ts>0:
        Lt = eps_t*sb_constant*TsK**4
    else:
        Lt = 0
    Rnl = La - Lt
    #print('L',La,Lt,L)
    #return L
    return A,S,Rnl # kJ/(m2 day)

def fun_Rnet(X,A_i,Ta,RH,doy,Tmin,Tmax,Ts,
    fixed_albedo=False,lattitude_loc=47.50505,
    elevation=422):
    
    A,Rns,Rnl = fun_Ln(X,A_i,Ta,RH,doy,Tmin,Tmax,Ts,
        fixed_albedo=fixed_albedo,lattitude_loc=lattitude_loc,
        elevation=elevation)
    return A, Rns, Rnl, Rns - Rnl
    
    #TaK,TsK = Ta+273.15, Ts+273.15 # convert to Kelvin
    #sb_constant = 4.89*10**(-11) # kJ/(m2 K4)
    ## fraction cloud coverage 
    #if precipitation_depth > 0.5/100:
        #fcc = 1
    #else:
        #fcc = 0
        
    #eps_a = (0.72 + 0.005*Ta)*(1 - 0.84*fcc) + 0.84*fcc
    #eps_t = 0.97
    
    #La = eps_a*sb_constant*TaK**4
    #if Ts>0:
        #Lt = eps_t*sb_constant*TsK**4
    #else:
        #Lt = 0
    #L = La - Lt
    ##print('L',La,Lt,L)
    #return L

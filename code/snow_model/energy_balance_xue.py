
from energy_inputs import *
#import basic_fit_funcs as btf
#main_dirc = cwd.split('peat_project', 1)[0]+'peat_project/' #change variable back

# https://www.eoas.ubc.ca/courses/atsc113/snow/met_concepts/07-met_concepts/07b-newly-fallen-snow-density/
#rho_s0l,rho_s0h = 50,100 # kg/m3    fresh falling snow
#rho_s1l,rho_s1h = 100,200 # new top snow, uncompacted
#rho_s2l,rho_s2h = 200,300 # settled snow, self compacted after several days

#def porosity(pore_volume,total_volume):
    #return pore_volume/total_volume

def characteristic_slope(z,water_volume,pore_volume,t0=0,n=3,gam=7.15,
    alpha=5.47*10**6, d = 1.3*10**(-3), Swi=0.07, por0 = 7.40734483*10**(-3),
    c = 2*6.30755824*10**(-6),rho_ice=917,rho_w=1000,rho_snow=50,):
    
    # water saturation as a fraction of pore volume
    Sw = water_volume/pore_volume
    #print(Sw)
    # effective saturation
    Se = (Sw - Swi)/(1 - Swi)
    # unsaturated or relative permeability to water [m2]
    beta = 6.0*10**(-5)*d**2
    phi = por0 - c*z
    phi_e = phi*(1 - Swi)
    k = beta*np.exp(gam*(phi))
    #print(k)
    kw = k*Se**n
    # noncapillary flow --> volume flux of water [m3/(m2s)]
    u = alpha*kw
    
    # ---
    t_t0 = n/(gam*c)*np.exp(-gam*por0/n)*(np.exp(gam*c*z/n)*(por0 - c*z + n/gam) - por0 - n/gam)*(1-Swi)/(n*(alpha*beta)**(1/n))*u**((1-n)/n) + t0
    if np.isnan(t_t0):
        t_t0 = 0
        kg_flux = 0
        #rho_snow = 75
    else:
        dz_dt_at_u = n*alpha**(1/n)*u**((n-1)/n) *k**(1/n)/phi_e
        z_t_t0 = dz_dt_at_u*t_t0
        # density of snow
        #rho_snow = 262#rho_ice*(1-phi) + rho_w*Sw*phi
        #print(rho_snow,phi,Sw)
        uK = u*rho_snow
        kg_flux = uK*t_t0
    #print(kg_flux,t_t0,rho_snow)
    return kg_flux, t_t0, rho_snow

def loop_melt(snowpack_depth,trapped_water_kg,water_volume,pore_volume,t0=0,n=3,gam=7.15,
    alpha=5.47*10**6, # rho_water*g/viscosity [1/(ms)]
    d = 1.3*10**(-3), # mean grain size (m)
    Swi=0.07, # irreducible water saturation
    #k=11*10**(-11), # 13.9 - 38.1 * E-11 [m2] 
    por0 = 7.40734483*10**(-3), # derived with SWE measurements
    c = 2*6.30755824*10**(-6),
    rho_ice=917,rho_w=1000,rho_snow=50,hours_to_drain=24):
        
    #zs = np.linspace(
    runoff = 0
    t_passed = 0
    check_pass1 = True
    loops = 0
    depth = snowpack_depth
    while (t_passed<hours_to_drain*3600 or runoff<trapped_water_kg) and check_pass1==True:
        kgf,tf,rho = characteristic_slope(depth,water_volume=water_volume,pore_volume=pore_volume,t0=t_passed,
            n=n,gam=gam,
            alpha=alpha, d=d, Swi=Swi, por0=por0,
            c=c,rho_ice=rho_ice,rho_w=rho_w,rho_snow=rho_snow)
        if loops==0 and tf==0:
            check_pass1 = False
        if loops>12:
            check_pass1 = False
        runoff += kgf
        t_passed += tf
        loops += 1
        #print(loops,check_pass1)
        #print(trapped_water_kg,runoff)
        #print('    ',24*3600,t_passed)
    actual_runoff = runoff if runoff<=trapped_water_kg else trapped_water_kg
    return actual_runoff#,rho

#def fun_rho_snow(j):
    #if j>=0 and j<1:
        #p = 75
    #elif j>=1 and j<=3:
        #p = 100 + 50*(j-1)
    #elif j>3 and j<=13:
        #p = 200 + 10*(j-3)
    #elif j>13:
        #p = 300
    #return p
    
def one_step_SWE(X_cm,#Ice_chunk,SWE_i,P_frz,
    icF,snF,wF,
    Ticei,Tsnowi,Tswi,
    Ta,RH,Qnet,rho_snow):#, Cw = 4.2*10**9,Cs = 0.0021): 
        
    rho_ice = 917 # kg/m3    ice
    rho_water = 1000 # water
    # ---
    #rho_sfall,rho_sp = 50,50
    rho_snow_star = 50 + 3.4*(Ta + 15)
    rho_sfall = rho_snow_star if rho_snow_star<50 else 50
        
    # https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html
    Ck_snow = 2090/1000 # kJ/(kg C)   ice,snow (-5C)
    Ck_ice = 2093/1000 # ice (0C)
    Ck_water = 4182/1000 # water (20C)
    #C_snow = 0.0021 # kJ/(m3 C)
    #C_water = 4.2*10**9 # kJ/(m3 C)
    #Ck_snow,Ck_water = C_snow/rho_sp,C_water/rho_water
    #Ck_ice = Ck_snow
    #https://www.engineeringtoolbox.com/latent-heat-melting-solids-d_96.html
    lamk = 334 # kJ/kg

    X = X_cm/100 # cm --> m
    Xkg = X*1000 # m --> kg/m2
    Fs,Fr,Tb = rain_snow_amount(Xkg,Ta,RH)
    
    Penergy = Ck_water*Fr*Ta + Ck_snow*Fs*Ta
    Traini = 0
    #Tsfalli = Tb if Tb<0 else 0
    Tsfalli = 0
    Enet = Qnet + Penergy #- lam*Xs
    
    # (A1)
    # set ground temp for first snowfall
    if (icF + snF + wF + Fs)==0: # (no initial snowpack or snowfall)
        Fsw,Fsp,Fip,Fru = 0,0,0,Fr
        Tsw,Tsnow,Tice,Tru = 0,0,0,Tb
        rho_sp = rho_snow
        
        Ff,msF,miF = 0,0,0
        Left_energy = 0
        
    # (A2)
    # there is a snowpack and/or snowfall
    else:
        
        # (SHOULD NEVER BE NEEDED)
        if Tswi<0:
            Rnet -= wF*Ck_water*(0 - Tswi)
            Tswi = 0
            
        # (B1) freeze possible
        if Enet<0: #(freeze possible)
            if Tswi>0 or Traini>0: # need to cool water first
                Tlow = 0
                Cool_energy = wF*Ck_water*(Tlow - Tswi) + Fr*Ck_water*(Tlow - Traini) 
                Cnet = Enet - Cool_energy
                if Cnet>0:
                    Tsw = (Enet + (Fr + wF)*Ck_water*Tlow)/((Fr + wF)*Ck_water)
                    Enet = 0
                else:
                    Tsw,Enet = Tlow,Cnet
            else:
                if Fr+wF>0:
                    Tsw = (Fr*Traini + wF*Tswi)/(Fr + wF) # let water mix to get new temp
                else:
                    Tsw = 0
                
            Ff = -Enet/lamk # leftover energy goes to freezing
            if Ff > (Fr + wF):
                Ff = Fr + wF
                #print('more ice produced than water available')
            Rnet = Enet + Ff*lamk
        else: 
            Rnet = Enet
            Ff = 0
            if Fr + wF>0:
                Tsw = (Fr*Traini + wF*Tswi)/(Fr + wF) # let water mix to get new temp
            else:
                Tsw = 0
        # -----
        #Fsw = Fr + wF - Ff
        Fsw = wF - Ff
            
        # ---
        # (B2) warming + melt possible
        
        # ---- warming ------
        # (b2.1.snow --> warming)
        if (snF + Fs)>0 and Rnet>0:
            if Tsfalli<0 or Tsnowi<0: # need to warm snow first
                Tup = 0
                Warm_energy = snF*Ck_snow*(Tup - Tsnowi) #+ Fs*Ck_snow*(Tup - Tsfalli) 
                Wnet = Rnet - Warm_energy
                if Wnet<0:
                    bsnow = snF*Ck_snow #+ Fs*Ck_snow 
                    Tsnow_pk = (Rnet + snF*Ck_snow*Tsnowi)/bsnow #+ Fs*Ck_snow*Tsfalli
                    Rnet = 0
                else:
                    Tsnow_pk,Rnet = Tup,Wnet
            #else: # let snow temps equilibrate
            if (Fs + snF)>0:
                Tsnow = (Fs*Tsfalli + snF*Tsnowi)/(Fs + snF)
                rho_sp = (Fs*rho_sfall + snF*rho_snow)/(Fs + snF)
            else:
                Tsnow = 0
                rho_sp = rho_sfall   
        else:
            if (Fs + snF)>0:
                Tsnow = (Fs*Tsfalli + snF*Tsnowi)/(Fs + snF)
                rho_sp = (Fs*rho_sfall + snF*rho_snow)/(Fs + snF)
            else:
                Tsnow = 0
                rho_sp = rho_sfall
            #C_sp,rho_sp = C_snow,rho_snow
        Fsp = Fs + snF
        #WXsp = Fsp/rho_water
        
                
        # (b2.1.ice --> warming)
        if (icF + Ff)>0 and Rnet>0:
            if Ticei<0 and Rnet>0: # need to warm ice first
                Tup2 = 0
                Warm_energy2 = Ff*Ck_ice*(Tup2 - 0) + icF*Ck_ice*(Tup2 - Ticei)
                Wnet2 = Rnet - Warm_energy2
                if Wnet2<0:
                    bice = (Ff + icF)*Ck_ice 
                    Tice = (Rnet + Ff*Ck_ice*0 + icF*Ck_ice*Ticei)/bice
                    Rnet = 0
                else:
                    Tice,Rnet = Tup2,Wnet2
            else: # let ice temps equilibrate
                if (Ff + icF)> 0:
                    Tice = (Ff*0 + icF*Ticei)/(Ff + icF)
                else:
                    Tice = 0
        else:
            if (Ff + icF)> 0:
                Tice = (Ff*0 + icF*Ticei)/(Ff + icF)
            else:
                Tice = 0
        Fip = Ff + icF
           
        # ---- melting ------
        # (b2.2.snow --> melting)
        if snF>0 and Rnet>0 and Tsnow==0:
            msF = Rnet/lamk
            if msF>snF:
                msF = snF
            Rnet -= msF*lamk
        else:
            msF = 0
            
        # (b2.2.ice --> melting)
        if Fip>0 and Rnet>0 and Tice==0:
            miF = Rnet/lamk
            if miF>Fip:
                miF = Fip
            Rnet -= miF*lamk
        else:
            miF = 0
            
        ##### new water amount and temp:
        if (Fsw + Fr + msF + miF)>0:
            Tswf = (Tsw*(Fsw+Fr) + 0*msF + 0*miF)/((Fsw+Fr) + msF + miF)
            Tsw = Tswf
        Tru = Tsw
        #Fsw1 = Fsw + Fr
        #Fsw2 = msF + miF
        #Fsw += (msF + miF)
        
        # runoff
        #----------------------------------
        #dreal = Fsp/rho_sp + Fip/rho_ice
        #dif_water = (Fsp+Fip)/rho_water
        #dvoid = dreal - dif_water
        #porosity = dvoid/dreal
        
        ## do for fresh melt
        if Fsw>0:
            V_sp,V_ip = Fsp/rho_sp, Fip/rho_ice
            V_sp_if_all_ice = Fsp/rho_sp
            V_air = V_sp - V_sp_if_all_ice
            V_sw = Fsw/rho_water
            V_voids = V_air + V_sw
            V_tot = V_sp + V_ip + V_sw
            
            drainage_0 = loop_melt(V_tot,trapped_water_kg=Fsw,water_volume=V_sw,pore_volume=V_voids,
                rho_ice=rho_ice,rho_w=rho_water,rho_snow=rho_sp,)
        else:
            drainage_0 = 0
            
        # find standing water and drainage
        Fru = drainage_0 if drainage_0<=Fsw else Fsw
        Fsw -= Fru
        Fsw += (Fr + msF + miF)
        # --
        Fsp -= msF
        Fip -= miF
        if (Fsp+Fip)<0.00001:
            Fru += Fsw
            Fsw = 0
        #if Fru>(Fsw1+Fsw2) or (Fsp+Fip)==0:
            #Fru = Fsw1 + Fsw2
        #Fsw = Fsw1 + Fsw2 - Fru
        #if Fsw==0:
            #Tsw = 0
        #rho_sp = rho_all
        
        Left_energy = Rnet

    
    # (C)
    dic = {}
    namers = ['rainfall','snowfall','snowpack','icepack','trapped water',
        'runoff','refreeze','snowmelt','icemelt']
    placersF = [Fr,Fs,Fsp,Fip,Fsw,Fru,Ff,msF,miF]
    placers_rho = [rho_water,rho_sfall,rho_sp,rho_ice,rho_water,
        rho_water,rho_ice,rho_sp,rho_ice]
    placersT = [Tb,Tsfalli,Tsnow,Tice,Tsw,Tsw,0,0,0]
    
    ct = 0
    for nam in namers:
        F = placersF[ct]
        WF = F/rho_water*100
        XF = F/placers_rho[ct]*100
        dic.update({nam:[F,WF,XF,placersT[ct]]})
        ct += 1
    #--
    if (Fsp + Fip)>0:
        savT = (Tsnow*Fsp + Tice*Fip)/(Fsp+Fip)
    else:
        savT = 0
    F = dic['snowpack'][0] + dic['icepack'][0]
    WF = dic['snowpack'][1] + dic['icepack'][1]
    XF = dic['snowpack'][2] + dic['icepack'][2]
    dic.update({'solid pack': [F,WF,XF,savT]})
    #--
    if (Fsp + Fip + Fsw)>0:
        avT = (Tsnow*Fsp + Tice*Fip + Tsw*Fsw)/(Fsp+Fip+Fsw)
    else:
        avT = 0
    F = dic['snowpack'][0] + dic['icepack'][0] + dic['trapped water'][0]
    WF = dic['snowpack'][1] + dic['icepack'][1] + dic['trapped water'][1]
    XF = dic['snowpack'][2] + dic['icepack'][2] + dic['trapped water'][2]
    dic.update({'pack': [F,WF,XF,avT]})
    #--
    if (msF + miF)>0:
        savT = (Tsnow*Fsp + Tice*Fip)/(Fsp+Fip)
    else:
        savT = 0
    F = dic['snowmelt'][0] + dic['icemelt'][0]
    WF = dic['snowmelt'][1] + dic['icemelt'][1]
    XF = dic['snowmelt'][2] + dic['icemelt'][2]
    dic.update({'melt': [F,WF,XF,savT]})
    #--
    # --------------------
    dic.update({'Prec Energy': Penergy})
    dic.update({'Left Energy': Left_energy})
    dic.update({'Net Energy': Qnet + Penergy})
    dic.update({'snow density':rho_sp})
    
    return dic


def surface_water_input(loc='BLF',save_blf=False,
    spring_adjust=True,winter_constant_G=True,
    use_heat_fluxes=False,use_radiations=False):
    #T_threshold = 1, # C, threshold wet-bulb temp for determining snowfall event
    #lam = 3.35*10**5, # kJ/m3, latent heat of fusion
    #Cw = 4.2*10**9, # kJ/(m3 C), heat capacity of water
    #Cs = 0.0021): # kJ/(m3 C), heat capacity of snow
        
    # open dictionary and set summer albedo:
    v = load_obj('paramsL_rad')
    n = v
    A0,Tsoil_call = 0.5,'Ts10'
        
    # find soil temp ranges:
    Tsoil_all = v[Tsoil_call][1517]
    minTsoil,maxTsoil = min(Tsoil_all),max(Tsoil_all)
    Gpercent = line_2pts([minTsoil,.03],[maxTsoil,.08])
    #print(v['Rn'][2010])
    
    # start dic components:
    # [Accum,Accum2], [Am_runoff,Am_rain,Am_melt], [Ice,Tice], [SWE,Tsnow], [Am_sw,Tswf],[Xr,Xs],[Traini,Tsfalli]
    parms = ['rainfall','snowfall','snowpack','icepack','trapped water','runoff',
        'pack','solid pack','melt','refreeze']
    calls = []
    fcs,ucs = [],[]
    for pp in parms:
        calls.append(pp+', kg')
        calls.append(pp+', cm (WQ)')
        calls.append(pp+', cm')
        calls.append(pp+', temp')
        ucs.append('kg')
        ucs.append('cm (water equivalent)')
        ucs.append('cm')
        ucs.append('C')
        for fc in range(0,4):
            fcs.append(pp)
    calls = np.append(calls,['netQ',#'netHLe','netRsLn','G','Prec Energy','Left Energy',
        'Rnet','Le','H','G','Prec Energy',
        'Snow(m)','SnowA(m)','SnowMelt(m)','Infil(m)','Ground Temp(m)','snow density'])
    fcs = np.append(fcs,['Net input energy',
        'Net radiation','Latent energy','Sensible heat','Ground heat flux','Preciptation input energy',
        'Snowfall','Snowpack','Snowmelt','Surface water input','Surface temperature','Snowpack density'])
    ucs = np.append(ucs,['kJ/m2',
        'kJ/m2','kJ/m2','kJ/m2','kJ/m2','kJ/m2',
        'cm','cm','cm','cm','C','kg/m2'])
    
    all_holds = [ [] for cc in calls]
    for cc in calls:
        v.update({cc:{}})
    
    # add energy inputs:
    #add_Rs_and_Ln(v,A0=A0,fixed_albedo=A0, use_radiations=use_radiations)
    #add_LE_and_H(v, use_heat_fluxes=use_heat_fluxes)
    
    A_i,Ts = 0,0#A0,0
    #Ice,SWE,X_wt = 0,0,0
    icF,snF,wF = 0,10,0
    Ticei,Tsnowi,Tswi = 0,0,0
    days_packed = 0
    ii1517 = 0
    rho_snow = 50
    for yr in v['good years']:
        holders = [ [] for cc in calls]
        for ii in range(0,len(v['DoY'][yr])):
            Prec_cm = v['Prec'][1517][ii1517]#/100 # --> convert to m
            Tair,RH,Tsoil = v['Tair'][1517][ii1517],v['min RH'][1517][ii1517],v[Tsoil_call][1517][ii1517]

            Tsurf = 0.534 + 0.705*Tsoil
                
            Tmin,Tmax = v['min Tair'][yr][ii], v['max Tair'][yr][ii]
            #Q1 = v['Le'][yr][ii] + v['H'][yr][ii]

            #------------------------------------
            Rn = v['Rn'][yr][ii]*24*3600/1000 # W/m2 --> kJ/m2
            LE = v['LE'][yr][ii]*24*3600/1000*2 # W/m2 --> kJ/m2
            H = v['senH'][yr][ii]*24*3600/1000
            if winter_constant_G==True and (icF+snF<0.00001):
                G = 173
            else:
                gp = Gpercent(Tsoil)
                G = gp*Rn
            #G = 173#-1000
            #------------------------------------
            Q1 = LE + H
            Qnet = Q1 + Rn + G 
            md = one_step_SWE(Prec_cm,icF,snF,wF,
                Ticei,Tsnowi,Tswi,
                Tair,RH,Qnet,rho_snow)
            
            # ------------
            icF,snF,wF = md['icepack'][0],md['snowpack'][0],md['trapped water'][0]
            Ticei,Tsnowi,Tswi = md['icepack'][3],md['snowpack'][3],md['trapped water'][3]
            A_i = albedo
            if (icF+snF)>0:
                Ts = md['solid pack'][3]
                days_packed += 1
            else:
                Ts = Tsoil
                days_packed = 0 
            # --------------
            adds = [None for cc in calls]
            cmi = 0
            for cm in parms:
                r = md[cm]
                adds[cmi],adds[cmi+1],adds[cmi+2],adds[cmi+3] = r[0],r[1],r[2],r[3]
                cmi += 4
            adds[cmi] = Qnet + md['Prec Energy']
            adds[cmi+1] = Rn
            adds[cmi+2] = LE
            adds[cmi+3] = H
            adds[cmi+4] = G
            adds[cmi+5] = md['Prec Energy']
            # ----
            adds[cmi+6] = md['snowfall'][1]
            adds[cmi+7] = md['solid pack'][1]
            adds[cmi+8] = md['snowmelt'][1] + md['icemelt'][1] - md['refreeze'][1]
            adds[cmi+9] = md['runoff'][1]
            adds[cmi+10] = Ts 
            adds[cmi+11] = md['snow density']
            #--------------
            for jj in range(0,len(calls)):
                holders[jj].append(adds[jj])
            # ----
            ii1517 += 1
            rho_snow = md['snow density']
        
        for jj in range(0,len(calls)):
            v[calls[jj]].update({yr:holders[jj]})
            all_holds[jj] = np.append(all_holds[jj],holders[jj])
    for jj in range(0,len(calls)):
        v[calls[jj]].update({1517:all_holds[jj]})
    
    idx = 0
    for cc in calls:
        n['full calls'].update({cc:fcs[idx]})
        n['units'].update({cc:ucs[idx]})
        idx += 1
        
    cls = n['calls']
    ncls = np.append(cls,calls)
    n.update({'calls':ncls})
    #v.update({'naming':n})
    
    if loc=='BLF' and save_blf==True:
        save_obj(v,'paramsL22')
    return v
    
    # -------------------------------------------------------------------------------------------------------------------



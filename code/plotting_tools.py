
import sys, os, os.path
cwd = os.getcwd()
#main_dirc = cwd.split('peat_project', 1)[0]
#cwd_code = main_dirc + 'peat_project/code'
sys.path.insert(0, cwd+'/snow_model')
from energy_balance_xue import *

#sys.path.insert(0, cwd+'/snow_model/prelims')
#from save_funcs import *
#import basic_fit_funcs as btf

def plot_snow_model(v,loc='BLF',show_fluxes=False,show_temps=False,show_runoff=False,show_precips=False,show_corr=False,
    plot_units='kg',plot_years=1517):
        
    if type(plot_years)!=list:
        plot_years = [plot_years]
    if show_fluxes or show_temps or show_precips or show_runoff or show_corr:
        show_p = True
    else:
        show_p = False
    uu = ', '+plot_units
    
    if show_fluxes==True:
        nn,rr = 1,1
        fake_plot_adj(xN=nn,yN=rr,timeseries=True)
        fig,ax = plt.subplots(nn,rr)
        plt.subplot(nn,rr,1)
        for pyr in plot_years:
            T = v['TotDays'][pyr]
            if pyr==plot_years[0]:
                plt.plot(T,v['Prec Energy'][pyr],'slateblue',label='Prec Energy')
                plt.plot(T,v['Rnet'][pyr],'gold',label='Rnet')
                plt.plot(T,v['H'][pyr],'darkred',label='H',alpha=0.5)
                plt.plot(T,v['Le'][pyr],'mediumseagreen',label='LE')
                plt.plot(T,v['G'][pyr],'darkgoldenrod',label='G')
                plt.plot(T,v['netQ'][pyr],'pink',label='Qnet',ls='-.',alpha=0.8)
            else:
                plt.plot(T,v['Prec Energy'][pyr],'slateblue')
                plt.plot(T,v['Rnet'][pyr],'gold')
                plt.plot(T,v['H'][pyr],'darkred',alpha=0.5)
                plt.plot(T,v['Le'][pyr],'mediumseagreen')
                plt.plot(T,v['G'][pyr],'darkgoldenrod')
                plt.plot(T,v['netQ'][pyr],'pink',ls='-.',alpha=0.8)
        plt.title(loc+': energy balance inputs')
        #plt.ylim(ymin=-5000,ymax=18000)
        plt.xlim(xmin=1,xmax=1+364*(2019-2009))
        plt.ylabel('SWE (cm)')
        plt.grid()
        plt.legend()
        plt.xticks([1+364*(year-2009) for year in v['good years']],['     '+str(year) for year in v['good years']])
        fake_plot_end(fig,xN=nn,yN=rr,hspace=0)
        fig_type = 'pdf'
        plt.savefig(gn('energy_balance_inputs',f='.'+fig_type),format=fig_type)
    
    if show_runoff==True:
        nn,rr = 1,1
        fake_plot_adj(xN=nn,yN=rr,timeseries=True)
        fig,ax = plt.subplots(nn,rr)
        plt.subplot(nn,rr,1)
        for pyr in plot_years:
            T = v['TotDays'][pyr]
            if pyr==plot_years[0]:
                plt.plot(T,v['pack, kg'][pyr],'k',label='snowpack',alpha=0.4)
                plt.plot(T,v['trapped water, kg'][pyr],'cyan',label='trapped water',alpha=0.8)
                plt.plot(T,[-rr for rr in v['runoff, kg'][pyr]],'navy',label='runoff',alpha=0.8)
            else:
                plt.plot(T,v['pack, kg'][pyr],'k',alpha=0.4)
                plt.plot(T,v['trapped water, kg'][pyr],'cyan',alpha=0.5)
                plt.plot(T,[-rr for rr in v['runoff, kg'][pyr]],'navy',alpha=0.8)
            px1,py1,py2 = btf.area_arrays([0 for dd in T],v['pack, kg'][pyr],x=T,top=2)
            plt.fill_between(px1,py1,py2,color='k',alpha=0.4,zorder=2)
        plt.title(loc+': runoff and water storage')
        #plt.ylim(ymin=-1,ymax=1)
        plt.xlim(xmin=1,xmax=1+364*(2019-2009))
        plt.ylabel('amount (kg)')
        plt.grid()
        plt.legend()
        plt.xticks([1+364*(year-2009) for year in v['good years']],['     '+str(year) for year in v['good years']])
        fake_plot_end(fig,xN=nn,yN=rr,hspace=0)
        fig_type = 'pdf'
        plt.savefig(gn('runoff',f='.'+fig_type),format=fig_type)

    if show_corr==True:
        nn,rr = 3,1
        fake_plot_adj(xN=nn,yN=rr,timeseries=True)
        fig,ax = plt.subplots(nn,rr)
        pct = 1
        li = 0
        SWEdic = load_obj('BLF/snow_dic_byJUNC')
        #plt.subplot(nn,rr,pct)
        #for year in v['good years']:
        yrs = v['good years']#[2010,2011,2012]
        for year in yrs:
            #plt.subplot(nn,rr,pct)
            # PLOT 1 -----------------------------------------------------------
            plt.subplot(nn,rr,1)
            T = v['TotDays'][year]
            if pct==1:
                plt.plot(T,v['pack, cm (WQ)'][year],alpha=0.8,color='gray',zorder=1,label='model at BLF')
            else:
                plt.plot(T,v['pack, cm (WQ)'][year],alpha=0.8,color='gray',zorder=1)
            px1,py1,py2 = btf.area_arrays([0 for dd in T],v['pack, cm (WQ)'][year],x=T,top=2)
            plt.fill_between(px1,py1,py2,color='lightgray',alpha=0.4)
            
            T_m = SWEdic[li]['TotDays'][year]
            S_m = [ss for ss in SWEdic[li]['SWE'][year]]
            for ili in SWEdic['ind locs']:
                plt.plot(T_m,SWEdic[ili]['SWE'][year],'gold',alpha=0.7)
            if pct==1:
                    plt.scatter(T_m,S_m,s=50,marker='x',alpha=0.5,color='darkred',edgecolor='darkred',zorder=0,lw=1,label='measurements at junction fen')
            else:
                plt.scatter(T_m,S_m,s=50,marker='x',alpha=0.5,color='darkred',edgecolor='darkred',zorder=0,lw=1)
            plt.plot(T_m,S_m,alpha=0.5,color='darkred')
            px1,py1,py2 = btf.area_arrays([0 for dd in T_m],S_m,x=T_m,top=2)
            plt.fill_between(px1,py1,py2,color='darkred',alpha=0.1)
            
            
            ## PLOT 2 -----------------------------------------------------------
            #plt.subplot(nn,rr,2)
            ##T = v['TotDays'][year]
            #plt.plot(T,v['pack, cm'][year],alpha=0.8,color='gray',zorder=1)
            #px1,py1,py2 = btf.area_arrays([0 for dd in T],v['pack, cm'][year],x=T,top=2)
            #plt.fill_between(px1,py1,py2,color='lightgray',alpha=0.4)
            #S_m = [ss for ss in SWEdic[li]['snow depth'][year]]
            #for ili in SWEdic['ind locs']:
                #plt.plot(T_m,SWEdic[ili]['snow depth'][year],'gold',alpha=0.7)
            #plt.scatter(T_m,S_m,s=50,marker='x',alpha=0.5,color='darkred',edgecolor='darkred',zorder=0,lw=1)
            #plt.plot(T_m,S_m,alpha=0.5,color='darkred')
            #px1,py1,py2 = btf.area_arrays([0 for dd in T_m],S_m,x=T_m,top=2)
            #plt.fill_between(px1,py1,py2,color='darkred',alpha=0.1)
            
            # PLOT 2 -----------------------------------------------------------
            plt.subplot(nn,rr,2)
            T = v['TotDays'][year]
            if pct==1:
                #plt.plot(T,v['netQ'][year],alpha=0.8,color='gray',zorder=1)
                plt.plot(T,v['Prec Energy'][year],'slateblue',label='Prec Energy')
                plt.plot(T,v['Rnet'][year],'gold',label='Rnet')
                plt.plot(T,v['H'][year],'darkred',label='H',alpha=0.5)
                plt.plot(T,v['Le'][year],'mediumseagreen',label='LE')
                plt.plot(T,v['G'][year],'darkgoldenrod',label='G')
                plt.plot(T,v['netQ'][year],'pink',label='Qnet',ls='-.',alpha=0.8)
            else:
                plt.plot(T,v['Prec Energy'][year],'slateblue')
                plt.plot(T,v['Rnet'][year],'gold')
                plt.plot(T,v['H'][year],'darkred',alpha=0.5)
                plt.plot(T,v['Le'][year],'mediumseagreen')
                plt.plot(T,v['G'][year],'darkgoldenrod')
                plt.plot(T,v['netQ'][year],'pink',ls='-.',alpha=0.8)
            
            # PLOT 2 -----------------------------------------------------------
            plt.subplot(nn,rr,3)
            T = v['TotDays'][year]
            #plt.plot(T,v['netQ'][year],alpha=0.8,color='gray',zorder=1)
            if pct==1:
                plt.plot(T,v['rainfall, kg'][year],'y',label='rainfall',alpha=0.3)
                plt.plot(T,v['snowfall, kg'][year],'slateblue',label='snowfall',alpha=0.9)
                plt.plot(T,[-rr for rr in v['runoff, kg'][year]],'brown',alpha=0.8,label='runoff')
            else:
                plt.plot(T,v['rainfall, kg'][year],'y',alpha=0.3)
                plt.plot(T,v['snowfall, kg'][year],'slateblue',alpha=0.9)
                plt.plot(T,[-rr for rr in v['runoff, kg'][year]],'brown',alpha=0.8)
            
            pct += 1
            
                
        plt.subplot(nn,rr,1)
        plt.xticks([1+365*(year-2009) for year in yrs],['' for year in yrs])
        plt.xlim(xmin=1+365*(min(yrs)-2009),xmax=1+365*(max(yrs)+1-2009))
        plt.ylabel('SWE (cm)')
        plt.grid()
        plt.legend()
        
        plt.subplot(nn,rr,2)
        plt.xticks([1+365*(year-2009) for year in yrs],['     '+str(year) for year in yrs])
        plt.xlim(xmin=1+365*(min(yrs)-2009),xmax=1+365*(max(yrs)+1-2009))
        plt.ylabel('energy (kJ/m2)')
        plt.grid()
        plt.legend()
        
        plt.subplot(nn,rr,3)
        plt.xticks([1+365*(year-2009) for year in yrs],['     '+str(year) for year in yrs])
        plt.xlim(xmin=1+365*(min(yrs)-2009),xmax=1+365*(max(yrs)+1-2009))
        plt.ylabel('events (kg)')
        plt.grid()
        plt.legend()


        fake_plot_end(fig,xN=nn,yN=rr,hspace=0)
        fig_type='pdf'
        plt.savefig(gn('snow_model_'+'_junc'+str(li),f='.'+fig_type),format=fig_type)
        
    if show_precips==True:
        nn,rr = 1,2
        fake_plot_adj(xN=nn,yN=rr)
        fig,ax = plt.subplots(nn,rr)
        
        pi = 1
        prams = [['snowfall','rainfall'],['pack','melt','refreeze']]
        x_var = 'Tair'
        col = ['k.','y.','b.']
        for pram in prams:
            plt.subplot(nn,rr,pi)
            for pyr in plot_years:
                ii = 0
                if pyr==plot_years[0]:
                    for pr in pram:
                        T = [v[x_var][pyr][jj] for jj in range(0,len(v[x_var][pyr])) if v[pr+', kg'][pyr][jj]>0]
                        Y = [val for val in v[pr+', kg'][pyr] if val>0]
                        plt.plot(T,Y,col[ii],label=pr,alpha=0.4)
                        ii += 1
                else:
                    for pr in pram:
                        T = [v[x_var][pyr][jj] for jj in range(0,len(v[x_var][pyr])) if v[pr+', kg'][pyr][jj]>0]
                        Y = [val for val in v[pr+', kg'][pyr] if val>0]
                        plt.plot(T,Y,col[ii],label=pr,alpha=0.4)
                        ii += 1
            #plt.title(loc+': snowfall and snowmelt')
            plt.ylabel('amount (kg)')
            plt.grid()
            plt.legend()
            #plt.xticks([1+364*(year-2009) for year in v['good years']],['     '+str(year) for year in v['good years']])
            plt.xlabel(v['full calls'][x_var]+', '+v['units'][x_var])
            pi += 1
            
        fake_plot_end(fig,xN=nn,yN=rr)
        fig_type = 'pdf'
        plt.savefig(gn('runoff_vs_'+x_var,f='.'+fig_type),format=fig_type)
    
    if show_p==True:
        plt.show()
    
            
swd = surface_water_input(loc='BLF',use_heat_fluxes=True,use_radiations=True,
    spring_adjust=True,winter_constant_G=True,save_blf=True,)
plot_snow_model(swd,show_corr=True,
    show_runoff=True,
    show_fluxes=True,
    #show_Rs=True,
    show_precips=True,
    #plot_units='cm (WQ)',plot_years=[2010,2011])
    plot_units='cm (WQ)',plot_years=[1517])
    
#surface_water_input(loc='s2',use_heat_fluxes=False,use_radiations=False,
    #show_accums=True,plot_units='cm')
       
#plt.legend()
plt.show()

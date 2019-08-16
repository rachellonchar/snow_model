
#regular code directory setup:
import sys, os, os.path
cwd = os.getcwd()
main_dirc = cwd.split('peat_project', 1)[0]
cwd_code = main_dirc + 'peat_project/code'
sys.path.insert(0, cwd_code+'/prelims')
from save_funcs import *
main_dirc = cwd.split('peat_project', 1)[0]+'peat_project/' #change variable back
#-------------------------------------------

font2 = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 5}

matplotlib.rc('font', **font2)
matplotlib.rcParams.update({'figure.figsize': [9,9]})
matplotlib.rcParams.update({'figure.titlesize': 10})




#plt.figure()
def fake_plot(rows=3,columns=2,fig_type='pdf'):
    
    if columns>=rows:
        calu,dif= columns//rows - 1, columns-rows
        ac = 15+calu if 15+calu<=18 else 18
        ar = 9-calu if 9-calu>=6 else 6
        
    else:
        calu,dif = rows//columns - 1, rows-columns
        ar = 15+calu if 15+calu<=18 else 18
        ac = 11-calu if 11-calu>=9 else 9
    times = columns*rows
    
    hspace = 0.2 +0.11*(rows-1)
    adjl = 12-ac if 12-ac>0 else 0
    left = 0.05 + adjl*0.05
    title_font = 21-times if 21-times>15 else 15
    axes_font = title_font-2
    leg_font = axes_font-1 
    
    # (start)
    fig,ax = plt.subplots(rows,columns,figsize=(ac,ar))
    ct = 1
    for pp in range(0,rows*columns):
        
        # (sub)
        plt.subplot(rows,columns,ct)
        if ct==1:
            plt.annotate('fontsize='+str(axes_font),xy=[0.1,0.9],fontsize=axes_font)
        plt.xticks([0.1,0.5,0.9])
        plt.yticks([0.1,0.5,0.9])
        plt.title('figsize=('+str(ac)+','+str(ar)+'), fontsize='+str(title_font),fontsize=title_font)
        plt.ylabel('y-ax',fontsize=axes_font)
        plt.xlabel('fontsize='+str(axes_font),fontsize=axes_font)
        ct += 1
    
    # (end)
    plt.tight_layout()
    plt.subplots_adjust(left=left, bottom=0.1, right=0.95, top=0.95, wspace=None, hspace=hspace)
    #plt.show()




def fake_plot_info(rows=3,columns=2):
    
    if columns>=rows:
        calu,dif= columns//rows - 1, columns-rows
        ac = 15+calu if 15+calu<=18 else 18
        ar = 9-calu if 9-calu>=6 else 6
        
    else:
        calu,dif = rows//columns - 1, rows-columns
        ar = 15+calu if 15+calu<=18 else 18
        ac = 11-calu if 11-calu>=9 else 9
    times = columns*rows
    
    hspace = 0.2 +0.11*(rows-1)
    adjl = 12-ac if 12-ac>0 else 0
    left = 0.05 + adjl*0.03
    title_font = 21-times if 21-times>15 else 15
    axes_font = title_font-2
    leg_font = axes_font-1
    num_ax_font = leg_font-3 if leg_font-3>=10 else 10
    return ac,ar,hspace,left,title_font,axes_font,leg_font,num_ax_font
    
    # (start)
def fake_plot_adj(xN=1,yN=1,timeseries=False):
    
    ac,ar,hspace,left,title_font,axes_font,leg_font,num_ax_font = fake_plot_info(rows=xN,columns=yN)
    
    #fig,ax = plt.subplots(rows,columns,figsize=(ac,ar))
    matplotlib.rc('font', **font2)
    matplotlib.rcParams.update({'font.size': axes_font})
    matplotlib.rcParams.update({'legend.fontsize': leg_font})
    matplotlib.rcParams.update({'figure.titlesize': title_font})
    matplotlib.rcParams.update({'xtick.labelsize': num_ax_font})
    matplotlib.rcParams.update({'ytick.labelsize': num_ax_font})
    
    matplotlib.rcParams.update({'legend.fancybox': True })
    matplotlib.rcParams.update({'legend.framealpha': 0.5 })
    matplotlib.rcParams.update({'legend.loc': 'upper left' })
    matplotlib.rcParams.update({'savefig.transparent' : True})
    matplotlib.rcParams.update({'savefig.bbox': 'tight'})
    matplotlib.rcParams.update({'figure.figsize': [ac,ar]})
    if timeseries==True:
        matplotlib.rcParams.update({'figure.figsize': [ac*2,ar]})



def fake_plot_end(fig,xN=1,yN=1,
    left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        
    ac,ar,hspaceB_,leftB_,title_font,axes_font,leg_font,num_ax_font = fake_plot_info(rows=xN,columns=yN)
    
    leftB = leftB_ if left==None else left
    hspaceB = hspaceB_ if hspace==None else hspace
    #--
    bottomB = 0.1 if bottom==None else bottom
    rightB = 0.95 if right==None else right
    topB = 0.95 if top==None else top
    wspaceB = None if wspace==None else wspace
    
    plt.tight_layout()
    plt.subplots_adjust(left=leftB, bottom=bottomB, right=rightB, top=topB, wspace=wspaceB, hspace=hspaceB)



    
    
    
    





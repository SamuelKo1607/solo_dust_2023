import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import commons_figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
from commons_read_ephemeris import fetch_heliocentric
from ag2023_download import fetch
from ag2023_bias import bias
import glob
from scipy.signal import savgol_filter
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.signal import welch
from scipy import interpolate
import scipy.stats
from commons_conversions import tt2000_to_jd
mpl.rcParams['text.usetex'] = False

#astronomical unit, km
au = 149597870.7 

#normal distribution
pnorm = scipy.stats.norm.cdf
qnorm = scipy.stats.norm.ppf

#triangulation
#z = 0.5 #how far is the antenna plane recessed behind the heat shield plane [m]
x1,y1 = 0       , 2.8371 #antenna 1  [m], positive x is prograde
x2,y2 = -1.8823 , -1.4185 #antenna 2  [m], positive y is gocentric north
x3,y3 = 1.8823  , -1.4185 #antenna 3  [m]
l = 6.5 #length of antenna past heat shield

#make list of cnn files
path = "/997_data/solo_amplitude_data"
txts = glob.glob(path+"/*e_2*.txt")

#fetch solo ephemerides
f_heliocentric_distance, f_heliocentric_phi, f_rad_v, f_tan_v = fetch_heliocentric("solo")

#used to fit exponential decay of the ion peak
def exponential_decay(x,RC,A):
    return A*np.exp(-x/RC)    


#used for a proxy for body peak
def gaussian_body(x, mu, sig):
    lhs = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    rhs = np.exp(-np.power(x - mu, 2.) / (2 * np.power(3*sig, 2.)))
    return (x<mu)*lhs+(x>=mu)*rhs


#a simple gaussian
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


#correct the time-series for a high-pass filter of X Hz through Laplace domain as Amalia did
def hipass_correction(signal,epochstep,cutoff=370):
    #cutoff in Hz
    #epochstep in s
    cumsignal = np.cumsum(signal)
    return signal + (2*np.pi*cutoff)*(cumsignal*epochstep)


#function for finding the highest correlation with a shape
def highest_corr(signal):
    #signal has to be 36 long
    shape = np.array([-25,-16,-9,-4,-1,0,0,0,0,0,0])
    corrs = np.zeros(25)
    for i in range(25):
        corrs[i] = np.corrcoef(signal[i:i+11],shape)[0,1]
    return(np.argmax(corrs)+5) #the index of the signal array centered on the peak
 

def construct_subplots():
    fig = plt.figure(figsize=(5,3))
    gs = fig.add_gridspec(3,2,width_ratios=[1, 2],hspace=0.1,wspace=0.1)
    ax = gs.subplots()
    ax[0,0].xaxis.set_ticklabels([])
    ax[1,0].xaxis.set_ticklabels([])
    ax[0,1].xaxis.set_ticklabels([])
    ax[1,1].xaxis.set_ticklabels([])
    ax[0,1].yaxis.set_ticklabels([])
    ax[1,1].yaxis.set_ticklabels([])
    ax[2,1].yaxis.set_ticklabels([])
    ax[0,0].set_ylabel("V1 [V]")
    ax[1,0].set_ylabel("V2 [V]")
    ax[2,0].set_ylabel("V3 [V]")
    ax[2,0].set_xlabel("time [$\mu s$]")
    ax[2,1].set_xlabel("time [$ms$]")
    return [fig,ax]
   

#waveforms plotting routine
def plot_waveforms(waveforms,    #list of arrays: [[ch1,ch2,ch3],[ch1,ch2,ch3],...]
                   epoch,      #x-axis = time
                   imax1,imax2,imax3,   #positions of maxima (ion peak)
                   index_center, #position of the view centering and of the most reliable peak
                   index_body, #position of the body peak
                   amplitude_body, #body amplitude value 
                   index_global,index_local,index_count_day,     #indexing for name
                   waveform_colors = ["blue","red","orange","green","grey","magenta","teal","firebrick","mint","navy"],
                   waveform_alphas = np.ones(10,dtype=float),
                   save=False,      #to save or not to save 
                   secondary_present = np.array([1,1,1]),    #which secondary peaks are there and which are not
                   additional_vlines=[],    
                   additional_vlines_colors=[],
                   additional_hlines=[],
                   additional_hlines_colors=[],
                   reason = " ",
                   folder = "",
                   name_prefix = "Lorem Ipsum"):
    
    subplots = construct_subplots()
    fig = subplots[0]
    ax = subplots[1]
    
    #range is set by the first waveform
    ylim = 1.1*max([max(np.abs(waveforms[0][0])),max(np.abs(waveforms[0][1])),max(np.abs(waveforms[0][2]))])
    xmin = -900
    xmax = +1600   
    for axis in ax[:,0]:
        axis.set_ylim(-ylim,ylim)
        axis.set_xlim(xmin,xmax)
        if reason != " ":
            axis.set_facecolor('salmon')
    for axis in ax[:,1]:
        axis.set_ylim(-ylim,ylim)
        axis.axvspan(xmin/1000, xmax/1000, alpha=0.6, color='lightgrey')
        #axis.vlines(xmin/1000,-ylim,ylim,color="black",alpha=0.3)
        #axis.vlines(xmax/1000,-ylim,ylim,color="black",alpha=0.3)
        if reason != " ":
            axis.set_facecolor('salmon')
    
    #time windows is centered about the most important peak
    epoch_center = epoch[index_center]
    
    for wf in range(len(waveforms)):
        ch1 = waveforms[wf][0]
        ch2 = waveforms[wf][1]
        ch3 = waveforms[wf][2]
    
        ax[0,0].plot((epoch-epoch_center)/1000,ch1,color=waveform_colors[wf],lw=0.5,alpha=waveform_alphas[wf])
        ax[1,0].plot((epoch-epoch_center)/1000,ch2,color=waveform_colors[wf],lw=0.5,alpha=waveform_alphas[wf])
        ax[2,0].plot((epoch-epoch_center)/1000,ch3,color=waveform_colors[wf],lw=0.5,alpha=waveform_alphas[wf])
    
        ax[0,1].plot((epoch-epoch_center)/1000000,ch1,color=waveform_colors[wf],lw=0.5,alpha=waveform_alphas[wf])
        ax[1,1].plot((epoch-epoch_center)/1000000,ch2,color=waveform_colors[wf],lw=0.5,alpha=waveform_alphas[wf])
        ax[2,1].plot((epoch-epoch_center)/1000000,ch3,color=waveform_colors[wf],lw=0.5,alpha=waveform_alphas[wf])
    
    #secondary_peaks
    ax[0,0].vlines((epoch[imax1]-epoch_center)/1000,-ylim,ylim,color="red",alpha=0.5*secondary_present[0],zorder=0)
    ax[1,0].vlines((epoch[imax2]-epoch_center)/1000,-ylim,ylim,color="red",alpha=0.5*secondary_present[1],zorder=0)
    ax[2,0].vlines((epoch[imax3]-epoch_center)/1000,-ylim,ylim,color="red",alpha=0.5*secondary_present[2],zorder=0)
    
    ax[0,1].vlines((epoch[imax1]-epoch_center)/1000000,-ylim,ylim,color="red",alpha=0.5*secondary_present[0],zorder=0)
    ax[1,1].vlines((epoch[imax2]-epoch_center)/1000000,-ylim,ylim,color="red",alpha=0.5*secondary_present[1],zorder=0)
    ax[2,1].vlines((epoch[imax3]-epoch_center)/1000000,-ylim,ylim,color="red",alpha=0.5*secondary_present[2],zorder=0)
    
    #body peaks
    for axs in [ax[0,0],ax[1,0],ax[2,0]]:
        axs.vlines((epoch[index_body]-epoch_center)/1000,-ylim,ylim,color="blue",alpha=0.5,zorder=0)
    for axs in [ax[0,1],ax[1,1],ax[2,1]]:
        axs.vlines((epoch[index_body]-epoch_center)/1000000,-ylim,ylim,color="blue",alpha=0.5,zorder=0)
    
    #print max amplitudes for ions and electrons
    ax[0,0].text(xmax-100,0.3*ylim,str(np.round(1000*waveforms[0][0][imax1],1))+" mV",fontsize="small",horizontalalignment='right',color="red",alpha = 1.0*secondary_present[0])
    ax[1,0].text(xmax-100,0.3*ylim,str(np.round(1000*waveforms[0][1][imax2],1))+" mV",fontsize="small",horizontalalignment='right',color="red",alpha = 1.0*secondary_present[1])
    ax[2,0].text(xmax-100,0.3*ylim,str(np.round(1000*waveforms[0][2][imax3],1))+" mV",fontsize="small",horizontalalignment='right',color="red",alpha = 1.0*secondary_present[2])
    ax[0,0].text(xmax-100,0.6*ylim,str(np.round(1000*waveforms[0][0][index_body],1))+" / "+str(np.round(1000*amplitude_body,1))+"mV",fontsize="small",horizontalalignment='right',color="blue")
    ax[1,0].text(xmax-100,0.6*ylim,str(np.round(1000*waveforms[0][1][index_body],1))+" / "+str(np.round(1000*amplitude_body,1))+" mV",fontsize="small",horizontalalignment='right',color="blue")
    ax[2,0].text(xmax-100,0.6*ylim,str(np.round(1000*waveforms[0][2][index_body],1))+" / "+str(np.round(1000*amplitude_body,1))+" mV",fontsize="small",horizontalalignment='right',color="blue")
    
    if len(additional_vlines)>0:
        for i in range(len(additional_vlines)):
            index = additional_vlines[i]
            icolor = additional_vlines_colors[i]
            for axis in ax[:,0]:
                axis.vlines((epoch[index]-epoch_center)/1000,-ylim,ylim,color=icolor,ls="solid",alpha=0.5,zorder=0)
            for axis in ax[:,1]:
                axis.vlines((epoch[index]-epoch_center)/1000000,-ylim,ylim,color=icolor,ls="solid",alpha=0.5,zorder=0)
    
    if len(additional_hlines)>0:
        for i in range(len(additional_hlines)):
            ivalue = additional_hlines[i]
            icolor = additional_hlines_colors[i]
            for axis in ax[:,0]:
                axis.hlines(ivalue,-2000,2000,color=icolor,ls="solid",alpha=0.5,zorder=0)
            
    name = name_prefix + "_event_"+str(index_local)+"_of_"+str(index_count_day)
    fig.suptitle(name+" \n "+reason)
    
    fig.tight_layout()
    
    if save:
        plt.savefig(folder+"waveforms_analytical/"+name+".pdf", format='pdf')
    else:
        plt.show()
    plt.close()


def plot_flexible(waveforms,    #list of arrays: [[ch1,ch2,ch3],[ch1,ch2,ch3],...]
                    epoch,      #x-axis = time
                    index_center, #position of the view centering and of the most reliable peak
                    event_index,     #indexing for name
                    max_index,     #number of events on the date
                    waveform_colors = ["blue","red","orange","green","grey","magenta","teal","firebrick","mint","navy"],
                    waveform_styles = ["solid","solid","solid","solid","solid","solid","solid","solid","solid","solid"],
                    waveform_alphas = np.ones(10,dtype=float),
                    waveform_labels = ["line1","line2","line3","line4","line5","line6","line7","line8","line9","line10"],
                    save=False,      #to save or not to save 
                    to_label = False,
                    additional_vlines=[],    
                    additional_vlines_colors=[],
                    additional_vlines_strokes=[],
                    additional_vlines_where = [],
                    additional_hlines=[],
                    additional_hlines_colors=[],
                    xrange = [-900,1600],
                    folder = "",
                    name_prefix = "lorem_ipsum",
                    name_suffix = "dolor_sit_amet"):
    
    
    subplots = construct_subplots()
    fig = subplots[0]
    ax = subplots[1]
    
    #range is set by the first waveform
    ylim = 1.1*max([max(np.abs(waveforms[0][0])),max(np.abs(waveforms[0][1])),max(np.abs(waveforms[0][2]))])
    xmin = xrange[0]
    xmax = xrange[1]   
    for axis in ax[:,0]:
        axis.set_ylim(-ylim,ylim)
        axis.set_xlim(xmin,xmax)
    for axis in ax[:,1]:
        axis.set_ylim(-ylim,ylim)
        axis.axvspan(xmin/1000, xmax/1000, alpha=0.6, color='lightgrey')
        #axis.vlines(xmin/1000,-ylim,ylim,color="black",alpha=0.3)
        #axis.vlines(xmax/1000,-ylim,ylim,color="black",alpha=0.3)
    
    #time windows is centered about the most important peak
    epoch_center = epoch[index_center]
    
    for wf in range(len(waveforms)):
        ch1 = waveforms[wf][0]
        ch2 = waveforms[wf][1]
        ch3 = waveforms[wf][2]
    
        ax[0,0].plot((epoch-epoch_center)/1000,ch1,color=waveform_colors[wf],
                     lw=0.5,alpha=waveform_alphas[wf],ls=waveform_styles[wf])
        ax[1,0].plot((epoch-epoch_center)/1000,ch2,color=waveform_colors[wf],
                     lw=0.5,alpha=waveform_alphas[wf],ls=waveform_styles[wf])
        ax[2,0].plot((epoch-epoch_center)/1000,ch3,color=waveform_colors[wf],
                     lw=0.5,alpha=waveform_alphas[wf],ls=waveform_styles[wf])
    
        ax[0,1].plot((epoch-epoch_center)/1000000,ch1,color=waveform_colors[wf],
                     lw=0.5,alpha=waveform_alphas[wf],ls=waveform_styles[wf],
                     label = waveform_labels[wf])
        ax[1,1].plot((epoch-epoch_center)/1000000,ch2,color=waveform_colors[wf],
                     lw=0.5,alpha=waveform_alphas[wf],ls=waveform_styles[wf])
        ax[2,1].plot((epoch-epoch_center)/1000000,ch3,color=waveform_colors[wf],
                     lw=0.5,alpha=waveform_alphas[wf],ls=waveform_styles[wf])
    
    if to_label:
        ax[0,1].legend(loc=1)
    
    if len(additional_vlines)>0:
        if len(additional_vlines_strokes)<len(additional_vlines):
            additional_vlines_strokes = ["dashed"]*len(additional_vlines)
        for i in range(len(additional_vlines)):
            index = additional_vlines[i]
            icolor = additional_vlines_colors[i]
            istroke = additional_vlines_strokes[i]
            where = additional_vlines_where[i]
            if where == 4:
                for axis in ax[:,0]:
                    axis.vlines((epoch[index]-epoch_center)/1000,-ylim,ylim,color=icolor,ls=istroke,alpha=0.5,zorder=0)
                for axis in ax[:,1]:
                    axis.vlines((epoch[index]-epoch_center)/1000000,-ylim,ylim,color=icolor,ls=istroke,alpha=0.5,zorder=0)
            else:
                ax[where,0].vlines((epoch[index]-epoch_center)/1000,-ylim,ylim,color=icolor,ls=istroke,alpha=0.5,zorder=0)
                ax[where,1].vlines((epoch[index]-epoch_center)/1000000,-ylim,ylim,color=icolor,ls=istroke,alpha=0.5,zorder=0)
            
        
    if len(additional_hlines)>0:
        for i in range(len(additional_hlines)):
            ivalue = additional_hlines[i]
            icolor = additional_hlines_colors[i]
            for axis in ax[:,0]:
                axis.hlines(ivalue,-2000,2000,color=icolor,ls="dashed",alpha=0.5,zorder=0)
            
    name = name_prefix + "_event_" + str(event_index) + "_of_" + str(max_index)
    fig.suptitle(name)
    
    fig.tight_layout()
    
    if save:
        plt.savefig(folder+"waveforms_instructive/"+name+"_"+name_suffix+".pdf", format='pdf')
    else:
        plt.show()
    plt.close()


def add_ternary(ax,
                amps, #1D len 3
                min_x,
                max_x,
                min_y,
                max_y):
    
    amps = np.vstack((amps,amps))
    amps_norm = np.transpose(1/np.sum(amps,axis=1)*amps.transpose())
    
    transform = np.array([[1., 0.5],
                          [0., np.sqrt(0.75)]])
    
    xy = np.matmul(amps_norm[:,[2,0]],transform.transpose())
    
    triaxes = plt.Polygon(np.array([(min_x,min_y),
                                    (max_x,min_y),
                                    ((min_x+max_x)/2,
                                         np.sqrt(0.75)*(max_y-min_y)+min_y)]), 
                          facecolor="white", edgecolor="black", 
                          lw=0.5, zorder = 5)
    #gridlines = add_gridlines_ternary(ax)
    ax.add_artist(triaxes)

    ax.plot(np.array([0,0.75])*(max_x-min_x)+min_x,
            np.array([0,np.sqrt(0.75)/2])*(max_y-min_y)+min_y,
            c="gray",ls="dashed",alpha=0.5,zorder=6)
    ax.plot(np.array([1,0.25])*(max_x-min_x)+min_x,
            np.array([0,np.sqrt(0.75)/2])*(max_y-min_y)+min_y,
            c="gray",ls="dashed",alpha=0.5,zorder=6)
    ax.plot(np.array([0.5,0.5])*(max_x-min_x)+min_x,
            np.array([0,np.sqrt(0.75)])*(max_y-min_y)+min_y,
            c="gray",ls="dashed",alpha=0.5,zorder=6)


    ax.scatter(xy[:,0]*(max_x-min_x)+min_x,xy[:,1]*(max_y-min_y)+min_y,
                          s=3,alpha=1,c="blue",zorder=7)


#waveforms plotting routine
def plot_waveforms_minimal(waveforms,    #arrays: [ch1,ch2,ch3]
                           epoch,      #x-axis = time
                           index_center, #position of the view centering and of the most reliable peak
                           index_body, #position of the body peak
                           index_global,index_local,index_count_day,     #indexing for name
                           waveform_color = "blue",
                           waveform_alpha = 1,
                           reason = " ",
                           ternary = False,
                           save = False,      #to save or not to save 
                           folder = "",
                           name_prefix = "Lorem Ipsum"):
    
    subplots = construct_subplots()
    fig = subplots[0]
    ax = subplots[1]
    
    #range is set by the first waveform
    ylim = 1.1*max([max(np.abs(waveforms[0])),
                    max(np.abs(waveforms[1])),
                    max(np.abs(waveforms[2]))])
    xmin = -900
    xmax = +1600   
    for axis in ax[:,0]:
        axis.set_ylim(-ylim,ylim)
        axis.set_xlim(xmin,xmax)        
    for axis in ax[:,1]:
        axis.set_ylim(-ylim,ylim)
        axis.axvspan(xmin/1000, xmax/1000, alpha=0.6, color='lightgrey')
        #axis.vlines(xmin/1000,-ylim,ylim,color="black",alpha=0.3)
        #axis.vlines(xmax/1000,-ylim,ylim,color="black",alpha=0.3)

    
    #time windows is centered about the most important peak
    epoch_center = epoch[index_center]
    

    ch1 = waveforms[0]
    ch2 = waveforms[1]
    ch3 = waveforms[2]

    #zoom in
    ax[0,0].plot((epoch-epoch_center)/1000,ch1,
                 color=waveform_color,lw=0.5,alpha=waveform_alpha,zorder=2)
    ax[1,0].plot((epoch-epoch_center)/1000,ch2,
                 color=waveform_color,lw=0.5,alpha=waveform_alpha,zorder=2)
    ax[2,0].plot((epoch-epoch_center)/1000,ch3,
                 color=waveform_color,lw=0.5,alpha=waveform_alpha,zorder=2)

    #zoom out
    ax[0,1].plot((epoch-epoch_center)/1000000,ch1,
                 color=waveform_color,lw=0.5,alpha=waveform_alpha,zorder=2)
    ax[1,1].plot((epoch-epoch_center)/1000000,ch2,
                 color=waveform_color,lw=0.5,alpha=waveform_alpha,zorder=2)
    ax[2,1].plot((epoch-epoch_center)/1000000,ch3,
                 color=waveform_color,lw=0.5,alpha=waveform_alpha,zorder=2)
    
    #ternary overlay PIP
    if ternary:
        #left or righ
        if max((epoch-epoch_center)/1000000)>30: #rhs
            max_x = ax[0,1].get_xlim()[1]-2
            min_x = ax[0,1].get_xlim()[1]-19
            min_y = ax[0,1].get_ylim()[0]*0.65
            max_y = ax[0,1].get_ylim()[1]*0.95
        else: #lhs
            max_x = ax[0,1].get_xlim()[0]+20
            min_x = ax[0,1].get_xlim()[0]+3
            min_y = ax[0,1].get_ylim()[0]*0.65
            max_y = ax[0,1].get_ylim()[1]*0.95
        #tbd plot ternary
        rect = mpl.patches.Rectangle((min_x, max_y), max_x-min_x, min_y-max_y, 
                                     linewidth=1, edgecolor='none', 
                                     zorder = 2, facecolor='white')
        ax[0,1].add_patch(rect)
        add_ternary(ax[0,1], [np.max(ch1),np.max(ch2),np.max(ch3)], 
                    min_x, max_x, min_y, max_y)
        
        
    #body peaks
    for axs in [ax[0,0],ax[1,0],ax[2,0]]:
        axs.vlines((epoch[index_body]-epoch_center)/1000,
                   -ylim,ylim,color="grey",alpha=0.5,zorder=0)
        axs.hlines(0,
                   xmin,xmax,color="grey",alpha=0.5,zorder=0)
    for axs in [ax[0,1],ax[1,1],ax[2,1]]:
        # axs.vlines((epoch[index_body]-epoch_center)/1000000,
        #             -ylim,ylim,color="grey",alpha=0.5,zorder=0)
        axs.hlines(0,
                    min((epoch-epoch_center)/1000000),
                    max((epoch-epoch_center)/1000000),
                    color="grey",alpha=0.5,zorder=0)
                
    name = name_prefix + "_event_"+str(index_local)+"_of_"+str(index_count_day)
    fig.suptitle(name+" \n "+reason)
    
    fig.tight_layout()
    
    if save:
        plt.savefig(folder+"waveforms_minimal/"+name+".pdf", format='pdf')
    else:
        plt.show()
    plt.close()


#spectra plotting routine    
def plot_spectra(orig,
                 filtered,
                 corrected,
                 time_step,
                 index_global,index_local,index_count_day,     #indexing for name
                 folder,
                 nperseg = 4096,
                 save = True,
                 name_prefix = "Lorem Ipsum"):
    #if it is necessary to inspect the spectra, look at number 39
    psd_orig=welch(orig,fs=(1/(time_step/1e9)),nperseg=nperseg)
    psd_smooth=welch(filtered,fs=(1/(time_step/1e9)),nperseg=nperseg)
    psd_corrected=welch(corrected,fs=(1/(time_step/1e9)),nperseg=nperseg)
    
    fig,ax = plt.subplots()
    ax.loglog(psd_orig[0][1:],psd_orig[1][1:],"red",label="Original")
    #ax.loglog(psd_smooth[0][1:],psd_smooth[1][1:],"blue",ls="dashed",label="Filtered")
    ax.loglog(psd_smooth[0][1:],psd_corrected[1][1:],"black",ls="solid",label="Corrected")
    ax.legend()
    ax.set_ylim(1e-17,1e-5)
    ax.set_xlabel("Frequency [Hz]")
    
    name = name_prefix + "_event_"+str(index_local)+"_of_"+str(index_count_day)
    fig.suptitle(name)
    
    fig.tight_layout()
    
    ax.set_position([0.16,0.17,0.75,0.662])
    
    if save:
        plt.savefig(folder+"spectra/"+name+".pdf", format='pdf')
    else:
        plt.show()
    plt.close()
    

def plot_extraction_process(YYYYMMDD,
                            index,
                            isave=True,
                            folder="/997_data/solo_features/plots/"):



        amplitudes = np.zeros((0,3))       #global maxima
        polarities = np.zeros(0)
        stupid_max = np.zeros((0,3))   #stupid max - median
        ant_decay_times = np.zeros(0)
        maxWF = np.zeros((0,3))    #jakubs analysis
        ant_decay_times_alt = np.zeros(0)
        body_decay_times = np.zeros(0)
        negative_prespikes = np.zeros(0)
        body_risetimes = np.zeros(0)
        looks_good = np.zeros(0,dtype="bool")
        reasons = []
        saturated = np.zeros(0,dtype="bool")
        
        try:            
            cdf_file_e = fetch(YYYYMMDD,
                         '/997_data/solo_rpw/tds_wf_e',
                         "tds_wf_e",
                         "_rpw-tds-surv-tswf-e_",
                         ["V06.cdf","V05.cdf","V04.cdf","V03.cdf","V02.cdf","V01.cdf"],    
                         True)
            
            e = cdf_file_e.varget("WAVEFORM_DATA_VOLTAGE") #[event,channel,time], channel = 2 is monopole in XLD1
            epoch = cdf_file_e.varget("EPOCH")              #start of each impact
            epoch_offset = cdf_file_e.varget("EPOCH_OFFSET")     #time variable for each sample of each impact
            sw = cdf_file_e.attget("Software_version",entry=0)["Data"]    #should be 2.1.1 or higher
            
        except:
            print("no file on "+YYYYMMDD)
        else:
            i=index
            reason = " "

            #time step will be handy later
            time_step = epoch_offset[i][1]-epoch_offset[i][0]
            
            #original vaweforms, as Jakub calls it
            WF1 = e[i,0,:]
            WF2 = e[i,1,:]
            WF3 = e[i,2,:]
            
            #highest absolute value in original waveforms, sign respective
            maxWF1 = WF1[np.argmax(np.abs(WF1))]
            maxWF2 = WF2[np.argmax(np.abs(WF2))]
            maxWF3 = WF3[np.argmax(np.abs(WF3))]
            maxWF = np.vstack((maxWF,[maxWF1,maxWF2,maxWF3]))
            
            #in case of true monopole SE1 data
            e1wf = e[i,2,:]-e[i,1,:]
            e2wf = e[i,2,:]
            e3wf = e[i,2,:]-e[i,1,:]-e[i,0,:]
            
            #backgrounds - shouldn't use median because we do a cumsum later
            bg1 = np.mean(e1wf)  
            bg2 = np.mean(e2wf)
            bg3 = np.mean(e3wf)
            
            #smoothed monopole wavefroms without backgrounds
            sos = butter(32,7e4,btype="lowpass",fs=(1/(time_step/1e9)),output="sos")
            smooth_1 = sosfilt(sos, e1wf)-bg1
            smooth_2 = sosfilt(sos, e2wf)-bg2
            smooth_3 = sosfilt(sos, e3wf)-bg3
               
            #lag removal
            smooth_1 = np.append(smooth_1[10:],10*[smooth_1[-1]])
            smooth_2 = np.append(smooth_2[10:],10*[smooth_2[-1]])
            smooth_3 = np.append(smooth_3[10:],10*[smooth_3[-1]])


            #corrected for the highpass artefact - is okay to do to sums, as each individual channel needs the same correction
            corrected_1 = hipass_correction(smooth_1,time_step/1e9)
            corrected_2 = hipass_correction(smooth_2,time_step/1e9)
            corrected_3 = hipass_correction(smooth_3,time_step/1e9)
            
            calibrate = True
            if calibrate:
                smooth_1 = corrected_1-np.mean(corrected_1)
                smooth_2 = corrected_2-np.mean(corrected_2)
                smooth_3 = corrected_3-np.mean(corrected_3)

            #time index of the global maximum
            time_index_1 = np.argmax(np.abs(smooth_1))
            time_index_2 = np.argmax(np.abs(smooth_2))
            time_index_3 = np.argmax(np.abs(smooth_3))
            
            #stupid ion
            stupid_max_1 = np.max(smooth_1)
            stupid_max_2 = np.max(smooth_2)
            stupid_max_3 = np.max(smooth_3)
            stupid_max = np.vstack((stupid_max,[stupid_max_1,stupid_max_2,stupid_max_3]))
            
            #provisional amplitudes
            e10 = smooth_1[time_index_1]
            e20 = smooth_2[time_index_2]
            e30 = smooth_3[time_index_3]
            
            #where to center the window
            index_center = [time_index_1,time_index_2,time_index_3][np.argmax([abs(e10),abs(e20),abs(e30)])]
            
            #make the first plot
            plot_flexible([[smooth_1,smooth_2,smooth_3],[e1wf,e2wf,e3wf]],
                              epoch_offset[i], 
                              index_center, #position of the view centering and of the most reliable peak
                              index,     #indexing for name
                              len(epoch), #number of event on the day
                              waveform_colors = ["black","red","orange","green","grey","magenta","teal","firebrick","mint","navy"],
                              waveform_alphas = np.ones(10,dtype=float),
                              xrange = [-450,800],
                              waveform_labels = ["Corrected","Original"],
                              to_label = True,
                              save=True,      #to save or not to save 
                              additional_vlines=[index_center],    
                              additional_vlines_colors=["grey"],
                              additional_vlines_where = [4],
                              additional_hlines=[0],
                              additional_hlines_colors=["grey"],
                              folder = "/997_data/solo_features/plots/",
                              name_prefix = cdf_file_e.attget(attribute='access_url',entry=0)["Data"][-16:-4],
                              name_suffix = "first_step")
            
            

            
            
            
            

            deriv2_1 = savgol_filter(np.maximum(smooth_1,0),17,2,deriv=2)*100
            deriv2_2 = savgol_filter(np.maximum(smooth_2,0),17,2,deriv=2)*100
            deriv2_3 = savgol_filter(np.maximum(smooth_3,0),17,2,deriv=2)*100
            
            #look for the minimum, but only in a window around index_center
            min_deriv2_1 = np.argmin(deriv2_1[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
            min_deriv2_2 = np.argmin(deriv2_2[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
            min_deriv2_3 = np.argmin(deriv2_3[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
            
            #the chronologically first
            body_index = min([min_deriv2_1,min_deriv2_2,min_deriv2_3]) #provisional
            
            #the largest of them
            #body_index = [min_deriv2_1,min_deriv2_2,min_deriv2_3][np.argmax([deriv2_1[min_deriv2_1],deriv2_2[min_deriv2_2],deriv2_3[min_deriv2_3]])] #provisional
            
            #time index of the electron prespike
            prespike = False
            prespike_index = np.argmin((smooth_1+smooth_2+smooth_3)[max(body_index-50,0):body_index+1])+max(body_index-50,0)
            prespike_value = ((smooth_1+smooth_2+smooth_3)/3)[prespike_index]
            prespike_mu = np.mean(((smooth_1+smooth_2+smooth_3)/3)[max(body_index-200,0):max(body_index-10,1)])
            prespike_sigma = (np.var(((smooth_1+smooth_2+smooth_3)/3)[max(body_index-200,0):max(body_index-10,1)]))**0.5
            if prespike_value < prespike_mu - 3*prespike_sigma:
                prespike = True
                negative_prespikes = np.append(negative_prespikes,prespike_value)
            else:
                negative_prespikes = np.append(negative_prespikes,np.nan)
                
                
            #make the second plot
            #    plot the provisional body peak and the prespike    
            plot_flexible([[smooth_1,smooth_2,smooth_3],[deriv2_1,deriv2_2,deriv2_3]],
                              epoch_offset[i], 
                              index_center, #position of the view centering and of the most reliable peak
                              index,     #indexing for name
                              len(epoch), #number of event on the day
                              waveform_colors = ["black","darkorange","green","grey","magenta","teal","firebrick","mint","navy"],
                              waveform_alphas = [1,1,0,0,0,0,0,0,0],
                              xrange = [-450,800],
                              waveform_labels = ["Corrected","2nd derivative"],
                              waveform_styles = [(0,(1,1)),"solid"],
                              to_label = True,
                              save=True,      #to save or not to save 
                              additional_vlines=[body_index,prespike_index],    
                              additional_vlines_colors=["blue","green"],
                              additional_vlines_strokes=["dashed","dotted"],
                              additional_vlines_where = [4,4],
                              additional_hlines=[0],
                              additional_hlines_colors=["grey"],
                              folder = "/997_data/solo_features/plots/",
                              name_prefix = cdf_file_e.attget(attribute='access_url',entry=0)["Data"][-16:-4],
                              name_suffix = "second_step")    
                
                
                
    
                
                
            #correct for more local background
            lower_limit = np.max([0,prespike_index-100])
            smooth_1 = smooth_1-np.mean(smooth_1[lower_limit:prespike_index+1])
            smooth_2 = smooth_2-np.mean(smooth_2[lower_limit:prespike_index+1])
            smooth_3 = smooth_3-np.mean(smooth_3[lower_limit:prespike_index+1])
            
            #precise body index identification using min in 1st deriv around the provisional body index
            lower_limit = np.max([1,body_index-5])
            upper_limit = np.min([len(smooth_1)-2,body_index+20])
            #body_index_1 = lower_limit+np.argmin(np.abs(smooth_1[lower_limit+1:upper_limit]-smooth_1[lower_limit-1:upper_limit-2]))
            #body_index_2 = lower_limit+np.argmin(np.abs(smooth_2[lower_limit+1:upper_limit]-smooth_2[lower_limit-1:upper_limit-2]))
            #body_index_3 = lower_limit+np.argmin(np.abs(smooth_3[lower_limit+1:upper_limit]-smooth_3[lower_limit-1:upper_limit-2]))
            
            deriv1_1 = savgol_filter(smooth_1,11,2,deriv=1)
            deriv1_2 = savgol_filter(smooth_2,11,2,deriv=1)
            deriv1_3 = savgol_filter(smooth_3,11,2,deriv=1)
            
            body_index_1 = lower_limit+np.argmin(np.abs(deriv1_1[lower_limit:upper_limit]))-1
            body_index_2 = lower_limit+np.argmin(np.abs(deriv1_2[lower_limit:upper_limit]))-1
            body_index_3 = lower_limit+np.argmin(np.abs(deriv1_3[lower_limit:upper_limit]))-1
            
            #max correlation with an arc or something
            if body_index>10 and body_index<len(smooth_1)-30:
                body_index_1 = lower_limit + highest_corr(smooth_1[lower_limit-5:lower_limit+31])
                body_index_2 = lower_limit + highest_corr(smooth_2[lower_limit-5:lower_limit+31])
                body_index_3 = lower_limit + highest_corr(smooth_3[lower_limit-5:lower_limit+31])
            
            body_amp_1 = smooth_1[body_index]
            body_amp_2 = smooth_2[body_index]
            body_amp_3 = smooth_3[body_index]
            strongest_body = np.argmax([body_amp_1,body_amp_2,body_amp_3])
            
            if strongest_body == 0:
                body_index = int(np.round(np.mean([body_index_2,body_index_3,body_index])))
            elif strongest_body == 1:
                body_index = int(np.round(np.mean([body_index_1,body_index_3,body_index])))
            else:
                body_index = int(np.round(np.mean([body_index_1,body_index_2,body_index])))  
            
            
            #find the parameters for the body peak
            #amplitude
            body_amp_1 = smooth_1[body_index]
            body_amp_2 = smooth_2[body_index]
            body_amp_3 = smooth_3[body_index]
            body_amp = ( body_amp_1+body_amp_2+body_amp_3 - np.max([body_amp_1,body_amp_2,body_amp_3]) )/2 #weaker 2 of the 3
            strongest_body = np.argmax([body_amp_1,body_amp_2,body_amp_3])
            #FWHM
            try:
                body_hwhm_1 = 20-np.argmin(np.abs(smooth_1[body_index-20:body_index+1]-smooth_1[body_index]/2))
                body_hwhm_2 = 20-np.argmin(np.abs(smooth_2[body_index-20:body_index+1]-smooth_2[body_index]/2))
                body_hwhm_3 = 20-np.argmin(np.abs(smooth_3[body_index-20:body_index+1]-smooth_3[body_index]/2))
                body_hwhms = [body_hwhm_1,body_hwhm_2,body_hwhm_3]
                body_hwhms.pop(strongest_body)
                body_fwhm = np.sum(body_hwhms) #the same as 2*average, since we have 2 elements only
                #define body approx wavefrom
                body_waveform = body_amp*gaussian_body(np.arange(len(smooth_1)),mu=body_index,sig=body_fwhm/2.355)
            except:
                body_waveform = np.zeros(len(smooth_1))
                reason += "no_body_peak "
                
                
                
            #make the third plot    
            plot_flexible([[smooth_1,smooth_2,smooth_3],[body_waveform,body_waveform,body_waveform]],
                              epoch_offset[i], 
                              index_center, #position of the view centering and of the most reliable peak
                              index,     #indexing for name
                              len(epoch), #number of event on the day
                              waveform_colors = ["black","slateblue","grey","magenta","teal","firebrick","mint","navy"],
                              waveform_alphas = [1,1,0,0,0,0,0,0,0],
                              xrange = [-450,800],
                              waveform_labels = ["Corrected","Primary peak"],
                              waveform_styles = [(0,(1,1)),(0,(6,2))],
                              to_label = True,
                              save=True,      #to save or not to save 
                              additional_vlines=[body_index,prespike_index],    
                              additional_vlines_colors=["blue","green"],
                              additional_vlines_where = [4,4],
                              additional_hlines=[0],
                              additional_hlines_colors=["grey"],
                              folder = "/997_data/solo_features/plots/",
                              name_prefix = cdf_file_e.attget(attribute='access_url',entry=0)["Data"][-16:-4],
                              name_suffix = "third_step")    
                

  
                
            
            #get ion waveforms as what is left when we get rid of the body waveform
            antenna_waveform_1 = smooth_1 - body_waveform
            antenna_waveform_2 = smooth_2 - body_waveform
            antenna_waveform_3 = smooth_3 - body_waveform
            
            #time index of the global maximum considering the body peak
            upper_limit = np.min([body_index+300,len(antenna_waveform_1)])
            antenna_max_1 = body_index+np.argmax(antenna_waveform_1[body_index:upper_limit])
            antenna_max_2 = body_index+np.argmax(antenna_waveform_2[body_index:upper_limit])
            antenna_max_3 = body_index+np.argmax(antenna_waveform_3[body_index:upper_limit])
            
            #are there actually secondary peaks?
            secondary_1 = (np.max(antenna_waveform_1[body_index:upper_limit])) / (body_amp) > 0.75
            secondary_2 = (np.max(antenna_waveform_2[body_index:upper_limit])) / (body_amp) > 0.75
            secondary_3 = (np.max(antenna_waveform_3[body_index:upper_limit])) / (body_amp) > 0.75
            
            #ampltiudes
            amplitudes = np.vstack((amplitudes,[smooth_1[antenna_max_1],smooth_2[antenna_max_2],smooth_3[antenna_max_3]]))
            
            #make the fourth plot    
            plot_flexible([[smooth_1,smooth_2,smooth_3],
                           [body_waveform,body_waveform,body_waveform],
                           [antenna_waveform_1,antenna_waveform_2,antenna_waveform_3]],
                              epoch_offset[i], 
                              index_center, #position of the view centering and of the most reliable peak
                              index,     #indexing for name
                              len(epoch), #number of event on the day
                              waveform_colors = ["black","slateblue","firebrick","magenta","teal","firebrick","mint","navy"],
                              waveform_alphas = [1,1,1,0,0,0,0,0,0],
                              xrange = [-450,800],
                              waveform_labels = ["Corrected","Primary peak","Secondary peak"],
                              to_label = True,
                              waveform_styles = [(0,(1,1)),(0,(6,2)),"solid"],
                              save=True,      #to save or not to save 
                              additional_vlines=[body_index,prespike_index,
                                                 antenna_max_1,
                                                 antenna_max_2,
                                                 antenna_max_3],    
                              additional_vlines_colors=["blue","green",
                                                        "red"*secondary_1+"none"*(1-secondary_1),
                                                        "red"*secondary_2+"none"*(1-secondary_2),
                                                        "red"*secondary_3+"none"*(1-secondary_3)],
                              additional_vlines_where = [4,4,0,1,2],
                              additional_hlines=[0],
                              additional_hlines_colors=["grey"],
                              folder = "/997_data/solo_features/plots/",
                              name_prefix = cdf_file_e.attget(attribute='access_url',entry=0)["Data"][-16:-4],
                              name_suffix = "fourth_step")  
            

def main(target_input_txt,target_output_npz):

    #scpot dates
    scpot_dates = []

    amplitudes = np.zeros((0,3))       #global maxima
    ion_amplitudes = np.zeros((0,3))   #amplitude of ion peak, minus electron peak
    electron_amplitudes = np.zeros(0)  #amplitude of electron peak
    delays = np.zeros((0,3))           #delay of ions after electrons
    secondary_present = np.zeros((0,3))#whether there are secondary peaks in the individual channels
    bias_current = np.zeros((0,3))     #in A, on each of the three antennas
    polarities = np.zeros(0)
    stupid_max = np.zeros((0,3))   #stupid max - median
    ant_decay_times = np.zeros(0)
    maxWF = np.zeros((0,3))    #jakubs analysis
    ant_decay_times_alt = np.zeros(0)
    body_decay_times = np.zeros(0)
    negative_prespikes = np.zeros(0)
    xld_risetimes = np.zeros((0,3))
    negative_prespikes = np.zeros(0)
    heliocentric_distances = []
    azimuthal_velocities = []
    radial_velocities = []
    ids = np.zeros(0)
    epochs = np.zeros(0)
    scpots = []
    body_risetimes = np.zeros(0)
    event_id = np.zeros(0)
    looks_good = np.zeros(0,dtype="bool")
    reasons = []
    alright = np.zeros(0,dtype="bool")
    saturated = np.zeros(0,dtype="bool")
    overshoot = np.zeros((0,3))
    overshoot_delay = np.zeros((0,3))

#for txt in txts[:100]:
    txt = target_input_txt

    #get the date
    YYYYMMDD = txt[-16:-8]

    #get the list of events with classification
    cnn_classification = pd.read_csv(txt)
    
    #is there something classified as dust?
    if np.isfinite(cnn_classification["Index"][0]):
        
        try:            
            cdf_file_e = fetch(YYYYMMDD,
                         '/997_data/solo_rpw/tds_wf_e',
                         "tds_wf_e",
                         "_rpw-tds-surv-tswf-e_",
                         ["V06.cdf","V05.cdf","V04.cdf","V03.cdf","V02.cdf","V01.cdf"],    
                         False)
            
            e = cdf_file_e.varget("WAVEFORM_DATA_VOLTAGE") #[event,channel,time], channel = 2 is monopole in XLD1
            channel_ref = cdf_file_e.varget("CHANNEL_REF")    #[13, 21, 20]
            quality_fact = cdf_file_e.varget("QUALITY_FACT")   #65535 = dust
            epoch = cdf_file_e.varget("EPOCH")              #start of each impact
            epoch_offset = cdf_file_e.varget("EPOCH_OFFSET")     #time variable for each sample of each impact
            sw = cdf_file_e.attget("Software_version",entry=0)["Data"]    #should be 2.1.1 or higher
            
            
            cdf_file_scpot = fetch(YYYYMMDD,
                             '/997_data/solo_rpw/lfr_scpot',
                             "lfr_scpot",
                             "_rpw-bia-scpot-10-seconds_",
                             ["V02.cdf","V01.cdf"],    
                             False)
            
            epoch_scpot = cdf_file_scpot.varget("EPOCH")              #10s frequency
            value_scpot = cdf_file_scpot.varget("SCPOT")              #10s frequency
            
            jd_scpot = tt2000_to_jd(epoch_scpot)
            f_scpot = interpolate.interp1d(jd_scpot[(value_scpot>-1e2)*(value_scpot<1e2)],value_scpot[(value_scpot>-1e2)*(value_scpot<1e2)],fill_value=0.0,bounds_error=False,kind=1)
            scpot_dates.append(YYYYMMDD) #mark that scpot is available on this date
            
        except:
            print("no file on "+YYYYMMDD)
        else:
            #reorder events by quality, epoch, because that is how Andreas reorders it
            plain_index = np.arange(len(e))
            flagged_dust = quality_fact==65535
            reordered_index = np.append(plain_index[flagged_dust==1],plain_index[flagged_dust==0])
        
            #list of dust/no dust flags from Andreas
            dust = np.array(cnn_classification["Label"])
            amplitude_candidates = np.array(cnn_classification["Amplitude"])
            dust = np.array(dust, dtype=bool)    
            indices_analyzed = np.array(cnn_classification["Index"])[dust]-1
            indices = reordered_index[indices_analyzed]   #the indices in the CDF file that correspond to dust according to Andreas
            
            #print and save waveforms
            for i in indices:
                #reason to exclude the waveform from anaylsis
                reason = " "
                
                #which channel is monopoloe? should be [2], or it is not an XLD1
                channels = cdf_file_e.varget("channel_ref")[i]
                monopole = np.array([0,1,2])[channels==20]
                
                #time step will be handy later
                time_step = epoch_offset[i][1]-epoch_offset[i][0]
                
                if monopole[0]!=2:
                    reason += "not_XLD1 "

                #original vaweforms, as Jakub calls it
                WF1 = e[i,0,:]
                WF2 = e[i,1,:]
                WF3 = e[i,2,:]
                
                #highest absolute value in original waveforms, sign respective
                maxWF1 = WF1[np.argmax(np.abs(WF1))]
                maxWF2 = WF2[np.argmax(np.abs(WF2))]
                maxWF3 = WF3[np.argmax(np.abs(WF3))]
                maxWF = np.vstack((maxWF,[maxWF1,maxWF2,maxWF3]))

                #monopole waveforms
                e1wf = e[i,2,:]-e[i,1,:]
                e2wf = e[i,2,:]
                e3wf = e[i,2,:]-e[i,1,:]-e[i,0,:]
                
                #backgrounds - shouldn't use median because we do a cumsum later
                bg1 = np.mean(e1wf)  
                bg2 = np.mean(e2wf)
                bg3 = np.mean(e3wf)
                
                #smoothed monopole wavefroms without backgrounds
                sos = butter(32,7e4,btype="lowpass",fs=(1/(time_step/1e9)),output="sos")
                smooth_1 = sosfilt(sos, e1wf)-bg1
                smooth_2 = sosfilt(sos, e2wf)-bg2
                smooth_3 = sosfilt(sos, e3wf)-bg3
                
                smooth_1 = np.append(smooth_1[10:],10*[smooth_1[-1]])
                smooth_2 = np.append(smooth_2[10:],10*[smooth_2[-1]])
                smooth_3 = np.append(smooth_3[10:],10*[smooth_3[-1]])
                
                smooth_1_uncorrected = np.copy(smooth_1)
                smooth_2_uncorrected = np.copy(smooth_2)
                smooth_3_uncorrected = np.copy(smooth_3)

                #corrected for the highpass artefact - is okay to do to sums, as each individual channel needs the same correction
                corrected_1 = hipass_correction(smooth_1,time_step/1e9)
                corrected_2 = hipass_correction(smooth_2,time_step/1e9)
                corrected_3 = hipass_correction(smooth_3,time_step/1e9)
                
                calibrate = True
                if calibrate:
                    smooth_1 = corrected_1-np.mean(corrected_1)
                    smooth_2 = corrected_2-np.mean(corrected_2)
                    smooth_3 = corrected_3-np.mean(corrected_3)

                #time index of the global maximum
                time_index_1 = np.argmax(np.abs(smooth_1))
                time_index_2 = np.argmax(np.abs(smooth_2))
                time_index_3 = np.argmax(np.abs(smooth_3))
                
                #stupid ion
                stupid_max_1 = np.max(smooth_1)
                stupid_max_2 = np.max(smooth_2)
                stupid_max_3 = np.max(smooth_3)
                stupid_max = np.vstack((stupid_max,[stupid_max_1,stupid_max_2,stupid_max_3]))
                
                #provisional amplitudes
                e10 = smooth_1[time_index_1]
                e20 = smooth_2[time_index_2]
                e30 = smooth_3[time_index_3]
                
                #where to center the window
                index_center = [time_index_1,time_index_2,time_index_3][np.argmax([abs(e10),abs(e20),abs(e30)])]

                deriv2_1 = savgol_filter(np.maximum(smooth_1,0),17,2,deriv=2)*100
                deriv2_2 = savgol_filter(np.maximum(smooth_2,0),17,2,deriv=2)*100
                deriv2_3 = savgol_filter(np.maximum(smooth_3,0),17,2,deriv=2)*100
                
                #look for the minimum, but only in a window around index_center
                min_deriv2_1 = np.argmin(deriv2_1[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
                min_deriv2_2 = np.argmin(deriv2_2[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
                min_deriv2_3 = np.argmin(deriv2_3[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
                
                #the chronologically first
                body_index = min([min_deriv2_1,min_deriv2_2,min_deriv2_3]) #provisional
                
                #the largest of them
                #body_index = [min_deriv2_1,min_deriv2_2,min_deriv2_3][np.argmax([deriv2_1[min_deriv2_1],deriv2_2[min_deriv2_2],deriv2_3[min_deriv2_3]])] #provisional
                
                #time index of the electron prespike
                prespike = False
                prespike_index = np.argmin((smooth_1+smooth_2+smooth_3)[max(body_index-50,0):body_index+1])+max(body_index-50,0)
                prespike_value = ((smooth_1+smooth_2+smooth_3)/3)[prespike_index]
                prespike_mu = np.mean(((smooth_1+smooth_2+smooth_3)/3)[max(body_index-200,0):max(body_index-10,1)])
                prespike_sigma = (np.var(((smooth_1+smooth_2+smooth_3)/3)[max(body_index-200,0):max(body_index-10,1)]))**0.5
                if prespike_value < prespike_mu - 3*prespike_sigma:
                    prespike = True
                    negative_prespikes = np.append(negative_prespikes,prespike_value)
                else:
                    negative_prespikes = np.append(negative_prespikes,np.nan)
                    
                #correct for more local background
                lower_limit = np.max([0,prespike_index-100])
                smooth_1 = smooth_1-np.mean(smooth_1[lower_limit:prespike_index+1])
                smooth_2 = smooth_2-np.mean(smooth_2[lower_limit:prespike_index+1])
                smooth_3 = smooth_3-np.mean(smooth_3[lower_limit:prespike_index+1])
                
                #precise body index identification using min in 1st deriv around the provisional body index
                lower_limit = np.max([1,body_index-5])
                upper_limit = np.min([len(smooth_1)-2,body_index+20])
                #body_index_1 = lower_limit+np.argmin(np.abs(smooth_1[lower_limit+1:upper_limit]-smooth_1[lower_limit-1:upper_limit-2]))
                #body_index_2 = lower_limit+np.argmin(np.abs(smooth_2[lower_limit+1:upper_limit]-smooth_2[lower_limit-1:upper_limit-2]))
                #body_index_3 = lower_limit+np.argmin(np.abs(smooth_3[lower_limit+1:upper_limit]-smooth_3[lower_limit-1:upper_limit-2]))
                
                deriv1_1 = savgol_filter(smooth_1,11,2,deriv=1)
                deriv1_2 = savgol_filter(smooth_2,11,2,deriv=1)
                deriv1_3 = savgol_filter(smooth_3,11,2,deriv=1)
                
                body_index_1 = lower_limit+np.argmin(np.abs(deriv1_1[lower_limit:upper_limit]))-1
                body_index_2 = lower_limit+np.argmin(np.abs(deriv1_2[lower_limit:upper_limit]))-1
                body_index_3 = lower_limit+np.argmin(np.abs(deriv1_3[lower_limit:upper_limit]))-1
                
                #max correlation with an arc or something
                if body_index>10 and body_index<len(smooth_1)-30:
                    body_index_1 = lower_limit + highest_corr(smooth_1[lower_limit-5:lower_limit+31])
                    body_index_2 = lower_limit + highest_corr(smooth_2[lower_limit-5:lower_limit+31])
                    body_index_3 = lower_limit + highest_corr(smooth_3[lower_limit-5:lower_limit+31])
                
                body_amp_1 = smooth_1[body_index]
                body_amp_2 = smooth_2[body_index]
                body_amp_3 = smooth_3[body_index]
                strongest_body = np.argmax([body_amp_1,body_amp_2,body_amp_3])
                
                if strongest_body == 0:
                    body_index = int(np.round(np.mean([body_index_2,body_index_3,body_index])))
                elif strongest_body == 1:
                    body_index = int(np.round(np.mean([body_index_1,body_index_3,body_index])))
                else:
                    body_index = int(np.round(np.mean([body_index_1,body_index_2,body_index])))  
                
                
                #find the parameters for the body peak
                #amplitude
                body_amp_1 = smooth_1[body_index]
                body_amp_2 = smooth_2[body_index]
                body_amp_3 = smooth_3[body_index]
                body_amp = ( body_amp_1+body_amp_2+body_amp_3 - np.max([body_amp_1,body_amp_2,body_amp_3]) )/2 #weaker 2 of the 3
                strongest_body = np.argmax([body_amp_1,body_amp_2,body_amp_3])
                #FWHM
                try:
                    body_hwhm_1 = 20-np.argmin(np.abs(smooth_1[body_index-20:body_index+1]-smooth_1[body_index]/2))
                    body_hwhm_2 = 20-np.argmin(np.abs(smooth_2[body_index-20:body_index+1]-smooth_2[body_index]/2))
                    body_hwhm_3 = 20-np.argmin(np.abs(smooth_3[body_index-20:body_index+1]-smooth_3[body_index]/2))
                    body_hwhms = [body_hwhm_1,body_hwhm_2,body_hwhm_3]
                    body_hwhms.pop(strongest_body)
                    body_fwhm = np.sum(body_hwhms) #the same as 2*average, since we have 2 elements only
                    #define body approx wavefrom
                    body_waveform = body_amp*gaussian_body(np.arange(len(smooth_1)),mu=body_index,sig=body_fwhm/2.355)
                except:
                    body_waveform = np.zeros(len(smooth_1))
                    reason += "no_body_peak "
                
                #get ion waveforms as what is left when we get rid of the body waveform
                antenna_waveform_1 = smooth_1 - body_waveform
                antenna_waveform_2 = smooth_2 - body_waveform
                antenna_waveform_3 = smooth_3 - body_waveform
                
                #time index of the global maximum considering the body peak
                upper_limit = np.min([body_index+300,len(antenna_waveform_1)])
                antenna_max_1 = body_index+np.argmax(antenna_waveform_1[body_index:upper_limit])
                antenna_max_2 = body_index+np.argmax(antenna_waveform_2[body_index:upper_limit])
                antenna_max_3 = body_index+np.argmax(antenna_waveform_3[body_index:upper_limit])
                
                #get delays for antenna peaks with respect to body peaks
                delay_1 = (antenna_max_1 - body_index)*time_step
                delay_2 = (antenna_max_2 - body_index)*time_step
                delay_3 = (antenna_max_3 - body_index)*time_step
  
                #are there actually secondary peaks?
                secondary_1 = (np.max(antenna_waveform_1[body_index:upper_limit])) / (body_amp) > 0.75
                secondary_2 = (np.max(antenna_waveform_2[body_index:upper_limit])) / (body_amp) > 0.75
                secondary_3 = (np.max(antenna_waveform_3[body_index:upper_limit])) / (body_amp) > 0.75
                
                #make them contain the actual amplitudes or 0 if not present
                secondary_1 *= np.max(antenna_waveform_1[body_index:upper_limit])
                secondary_2 *= np.max(antenna_waveform_2[body_index:upper_limit])
                secondary_3 *= np.max(antenna_waveform_3[body_index:upper_limit])
                
                #get the overshoots as local minima
                overshoot_upper_limit = np.min([body_index+1500,len(antenna_waveform_1)])
                overshoot_1 = np.min(smooth_1[body_index:overshoot_upper_limit])
                overshoot_2 = np.min(smooth_2[body_index:overshoot_upper_limit])
                overshoot_3 = np.min(smooth_3[body_index:overshoot_upper_limit])
                overshoot = np.vstack((overshoot,[overshoot_1,overshoot_2,overshoot_3]))
                overshoot_delay_1 = np.argmin(smooth_1[body_index:overshoot_upper_limit])*time_step
                overshoot_delay_2 = np.argmin(smooth_2[body_index:overshoot_upper_limit])*time_step
                overshoot_delay_3 = np.argmin(smooth_3[body_index:overshoot_upper_limit])*time_step
                overshoot_delay = np.vstack((overshoot_delay,[overshoot_delay_1,overshoot_delay_2,overshoot_delay_3]))
                
                #ampltiudes
                amplitudes = np.vstack((amplitudes,[smooth_1[antenna_max_1],smooth_2[antenna_max_2],smooth_3[antenna_max_3]]))
                
                #get the time scale of the ion peak decay to 1/e
                strongest_antenna = np.argmax([smooth_1[antenna_max_1],smooth_2[antenna_max_2],smooth_3[antenna_max_3]])
                strongest_antenna_time_index = np.array([antenna_max_1,antenna_max_2,antenna_max_3])[strongest_antenna]
                strongest_waveform = np.vstack((smooth_1,smooth_2,smooth_3))[strongest_antenna,:]
                threshold = (strongest_waveform[strongest_antenna_time_index])*(1/np.e)
                threshol_index = np.argmin(np.abs(strongest_waveform[strongest_antenna_time_index:strongest_antenna_time_index+250]))+strongest_antenna_time_index
                decay_time = (threshol_index - strongest_antenna_time_index)*3.81e-6 #s
                ant_decay_times = np.append(ant_decay_times,decay_time)
                ant_decay_times_alt = np.append(ant_decay_times_alt,decay_time) #legacy reasons
                
                #get the time scale of electron peak decay to 1/e
                weakest_body = np.argmin(np.array([smooth_1[body_index],
                                          smooth_2[body_index],
                                          smooth_3[body_index]]))
                weakest_waveform = np.vstack((smooth_1,smooth_2,smooth_3))[weakest_body,:]
                threshold_e = (weakest_waveform[body_index])*(1/np.e)
                threshol_index_e = np.argmin(np.abs(weakest_waveform[body_index:body_index+53]-threshold_e))+body_index
                decay_time_e = (threshol_index_e - body_index)*3.81e-6 #s
                if np.min(weakest_waveform[body_index:body_index+53])>threshold_e: #if not crossed
                    decay_time_e = 53*3.81e-6 #s
                body_decay_times = np.append(body_decay_times,decay_time_e)

                #XLD1 risetimes
                if body_index > 50 and body_index < len(WF1)-50:
                    WF1_bg = np.median(WF1[body_index-40:body_index-25])
                    WF2_bg = np.median(WF2[body_index-40:body_index-25])
                    WF3_bg = np.median(WF3[body_index-40:body_index-25])
                    
                    WF1f = sosfilt(sos, WF1)-WF1_bg
                    WF2f = sosfilt(sos, WF2)-WF2_bg
                    WF3f = sosfilt(sos, WF3)-WF3_bg
                    
                    WF1_index = body_index#-10 + np.argmax(WF1f[body_index-10:body_index+10])
                    WF2_index = body_index#-10 + np.argmax(WF2f[body_index-10:body_index+10])
                    WF3_index = body_index#-10 + np.argmax(WF3f[body_index-10:body_index+10])
                    
                    WF1_max = WF1f[WF1_index]
                    WF2_max = WF2f[WF2_index]
                    WF3_max = WF3f[WF3_index]
                    
                    if WF1_max>0 and WF2_max>0 and WF3_max>0:
                        WF1_threshold = (WF1_max)/np.e
                        WF2_threshold = (WF2_max)/np.e
                        WF3_threshold = (WF3_max)/np.e
                        
                        WF1_risetime = np.argmin(np.abs(WF1f[WF1_index-25:WF1_index]-WF1_threshold))*3.81e-6 #s
                        WF2_risetime = np.argmin(np.abs(WF2f[WF2_index-25:WF2_index]-WF2_threshold))*3.81e-6 #s
                        WF3_risetime = np.argmin(np.abs(WF3f[WF3_index-25:WF3_index]-WF3_threshold))*3.81e-6 #s
                        
                        xld_risetimes = np.vstack((xld_risetimes,[WF1_risetime,WF2_risetime,WF3_risetime]))
                    else:
                        xld_risetimes = np.vstack((xld_risetimes,[0,0,0]))
                else:
                    xld_risetimes = np.vstack((xld_risetimes,[0,0,0]))
                
                #electron peak risetime (time to get from 43% to 80%)
                mark_43 = body_amp*0.43
                mark_80 = body_amp*0.80
                strongest_electron_peak = np.argmax(np.array([smooth_1[body_index],smooth_2[body_index],smooth_3[body_index]]))                                     
                if strongest_electron_peak == 0:
                    smooth = smooth_2 + smooth_3
                elif strongest_electron_peak == 1:
                    smooth = smooth_1 + smooth_3
                elif strongest_electron_peak == 2:
                    smooth = smooth_2 + smooth_1
                else:
                    raise Exception()
                #define the window where to look for the timescale
                if body_index > 0:
                    index_43 = np.argmin(np.abs(smooth[max(body_index-100,0):body_index]-mark_43))
                    index_80 = np.argmin(np.abs(smooth[max(body_index-100,0):body_index]-mark_80))
                    body_risetime = (index_80 - index_43)*3.81e-6 #s
                else:
                    body_risetime = 0 #s
                body_risetimes = np.append(body_risetimes,body_risetime)

                 #correct for electron prespike
                if prespike:
                    body_amp -= prespike_value   
                    
                #positive if the most important waveform is positive and overall is positive
                upper_limit = np.min([len(smooth_1)-1,body_index+100])
                if np.array([e10,e20,e30])[strongest_antenna]>0 and np.mean((smooth_1+smooth_2+smooth_3)[body_index:upper_limit])>0:
                    if body_amp>0:
                        polarity = "pos"
                        polarities = np.append(polarities,1)
                    else:
                        polarity = "neg" 
                        polarities = np.append(polarities,-1)
                        reason += "neg "
                else:
                    polarity = "neg" 
                    polarities = np.append(polarities,-1)
                    reason += "neg "
                    
                if body_amp < 0:
                    polarity = "neg"
                    
                #decide if we cought the waveform
                if [body_index_1,body_index_2,body_index_3][strongest_body] < 100 or [time_index_1,time_index_2,time_index_3][strongest_body] > len(epoch_offset[0])-100:
                    reason += "missed_window "
                    
                #decide if the original signal starts in the tail
                tot = e1wf+e2wf+e3wf - np.mean(e1wf+e2wf+e3wf)
                tot_stdev = (np.var(tot))**0.5
                if np.abs(np.mean(tot[:200])) > 5*tot_stdev:
                    reason += " tail_start "
                    
                #decide if it is saturated
                max_v = 0.35
                #if max([e10,e20,e30])>max_v:
                if max([stupid_max_1,stupid_max_2,stupid_max_3])>max_v:
                    reason += "saturated "
                    saturated = np.append(saturated,True)
                else:
                    saturated = np.append(saturated,False)
                
                #decide if the software version is good
                if sw!="2.1.1":
                    reason += "wrong_sw_v "
                
                #is there any problem?
                if reason==" ":
                    looks_good = np.append(looks_good,1)
                else:
                    looks_good = np.append(looks_good,0)
                reasons.append(reason)
                
                folder = "/997_data/solo_features/plots/"
                
                isave = True
                
                plot_spectra(e1wf,
                              smooth_1_uncorrected,
                              corrected_1,
                              time_step,
                              len(amplitudes),i,len(epoch),
                              folder = folder,
                              save = isave,
                              name_prefix = txt[-16:-4])
                
                plot_waveforms( [[smooth_1,smooth_2,smooth_3],
                                  [e1wf,e2wf,e3wf], 
                                  [-WF2,WF2-WF1,WF1] , 
                                  [body_waveform,body_waveform,body_waveform],
                                  [antenna_waveform_1,antenna_waveform_2,antenna_waveform_3]],
                                  #[decay_model_1,decay_model_2,decay_model_3],
                                  #[corrected_1,corrected_2,corrected_3]],
                                epoch_offset[i],
                                antenna_max_1,antenna_max_2,antenna_max_3,
                                index_center,
                                body_index,
                                body_amp,
                                len(amplitudes),i,len(epoch),
                                reason = reason,
                                waveform_alphas = [1,
                                                    0.1,
                                                    1,
                                                    0.5,
                                                    0.5],
                                                    #1,
                                                    #1],
                                waveform_colors = ["black",
                                                    "black",
                                                    "limegreen",
                                                    "deepskyblue",
                                                    "red"],
                                                    #"black",
                                                    #"magenta"],
                                save=isave,
                                secondary_present = np.array([secondary_1,secondary_2,secondary_3]).astype(dtype=bool),
                                additional_vlines=[prespike_index],
                                additional_vlines_colors=[prespike*"green"+(1-prespike)*"none"],
                                additional_hlines = [0],
                                additional_hlines_colors = ["grey"],
                                folder = folder,
                                name_prefix = txt[-16:])
                
                plot_waveforms_minimal( [smooth_1,smooth_2,smooth_3],
                                        epoch_offset[i],
                                        body_index,
                                        body_index,
                                        len(amplitudes),i,len(epoch),
                                        save=isave,
                                        reason = reason,
                                        folder = folder,
                                        name_prefix = txt[-16:-4],
                                        ternary=True)

                name = 10000000*int(target_input_txt[-16:-8]) + 100000*int(target_input_txt[-6:-4]) + i
                ids = np.append(ids,name)
                epochs = np.append(epochs,epoch[i])
                event_id = np.append(event_id,len(amplitudes))
                radial_velocities.append(f_rad_v(tt2000_to_jd(epoch[i])))
                azimuthal_velocities.append(f_tan_v(tt2000_to_jd(epoch[i])))
                heliocentric_distances.append(f_heliocentric_distance(tt2000_to_jd(epoch[i])))
                if YYYYMMDD in scpot_dates:
                    scpots.append(float(f_scpot(tt2000_to_jd(epoch[i]))))
                else:
                    scpots.append(0.0)
                ion_amplitudes = np.vstack((ion_amplitudes,[antenna_waveform_1[antenna_max_1],
                                                            antenna_waveform_2[antenna_max_2],
                                                            antenna_waveform_3[antenna_max_3]]))
                electron_amplitudes = np.append(electron_amplitudes,body_amp)
                delays = np.vstack((delays,[delay_1,delay_2,delay_3]))
                secondary_present = np.vstack((secondary_present,[secondary_1,secondary_2,secondary_3]))
                bias_current = np.vstack((bias_current,bias(tt2000_to_jd(epoch[i]))))
                if len(reason)<2:
                    alright = np.append(alright,True)
                else:
                    alright = np.append(alright,False) 

    # save
    iids = ids[:]
    ievent_id = event_id[:]
    imaxWF = maxWF[:,:]
    iamplitudes = amplitudes[:,:]
    iion_amplitudes = ion_amplitudes[:,:]
    ielectron_amplitudes = electron_amplitudes[:]
    idelays = delays[:,:]/1000 #to get to microseconds  
    isecondary_present = secondary_present[:,:]
    ipolarities = polarities[:] 
    istupid_max = stupid_max[:,:]
    iant_decay_times = ant_decay_times[:] #s
    iant_decay_times_alt = ant_decay_times_alt[:] #s, legacy
    ibody_decay_times = body_decay_times[:] #s
    iheliocentric_distances = np.array(heliocentric_distances)[:]
    iazimuthal_velocities = np.array(azimuthal_velocities)[:]
    iradial_velocities = np.array(radial_velocities)[:]
    iepochs = epochs[:]
    ibody_risetimes = body_risetimes[:] #s
    ibias_current = bias_current[:,:]
    inegative_prespikes = negative_prespikes[:]
    iscpots = np.array(scpots)[:]
    ialright = alright
    isaturated = saturated
    iovershoot = overshoot[:,:]
    iovershoot_delay = overshoot_delay[:,:]
    ixld_risetimes = xld_risetimes[:,:]
    
    #save all the the features 
    np.savez(target_output_npz,
             ids = iids,
             event_id = ievent_id,
             maxWF = imaxWF,
             amplitudes = iamplitudes,
             ion_amplitudes = iion_amplitudes,
             electron_amplitudes = ielectron_amplitudes,
             delays = idelays,
             secondary_present = isecondary_present,
             polarities = ipolarities,
             stupid_max = istupid_max,
             ant_decay_times = iant_decay_times,
             ant_decay_times_alt = iant_decay_times_alt,
             body_decay_times = ibody_decay_times,
             heliocentric_distances = iheliocentric_distances,
             azimuthal_velocities = iazimuthal_velocities,
             radial_velocities = iradial_velocities,
             epochs = iepochs,
             body_risetimes = ibody_risetimes,
             bias_current = ibias_current,
             negative_prespikes = inegative_prespikes,
             scpots = iscpots,
             alright = ialright,
             saturated = isaturated,
             overshoot = iovershoot,
             overshoot_delay = iovershoot_delay,
             xld_risetimes = ixld_risetimes)  


def show_monopole(YYYYMMDD,
                  index,
                  minimal_plot=True,
                  isave=True,
                  folder="/997_data/solo_features/plots/waveforms_se/"):
    """
    The function to plot a monopole event not doing statistics nor 
    XLD1 correction.

    Parameters
    ----------
    YYYYMMDD : str
        date of the file.
    index : int
        index of the event.
    minimal_plot : bool, optional
        if to plot minimal only, no analysis. The default is True.
    isave : bool, optional
        to save or not to save. The default is True.
    folder : str, optional
        folder where to save if save. The default is "/997_data/solo_features/plots/waveforms_se/".

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    """

    amplitudes = np.zeros((0,3))       #global maxima
    polarities = np.zeros(0)
    stupid_max = np.zeros((0,3))   #stupid max - median
    ant_decay_times = np.zeros(0)
    maxWF = np.zeros((0,3))    #jakubs analysis
    ant_decay_times_alt = np.zeros(0)
    body_decay_times = np.zeros(0)
    negative_prespikes = np.zeros(0)
    body_risetimes = np.zeros(0)
    looks_good = np.zeros(0,dtype="bool")
    reasons = []
    saturated = np.zeros(0,dtype="bool")
    
    try:            
        cdf_file_e = fetch(YYYYMMDD,
                     '/997_data/solo_rpw/tds_wf_e',
                     "tds_wf_e",
                     "_rpw-tds-surv-tswf-e_",
                     ["V06.cdf","V05.cdf","V04.cdf","V03.cdf","V02.cdf","V01.cdf"],    
                     True)
        
        e = cdf_file_e.varget("WAVEFORM_DATA_VOLTAGE") #[event,channel,time], channel = 2 is monopole in XLD1
        epoch = cdf_file_e.varget("EPOCH")              #start of each impact
        epoch_offset = cdf_file_e.varget("EPOCH_OFFSET")     #time variable for each sample of each impact
        sw = cdf_file_e.attget("Software_version",entry=0)["Data"]    #should be 2.1.1 or higher
        
    except:
        print("no file on "+YYYYMMDD)
    else:
        i=index
        reason = " "

        #time step will be handy later
        time_step = epoch_offset[i][1]-epoch_offset[i][0]
        

        #original vaweforms, as Jakub calls it
        WF1 = e[i,0,:]
        WF2 = e[i,1,:]
        WF3 = e[i,2,:]
        
        #highest absolute value in original waveforms, sign respective
        maxWF1 = WF1[np.argmax(np.abs(WF1))]
        maxWF2 = WF2[np.argmax(np.abs(WF2))]
        maxWF3 = WF3[np.argmax(np.abs(WF3))]
        maxWF = np.vstack((maxWF,[maxWF1,maxWF2,maxWF3]))
        
        #in case of true monopole SE1 data
        e1wf = e[i,0,:]
        e2wf = e[i,1,:]
        e3wf = e[i,2,:]
        
        #backgrounds - shouldn't use median because we do a cumsum later
        bg1 = np.mean(e1wf)  
        bg2 = np.mean(e2wf)
        bg3 = np.mean(e3wf)
        
        #smoothed monopole wavefroms without backgrounds
        sos = butter(32,7e4,btype="lowpass",fs=(1/(time_step/1e9)),output="sos")
        smooth_1 = sosfilt(sos, e1wf)-bg1
        smooth_2 = sosfilt(sos, e2wf)-bg2
        smooth_3 = sosfilt(sos, e3wf)-bg3

        #corrected for the highpass artefact - is okay to do to sums, as each individual channel needs the same correction
        corrected_1 = hipass_correction(smooth_1,time_step/1e9)
        corrected_2 = hipass_correction(smooth_2,time_step/1e9)
        corrected_3 = hipass_correction(smooth_3,time_step/1e9)
        
        calibrate = True
        if calibrate:
            smooth_1 = corrected_1-np.mean(corrected_1)
            smooth_2 = corrected_2-np.mean(corrected_2)
            smooth_3 = corrected_3-np.mean(corrected_3)

        #time index of the global maximum
        time_index_1 = np.argmax(np.abs(smooth_1))
        time_index_2 = np.argmax(np.abs(smooth_2))
        time_index_3 = np.argmax(np.abs(smooth_3))
        
        #stupid ion
        stupid_max_1 = np.max(smooth_1)
        stupid_max_2 = np.max(smooth_2)
        stupid_max_3 = np.max(smooth_3)
        stupid_max = np.vstack((stupid_max,[stupid_max_1,stupid_max_2,stupid_max_3]))
        
        #provisional amplitudes
        e10 = smooth_1[time_index_1]
        e20 = smooth_2[time_index_2]
        e30 = smooth_3[time_index_3]
        
        #where to center the window
        index_center = [time_index_1,time_index_2,time_index_3][np.argmax([abs(e10),abs(e20),abs(e30)])]

        deriv2_1 = savgol_filter(np.maximum(smooth_1,0),17,2,deriv=2)*100
        deriv2_2 = savgol_filter(np.maximum(smooth_2,0),17,2,deriv=2)*100
        deriv2_3 = savgol_filter(np.maximum(smooth_3,0),17,2,deriv=2)*100
        
        #look for the minimum, but only in a window around index_center
        min_deriv2_1 = np.argmin(deriv2_1[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
        min_deriv2_2 = np.argmin(deriv2_2[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
        min_deriv2_3 = np.argmin(deriv2_3[np.max([index_center-200,0]):np.min([index_center+10,len(epoch_offset[0])])])+np.max([index_center-200,0])
        
        #the chronologically first
        body_index = min([min_deriv2_1,min_deriv2_2,min_deriv2_3]) #provisional
        
        #the largest of them
        #body_index = [min_deriv2_1,min_deriv2_2,min_deriv2_3][np.argmax([deriv2_1[min_deriv2_1],deriv2_2[min_deriv2_2],deriv2_3[min_deriv2_3]])] #provisional
        
        #time index of the electron prespike
        prespike = False
        prespike_index = np.argmin((smooth_1+smooth_2+smooth_3)[max(body_index-50,0):body_index+1])+max(body_index-50,0)
        prespike_value = ((smooth_1+smooth_2+smooth_3)/3)[prespike_index]
        prespike_mu = np.mean(((smooth_1+smooth_2+smooth_3)/3)[max(body_index-200,0):max(body_index-10,1)])
        prespike_sigma = (np.var(((smooth_1+smooth_2+smooth_3)/3)[max(body_index-200,0):max(body_index-10,1)]))**0.5
        if prespike_value < prespike_mu - 3*prespike_sigma:
            prespike = True
            negative_prespikes = np.append(negative_prespikes,prespike_value)
        else:
            negative_prespikes = np.append(negative_prespikes,np.nan)
            
        #correct for more local background
        lower_limit = np.max([0,prespike_index-100])
        smooth_1 = smooth_1-np.mean(smooth_1[lower_limit:prespike_index+1])
        smooth_2 = smooth_2-np.mean(smooth_2[lower_limit:prespike_index+1])
        smooth_3 = smooth_3-np.mean(smooth_3[lower_limit:prespike_index+1])
        
        #precise body index identification using min in 1st deriv around the provisional body index
        lower_limit = np.max([1,body_index-5])
        upper_limit = np.min([len(smooth_1)-2,body_index+20])
        #body_index_1 = lower_limit+np.argmin(np.abs(smooth_1[lower_limit+1:upper_limit]-smooth_1[lower_limit-1:upper_limit-2]))
        #body_index_2 = lower_limit+np.argmin(np.abs(smooth_2[lower_limit+1:upper_limit]-smooth_2[lower_limit-1:upper_limit-2]))
        #body_index_3 = lower_limit+np.argmin(np.abs(smooth_3[lower_limit+1:upper_limit]-smooth_3[lower_limit-1:upper_limit-2]))
        
        deriv1_1 = savgol_filter(smooth_1,11,2,deriv=1)
        deriv1_2 = savgol_filter(smooth_2,11,2,deriv=1)
        deriv1_3 = savgol_filter(smooth_3,11,2,deriv=1)
        
        body_index_1 = lower_limit+np.argmin(np.abs(deriv1_1[lower_limit:upper_limit]))-1
        body_index_2 = lower_limit+np.argmin(np.abs(deriv1_2[lower_limit:upper_limit]))-1
        body_index_3 = lower_limit+np.argmin(np.abs(deriv1_3[lower_limit:upper_limit]))-1
        
        #max correlation with an arc or something
        if body_index>10 and body_index<len(smooth_1)-30:
            body_index_1 = lower_limit + highest_corr(smooth_1[lower_limit-5:lower_limit+31])
            body_index_2 = lower_limit + highest_corr(smooth_2[lower_limit-5:lower_limit+31])
            body_index_3 = lower_limit + highest_corr(smooth_3[lower_limit-5:lower_limit+31])
        
        body_amp_1 = smooth_1[body_index]
        body_amp_2 = smooth_2[body_index]
        body_amp_3 = smooth_3[body_index]
        strongest_body = np.argmax([body_amp_1,body_amp_2,body_amp_3])
        
        if strongest_body == 0:
            body_index = int(np.round(np.mean([body_index_2,body_index_3,body_index])))
        elif strongest_body == 1:
            body_index = int(np.round(np.mean([body_index_1,body_index_3,body_index])))
        else:
            body_index = int(np.round(np.mean([body_index_1,body_index_2,body_index])))  
        
        
        #find the parameters for the body peak
        #amplitude
        body_amp_1 = smooth_1[body_index]
        body_amp_2 = smooth_2[body_index]
        body_amp_3 = smooth_3[body_index]
        body_amp = ( body_amp_1+body_amp_2+body_amp_3 - np.max([body_amp_1,body_amp_2,body_amp_3]) )/2 #weaker 2 of the 3
        strongest_body = np.argmax([body_amp_1,body_amp_2,body_amp_3])
        #FWHM
        try:
            body_hwhm_1 = 20-np.argmin(np.abs(smooth_1[body_index-20:body_index+1]-smooth_1[body_index]/2))
            body_hwhm_2 = 20-np.argmin(np.abs(smooth_2[body_index-20:body_index+1]-smooth_2[body_index]/2))
            body_hwhm_3 = 20-np.argmin(np.abs(smooth_3[body_index-20:body_index+1]-smooth_3[body_index]/2))
            body_hwhms = [body_hwhm_1,body_hwhm_2,body_hwhm_3]
            body_hwhms.pop(strongest_body)
            body_fwhm = np.sum(body_hwhms) #the same as 2*average, since we have 2 elements only
            #define body approx wavefrom
            body_waveform = body_amp*gaussian_body(np.arange(len(smooth_1)),mu=body_index,sig=body_fwhm/2.355)
        except:
            body_waveform = np.zeros(len(smooth_1))
            reason += "no_body_peak "
        
        #get ion waveforms as what is left when we get rid of the body waveform
        antenna_waveform_1 = smooth_1 - body_waveform
        antenna_waveform_2 = smooth_2 - body_waveform
        antenna_waveform_3 = smooth_3 - body_waveform
        
        #time index of the global maximum considering the body peak
        upper_limit = np.min([body_index+300,len(antenna_waveform_1)])
        antenna_max_1 = body_index+np.argmax(antenna_waveform_1[body_index:upper_limit])
        antenna_max_2 = body_index+np.argmax(antenna_waveform_2[body_index:upper_limit])
        antenna_max_3 = body_index+np.argmax(antenna_waveform_3[body_index:upper_limit])
        
        #are there actually secondary peaks?
        secondary_1 = (np.max(antenna_waveform_1[body_index:upper_limit])) / (body_amp) > 0.75
        secondary_2 = (np.max(antenna_waveform_2[body_index:upper_limit])) / (body_amp) > 0.75
        secondary_3 = (np.max(antenna_waveform_3[body_index:upper_limit])) / (body_amp) > 0.75
        
        #make them contain the actual amplitudes or 0 if not present
        secondary_1 *= np.max(antenna_waveform_1[body_index:upper_limit])
        secondary_2 *= np.max(antenna_waveform_2[body_index:upper_limit])
        secondary_3 *= np.max(antenna_waveform_3[body_index:upper_limit])
        
        #ampltiudes
        amplitudes = np.vstack((amplitudes,[smooth_1[antenna_max_1],smooth_2[antenna_max_2],smooth_3[antenna_max_3]]))
        
        #get the time scale of the ion peak decay to 1/e
        strongest_antenna = np.argmax([smooth_1[antenna_max_1],smooth_2[antenna_max_2],smooth_3[antenna_max_3]])
        strongest_antenna_time_index = np.array([antenna_max_1,antenna_max_2,antenna_max_3])[strongest_antenna]
        strongest_waveform = np.vstack((smooth_1,smooth_2,smooth_3))[strongest_antenna,:]
        threshol_index = np.argmin(np.abs(strongest_waveform[strongest_antenna_time_index:strongest_antenna_time_index+250]))+strongest_antenna_time_index
        decay_time = (threshol_index - strongest_antenna_time_index)*3.81e-6 #s
        ant_decay_times = np.append(ant_decay_times,decay_time)
        ant_decay_times_alt = np.append(ant_decay_times_alt,decay_time) #legacy reasons
        
        #get the time scale of electron peak decay to 1/e
        weakest_body = np.argmin(np.array([smooth_1[body_index],
                                  smooth_2[body_index],
                                  smooth_3[body_index]]))
        weakest_waveform = np.vstack((smooth_1,smooth_2,smooth_3))[weakest_body,:]
        threshold_e = (weakest_waveform[body_index])*(1/np.e)
        threshol_index_e = np.argmin(np.abs(weakest_waveform[body_index:body_index+50]-threshold_e))+body_index
        decay_time_e = (threshol_index_e - body_index)*3.81e-6 #s
        if np.min(weakest_waveform[body_index:body_index+50])>threshold_e: #if not crossed
            decay_time_e = 50*3.81e-6 #s
        body_decay_times = np.append(body_decay_times,decay_time_e)
        
        #electron peak risetime (time to get from 43% to 80%)
        mark_43 = body_amp*0.43
        mark_80 = body_amp*0.80
        strongest_electron_peak = np.argmax(np.array([smooth_1[body_index],smooth_2[body_index],smooth_3[body_index]]))                                     
        if strongest_electron_peak == 0:
            smooth = smooth_2 + smooth_3
        elif strongest_electron_peak == 1:
            smooth = smooth_1 + smooth_3
        elif strongest_electron_peak == 2:
            smooth = smooth_2 + smooth_1
        else:
            raise Exception()
        #define the window where to look for the timescale
        if body_index > 0:
            index_43 = np.argmin(np.abs(smooth[max(body_index-100,0):body_index]-mark_43))
            index_80 = np.argmin(np.abs(smooth[max(body_index-100,0):body_index]-mark_80))
            body_risetime = (index_80 - index_43)*3.81e-6 #s
        else:
            body_risetime = 0 #s
        body_risetimes = np.append(body_risetimes,body_risetime)
        
        #correct for electron prespike
        if prespike:
            body_amp -= prespike_value
            
        #positive if the most important waveform is positive and overall is positive
        upper_limit = np.min([len(smooth_1)-1,body_index+100])
        if np.array([e10,e20,e30])[strongest_antenna]>0 and np.mean((smooth_1+smooth_2+smooth_3)[body_index:upper_limit])>0:
            if body_amp>0:
                polarities = np.append(polarities,1)
            else:
                polarities = np.append(polarities,-1)
                reason += "neg "
        else:
            polarities = np.append(polarities,-1)
            reason += "neg "
            
        #decide if we cought the waveform
        if [body_index_1,body_index_2,body_index_3][strongest_body] < 100 or [time_index_1,time_index_2,time_index_3][strongest_body] > len(epoch_offset[0])-100:
            reason += "missed_window "
            
        #decide if the original signal starts in the tail
        tot = e1wf+e2wf+e3wf - np.mean(e1wf+e2wf+e3wf)
        tot_stdev = (np.var(tot))**0.5
        if np.abs(np.mean(tot[:200])) > 5*tot_stdev:
            reason += " tail_start "
            
        #decide if it is saturated
        max_v = 0.35
        if max([e10,e20,e30])>max_v:
            reason += "saturated "
            saturated = np.append(saturated,True)
        else:
            saturated = np.append(saturated,False)
        
        #decide if the software version is good
        if sw!="2.1.1":
            reason += "wrong_sw_v "
        
        #is there any problem?
        if reason==" ":
            looks_good = np.append(looks_good,1)
        else:
            looks_good = np.append(looks_good,0)
        reasons.append(reason)

        if not minimal_plot:
            plot_waveforms( [[smooth_1,smooth_2,smooth_3],
                              [e1wf,e2wf,e3wf], 
                              #[deriv2_1,deriv2_2,deriv2_3] , 
                              [body_waveform,body_waveform,body_waveform],
                              [antenna_waveform_1,antenna_waveform_2,antenna_waveform_3]],
                              #[decay_model_1,decay_model_2,decay_model_3],
                              #[corrected_1,corrected_2,corrected_3]],
                            epoch_offset[i],
                            antenna_max_1,antenna_max_2,antenna_max_3,
                            index_center,
                            body_index,
                            body_amp,
                            len(amplitudes),i,len(epoch),
                            reason = reason,
                            waveform_alphas = [1,
                                                0.1,
                                                #1,
                                                0.5,
                                                0.5],
                                                #1,
                                                #1],
                            waveform_colors = ["black",
                                                "black",
                                                #"orange",
                                                "deepskyblue",
                                                "red"],
                                                #"black",
                                                #"magenta"],
                            save=isave,
                            secondary_present = np.array([secondary_1,secondary_2,secondary_3]).astype(dtype=bool),
                            additional_vlines=[prespike_index],
                            additional_vlines_colors=[prespike*"green"+(1-prespike)*"none"],
                            additional_hlines = [0],
                            additional_hlines_colors = ["grey"],
                            folder = folder,
                            name_prefix = cdf_file_e.attget('ACCESS_URL',entry=0)["Data"][-16:-4])
        else:    
            plot_waveforms_minimal( [smooth_1,smooth_2,smooth_3],
                                    epoch_offset[i],
                                    body_index,
                                    body_index,
                                    len(amplitudes),i,len(epoch),
                                    save=isave,
                                    ternary = True,
                                    folder = folder,
                                    name_prefix = cdf_file_e.attget('ACCESS_URL',entry=0)["Data"][-16:-4])
    
            
    
#execute
main(sys.argv[1],sys.argv[2])

#instructive
#plot_extraction_process("20200712",258)
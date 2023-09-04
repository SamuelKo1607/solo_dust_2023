import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600
import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
import glob
mpl.rcParams['text.usetex'] = False
from ag2023_vuv_function import illumination
from commons_conversions import tt2000_to_jd

#this is the frame in which we will be working
arrays_to_define =  ["ids",
                     "alright",
                     "saturated",
                     "polarities",
                     "epochs",
                     "heliocentric_distances",
                     "scpots",
                     "body_amplitudes",
                     "body_rise_times",
                     "body_decay_times",
                     "negative_prespikes",
                     "stupid_max_v1",
                     "stupid_max_v2",
                     "stupid_max_v3",
                     "maxWF1",
                     "maxWF2",
                     "maxWF3",
                     "delay_v1",
                     "delay_v2",
                     "delay_v3",
                     "sec_present_v1",
                     "sec_present_v2",
                     "sec_present_v3",
                     "overshoot_1",
                     "overshoot_2",
                     "overshoot_3",
                     "overshoot_delay_1",
                     "overshoot_delay_2",
                     "overshoot_delay_3",
                     "xld_risetime_1",
                     "xld_risetime_2",
                     "xld_risetime_3",
                     "vuv_illumination"]


def npz_files_to_analyze(path):
    """
    Make a list of npz files that will be loaded.
    
    Parameters
    ----------
    path : str
        Directory in which to look for the npz files.

    Returns
    -------
    npzs : numpy.ndarray(1,:) of str
        An array of npz files, absolute paths.

    """
    npzs = glob.glob(path+"*.npz")
    return npzs


def create_df(names):
    """
    Creates an empy dataframe with the predefined structure.
    
    Parameters
    ----------
    names : list of string
        These are the coulmn names of the dataframe to create.

    Returns
    -------
    df : pd.dataframe
        Empty structure for the data.
    """
    arrays = {}
    for name in names:
        arrays[name] = np.zeros(0)
    df = pd.DataFrame(arrays)
    return df

    
def load_npz_into_df(target_npz):
    """
    Creates a single dataframe out of npz file
    
    Parameters
    ----------
    target_npz : str
        An absolute path to the npz that is to be unpacked and 
        made into dataframe.

    Returns
    -------
    df : pd.dataframe
        Dataframe made of contents of target_npz.npz file.
        
    """
    npz_content = np.load(target_npz)
    
    #the dictionary to translate from npz column names to local keys
    ids = npz_content["ids"]
    alright = npz_content["alright"]
    saturated = npz_content["saturated"]
    polarities = npz_content["polarities"]
    epochs = npz_content["epochs"]
    heliocentric_distances = npz_content["heliocentric_distances"]
    scpots = npz_content["scpots"]
    body_amplitudes = npz_content["electron_amplitudes"]
    body_rise_times = npz_content["body_risetimes"]
    body_decay_times = npz_content["body_decay_times"]
    negative_prespikes = npz_content["negative_prespikes"]
    stupid_max_v1 = npz_content["stupid_max"][:,0]
    stupid_max_v2 = npz_content["stupid_max"][:,1]
    stupid_max_v3 = npz_content["stupid_max"][:,2]
    maxWF1 = npz_content["maxWF"][:,0]
    maxWF2 = npz_content["maxWF"][:,1]
    maxWF3 = npz_content["maxWF"][:,2]
    delay_v1 = npz_content["delays"][:,0]
    delay_v2 = npz_content["delays"][:,1]
    delay_v3 = npz_content["delays"][:,2]
    sec_present_v1 = npz_content["secondary_present"][:,0]
    sec_present_v2 = npz_content["secondary_present"][:,1]
    sec_present_v3 = npz_content["secondary_present"][:,2]
    overshoot_1 = npz_content["overshoot"][:,0]
    overshoot_2 = npz_content["overshoot"][:,1]
    overshoot_3 = npz_content["overshoot"][:,2]
    overshoot_delay_1 = npz_content["overshoot_delay"][:,0]
    overshoot_delay_2 = npz_content["overshoot_delay"][:,1]
    overshoot_delay_3 = npz_content["overshoot_delay"][:,2]
    xld_risetime_1 = npz_content["xld_risetimes"][:,0]
    xld_risetime_2 = npz_content["xld_risetimes"][:,1]
    xld_risetime_3 = npz_content["xld_risetimes"][:,2]
    if len(epochs)>0:
        vuv_illumination = illumination(tt2000_to_jd(epochs))
    else:
        vuv_illumination = np.zeros(0)
    
    #shape them into one pd.df
    df = pd.DataFrame(data=np.vstack((ids,
                                      alright,
                                      saturated,
                                      polarities,
                                      epochs,
                                      heliocentric_distances,
                                      scpots,
                                      body_amplitudes,
                                      body_rise_times,
                                      body_decay_times,
                                      negative_prespikes,
                                      stupid_max_v1,
                                      stupid_max_v2,
                                      stupid_max_v3,
                                      maxWF1,
                                      maxWF2,
                                      maxWF3,
                                      delay_v1,
                                      delay_v2,
                                      delay_v3,
                                      sec_present_v1,
                                      sec_present_v2,
                                      sec_present_v3,
                                      overshoot_1,
                                      overshoot_2,
                                      overshoot_3,
                                      overshoot_delay_1,
                                      overshoot_delay_2,
                                      overshoot_delay_3,
                                      xld_risetime_1,
                                      xld_risetime_2,
                                      xld_risetime_3,
                                      vuv_illumination
                                      )).transpose(),
                      columns=arrays_to_define)
    return df


def add_gridlines_ternary(axis,
                          density=0.25,
                          col="gray",
                          lw=0.75,
                          alpha=0.5):
    """
    Draw some guiding lines on top of a ternary plot.
    
    Parameters
    ----------
    axis : TYPE
        The axis where to draw those lines.
    density : float, optional
        Distance between consecutive lines. The default is 0.25.
    col : str, optional
        Color of the lines. The default is "gray".
    alpha : float, optional
        Opacity of the lines. The default is 0.5.

    Returns
    -------
    gridlines : TYPE
        The list of used lines values.

    """
    
    n_sections = np.round(1./density)
    step = 1./n_sections
    gridlines = []
    if n_sections > 1:
        gridlines = np.arange(step,0.999,step)
    
        for line in gridlines:
            axis.plot([line,(line+1)/2.],
                      [0,(1-line)*np.sqrt(3)/2],
                      color=col,zorder=2,alpha=alpha,lw=lw)
            axis.plot([line,line/2.],
                      [0,(line)*np.sqrt(3)/2],
                      color=col,zorder=2,alpha=alpha,lw=lw)
            axis.plot([line/2.,(1-line/2.)],
                      [(line)*np.sqrt(3)/2,(line)*np.sqrt(3)/2],
                      color=col,zorder=2,alpha=alpha,lw=lw)
            
            axis.plot([0,0.75],[0,np.sqrt(0.75)/2],
                      c="darkgray",ls="dashed",alpha=0.5,zorder=2,lw=lw)
            axis.plot([1,0.25],[0,np.sqrt(0.75)/2],
                    c="darkgray",ls="dashed",alpha=0.5,zorder=2,lw=lw)
            axis.plot([0.5,0.5],[0,np.sqrt(0.75)],
                    c="darkgray",ls="dashed",alpha=0.5,zorder=2,lw=lw)
            
            axis.text(line/2.-0.15,(line)*np.sqrt(3)/2-0.02,
                      str(int(100*line))+"%",
                      fontsize="x-small",color=col)
            axis.text((line+1)/2.,(1-line)*np.sqrt(3)/2+0.02,
                      str(int(100*line))+"%",
                      fontsize="x-small",color=col,rotation=60)
            axis.text(line-0.02,-0.13,
                      str(int(100*(1-line)))+"%",
                      fontsize="x-small",color=col,rotation=-60)
            axis.text(1.05,0.45,"v3 %",rotation=0,
                      va="center",ha="center",color="gray",alpha=alpha)
            axis.text(-0.05,0.45,"v1 %",rotation=0,
                      va="center",ha="center",color="gray",alpha=alpha)
            axis.text(0.5,-0.2,"v2 %",rotation=0,
                      va="center",ha="center",color="gray",alpha=alpha)
            
    return gridlines


def make_plot_hist_prespike_potential(data,
                                      target_dir,
                                      name="hist_prespike_potential"):
    """
    A plotting routine for the analysis of prespike presence 
    as a function of the spacecraft potential.
    
    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "hist_prespike_potential.png".

    Returns
    -------
    None.

    """
    
    fig,ax = plt.subplots()
    ax.hist(data["scpots"][np.isnan(data["negative_prespikes"])],
            bins=np.arange(0,20,1),
            label="Without prespikes", 
            density=True, alpha = 1, color=u"#D01157", histtype="step",
            ls=(0,(1,0.7)),lw=0.7)
    ax.hist(data["scpots"][~np.isnan(data["negative_prespikes"])],
            bins=np.arange(0,20,1),
            label="With prespikes", 
            density=True, alpha = 1, color=u"#5DA7FC", histtype="step",
            ls=(0,(6,2)),lw=0.7)
    mean_scpots = np.mean(data["scpots"][(data["scpots"]>0)*(
                                data["scpots"]<20)*(
                                np.isnan(data["negative_prespikes"]))])
    mean_scpots_prespikes = np.mean(data["scpots"][(data["scpots"]>0)*(
                                data["scpots"]<20)*(
                                ~np.isnan(data["negative_prespikes"]))])
    ax.vlines([mean_scpots,mean_scpots_prespikes],0,0.2,
              color=[u"#D01157",u"#5DA7FC"],ls=[(0,(1,0.7)),(0,(6,2))])
    ax.set_xlabel("Spacecraft potential [V]")
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')
    
    
def make_plot_hist_heliocentric_nano(data,
                                     target_dir,
                                     name="hist_heliocentric_nano"):
    """
    A plotting routine for the analysis of frequency of nano-impacts. Chosen 
    as great amplification, small delay and small amplitude.
    
    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "hist_heliocentric_nano.png".

    Returns
    -------
    None.

    """
    
    #prepare the time delay of the antenna-wise maximum
    total_max_delays = np.zeros(len(data.index))
    for i in range(len(data.index)):
        df = data.iloc[i]
        total_max_delays[i] = [df["delay_v1"],
                               df["delay_v2"],
                               df["delay_v3"]][
                                   np.argmax([df["sec_present_v1"],
                                              df["sec_present_v2"],
                                              df["sec_present_v3"]])]
    
    total_max = np.max(np.vstack((data["stupid_max_v1"],
                                  data["stupid_max_v2"],
                                  data["stupid_max_v3"])).transpose(),
                       axis=1)
    
    maxima_each_other = total_max / data["body_amplitudes"]
    
    mask = (maxima_each_other>10) * (total_max<0.05) * (total_max_delays<20)
    
    fig,ax = plt.subplots()
    ax.hist(data["heliocentric_distances"][mask],
            bins=np.arange(0.4,1.1,0.1),
            label="nano candidates", 
            density=True, alpha = 1, color=u"#F3752B", histtype="step")
    ax.hist(data["heliocentric_distances"][~mask],
            bins=np.arange(0.4,1.1,0.1),
            label="the rest", 
            density=True, alpha = 1, color=u"#522B47", histtype="step")
    mean_loampl = np.mean(data["heliocentric_distances"][mask])
    mean_hiampl = np.mean(data["heliocentric_distances"][~mask])
    ax.text(0.4,2.0,str(sum(mask)),color=u"#F3752B")
    ax.text(0.4,1.5,str(sum(~mask)),color=u"#522B47")
    ax.vlines([mean_loampl,mean_hiampl],0,3,color=[u"#F3752B",u"#522B47"])
    ax.set_xlabel("Heliocentric distance [AU]")
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)
    

def make_plot_assignment_jakub(data,
                    target_dir,
                    hand_colors=False,
                    name="assignment_jakub"):
    """
    A function to replicate Jakub's analysis which antenna produced
    the most of the signal, based on dipoles' polarity.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    hand_colors : bool, optional
        If we only want the colors (to be used by a different routine)
    name : str, optional. Default is "assignment_jakub.png".
        Output png name.


    Returns
    -------
    colorcode : array of str
        Colors for a different plotting procedure.

    """
    maxWF = np.vstack((data["maxWF1"],
                      data["maxWF2"],
                      data["maxWF3"])).transpose()
    
    colorcode = []
    for i in range(len(data.index)):
        if -maxWF[i,0]<0 and -maxWF[i,1]>0:
            colorcode.append("blue")
        elif -maxWF[i,0]<maxWF[i,1]:
            colorcode.append("black")
        else:
            colorcode.append("red")
            
    if hand_colors:
        return(np.array(colorcode))
    else:     
        fig,ax = plt.subplots()
        ax.scatter(-maxWF[:,0],-maxWF[:,1],c=colorcode,marker="x",s=2,lw=0.2)
        ax.set_xlabel("-WF1 (13) [V]")
        ax.set_ylabel("-WF2 (21) [V]")
        ax.text(-0.2,0.4,"ant_1",fontsize="small")
        ax.text(-0.1,-0.2,"ant_2",fontsize="small")
        ax.text(0.2,0.4,"ant_3",fontsize="small")
        ax.text(-0.1,0.6,"datapoints: %s" % str(len(maxWF[:,0])),fontsize="small")
        
        plt.savefig(target_dir+name+".png", format='png', dpi=300)
    

def make_plot_ternary(data,
                      target_dir,
                      colorcode=[],
                      colorscale=" ",
                      vmin=-3.5, vmax=-0.5,
                      showhist = True,
                      note = "",
                      name="ternary"):
    """
    Make a ternary plot of all the datapoints, taking into account the 
    stupid max amplitude for all the provided imapcts.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    colorcode : array of color, optional
        Possibly, scatter points may be colored according to a rule.
    colorscale : string, optional
        Possibly, scatter points may be colored according to a rule. Then 
        colorscale is useful as colorbar has to be generated. 
    vmin : float, optional
        Minimum of colorscale. Default is -3.5.
    vmax : float, optional
        Maximum of colorscale. Default is -0.5.
    showhist : bool, optional
        Whether to show the x-axis histogram under the triangle.
    name : str, optional
        Output png name. Default is "ternary.png".

    Returns
    -------
    None.

    """
    amps = np.vstack((data["stupid_max_v1"],
                      data["stupid_max_v2"],
                      data["stupid_max_v3"])).transpose()
    
    amps_norm = np.transpose(1/np.sum(amps,axis=1)*amps.transpose())
    
    transform = np.array([[1., 0.5],
                          [0., np.sqrt(0.75)]])
    
    xy = np.matmul(amps_norm[:,[2,0]],transform.transpose())
    
    fig,ax=plt.subplots()
    canvas = plt.Rectangle(( 0 , 0 ), 1, np.sqrt(0.75), color="white")
    triaxes = plt.Polygon(np.array([(0,0),(1,0),(0.5,np.sqrt(0.75))]), facecolor="white", edgecolor="black", lw=0.5)
    ax.axis('off')
    ax.set_aspect(1)
    ax.set_xlim(-0.2,1.2)
    ax.set_ylim(-0.5,np.sqrt(0.75)+0.2)
    ax.add_artist(canvas)
    gridlines = add_gridlines_ternary(ax,lw=0.5)
    ax.add_artist(triaxes)
    ax.text(-0.2,-0.1,"V2")
    ax.text(1.1,-0.1,"V3")
    ax.text(0.45,np.sqrt(0.75)+0.1,"V1")
    ax.text(0.7,1,"datapoints: %s" % str(len(data.index)),fontsize="x-small")
    if len(note)>0:
        ax.text(0,1,note,fontsize="x-small")
        

    alphavalue = max(min(200 / len(data.index), 1), 0.01)
    
    if len(colorcode) == 0:
        scat = ax.scatter(xy[:,0],xy[:,1],
                          s=0.5,alpha=alphavalue,c="blue",zorder=5)
    elif isinstance(colorcode, str):
        scat = ax.scatter(xy[:,0],xy[:,1],
                          s=0.5,alpha=alphavalue,c=colorcode,zorder=5)
    else:
        #cmap = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-3.5, vmax=-0.5), cmap="plasma")
        scat = ax.scatter(xy[:,0],xy[:,1],
                          s=0.1,alpha=0.5,c=colorcode,cmap="viridis", 
                          vmin=vmin, vmax=vmax)
    
    if len(colorscale)<2:
        pass
    else:
        cbar = fig.colorbar(scat)
        cbar.set_label(colorscale, rotation=90)

    if showhist:
        bins = np.arange(0,1.1,0.05)
        #bincenters = (bins[1:]+bins[:-1])/2
        hst = np.histogram(xy[:,0],bins=bins,density=True)[0]
        offset = 0.49
        for line in np.append([0,1],gridlines):
            ax.vlines(line,-0.02-offset,0.02-offset,color="black",lw=0.5)
        ax.plot([0,1],[-offset,-offset],color="black",lw=0.5)
        if isinstance(colorcode, str):
            ax.step(bins[:-1],hst/np.max(hst)/5-offset,
                    where="post",color=colorcode,alpha=0.4)
            ax.vlines(np.mean(xy[:,0][(xy[:,0]>0)*(xy[:,0]<1)]),
                      0.2-offset,0-offset,
                      color=colorcode)
        else:
            ax.step(bins[:-1],hst/np.max(hst)/5-offset,
                    where="post",color="blue",alpha=0.4)
            ax.vlines(np.mean(xy[:,0][(xy[:,0]>0)*(xy[:,0]<1)]),
                      0.2-offset,0-offset,
                      color="blue")
        ax.text(0-0.1,-offset-0.05,"V2",fontsize="x-small")
        ax.text(1+0.03,-offset-0.05,"V3",fontsize="x-small")
        ax.text(0.5,-offset-0.1,"Horizontal position",
                ha="center",fontsize="small")
    
    plt.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_ternary_heatmap(data,
                      target_dir,
                      nbins = 50,
                      colorcode=[],
                      colorscale=" ",
                      vmin=-3.5, vmax=-0.5,
                      showhist = True,
                      note = "",
                      datapointcount = True,
                      colorbarticks = [0,5,10,15,20,25,30,35],
                      name="ternary_heatmap"):
    """
    Make a ternary plot (heatmap) of all the datapoints, 
    taking into account the  stupid max amplitude for 
    all the provided imapcts.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    nbins : int
        The number of linear bins in the heatmap. Default is 50.
    colorcode : array of color, optional
        Possibly, scatter points may be colored according to a rule.
    colorscale : string, optional
        Possibly, scatter points may be colored according to a rule. Then 
        colorscale is useful as colorbar has to be generated. 
    vmin : float, optional
        Minimum of colorscale. Default is -3.5.
    vmax : float, optional
        Maximum of colorscale. Default is -0.5.
    showhist : bool, optional
        Whether to show the x-axis histogram under the triangle.
    datapointcount : bool, optional
        Whetether to show the data point count in upper left corner. 
        Default is True.
    colorbarticks : list of float
        The list of ticks on the colorbar. Optional, default is 
        good for 50 bins. 
    name : str, optional
        Output png name. Default is "ternary.png".

    Returns
    -------
    None.

    """
    amps = np.vstack((data["stupid_max_v1"],
                      data["stupid_max_v2"],
                      data["stupid_max_v3"])).transpose()
    
    amps_norm = np.transpose(1/np.sum(amps,axis=1)*amps.transpose())
    
    transform = np.array([[1., 0.5],
                          [0., np.sqrt(0.75)]])
    
    xy = np.matmul(amps_norm[:,[2,0]],transform.transpose())
    
    fig,ax=plt.subplots()
    canvas = plt.Rectangle(( 0 , 0 ), 1, np.sqrt(0.75), color="white")
    triaxes = plt.Polygon(np.array([(0,0),(1,0),(0.5,np.sqrt(0.75))]), facecolor="none", edgecolor="black", lw=1,zorder=1.8)
    padding1 = plt.Polygon(np.array([(-0.5,-np.sqrt(0.75)),(-1,1),(1,2*np.sqrt(0.75))]), facecolor="white", edgecolor="black", lw=0,zorder=2)
    padding2 = plt.Polygon(np.array([(1.5,-np.sqrt(0.75)),(2,1),(0,2*np.sqrt(0.75))]), facecolor="white", edgecolor="black", lw=0,zorder=2)
    padding3 = plt.Polygon(np.array([(-5,0),(-5,-0.2),(6,-0.2),(6,0)]), facecolor="white", edgecolor="black", lw=0,zorder=2)
    ax.axis('off')
    ax.set_aspect(1)
    ax.set_xlim(-0.2,1.2)
    ax.set_ylim(-0.5,np.sqrt(0.75)+0.2)
    ax.add_artist(canvas)
    gridlines = add_gridlines_ternary(ax,lw=0.5)
    ax.add_artist(triaxes)
    ax.add_artist(padding1)
    ax.add_artist(padding2)
    ax.add_artist(padding3)
    ax.text(-0.2,-0.1,"V2",zorder=3)
    ax.text(1.1,-0.1,"V3",zorder=3)
    ax.text(0.45,np.sqrt(0.75)+0.1,"V1",zorder=3)
    if datapointcount:
        ax.text(0.7,1,"datapoints: %s" % str(len(data.index)),fontsize="x-small",zorder=3)
    if len(note)>0:
        ax.text(0,1,note,fontsize="x-small",zorder=3)
        
    alphavalue = max(min(200 / len(data.index), 1), 0.01)
    
    heatmap, xedges, yedges = np.histogram2d(xy[:,0],xy[:,1], bins=nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    
    
    
    
    
    if len(colorcode) == 0:
        heat = plt.imshow(heatmap.T, extent=extent, origin='lower',
                          cmap="BuPu",zorder=1.75)
    elif isinstance(colorcode, str):
        heat = ax.scatter(xy[:,0],xy[:,1],
                          s=0.5,alpha=alphavalue,c=colorcode,zorder=5)
    else:
        #cmap = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-3.5, vmax=-0.5), cmap="plasma")
        heat = ax.scatter(xy[:,0],xy[:,1],
                          s=0.1,alpha=0.5,c=colorcode,cmap="viridis", 
                          vmin=vmin, vmax=vmax)
    
    #cbar = fig.colorbar(heat)
    cbar = plt.colorbar(heat ,ax = [ax], location = 'right')
    cbar.set_label("Events per bin [1]", rotation=90,fontsize="small")
    cbar.set_ticks(colorbarticks,labels=colorbarticks,fontsize="small")

    if showhist:
        bins = np.arange(0,1.1,0.05)
        #bincenters = (bins[1:]+bins[:-1])/2
        hst = np.histogram(xy[:,0],bins=bins,density=True)[0]
        offset = 0.49
        for line in np.append([0,1],gridlines):
            ax.vlines(line,-0.02-offset,0.02-offset,color="black",lw=0.5)
        ax.plot([0,1],[-offset,-offset],color="black",lw=0.5)
        if isinstance(colorcode, str):
            ax.step(bins[:-1],hst/np.max(hst)/5-offset,
                    where="post",color=colorcode,alpha=0.4)
            ax.vlines(np.mean(xy[:,0][(xy[:,0]>0)*(xy[:,0]<1)]),
                      0.2-offset,0-offset,
                      color=colorcode)
        else:
            ax.step(bins[:-1],hst/np.max(hst)/5-offset,
                    where="post",color="steelblue",alpha=0.4)
            ax.vlines(np.mean(xy[:,0][(xy[:,0]>0)*(xy[:,0]<1)]),
                      0.2-offset,0-offset,
                      color="steelblue")
        ax.text(0-0.1,-offset-0.05,"V2",fontsize="x-small")
        ax.text(1+0.03,-offset-0.05,"V3",fontsize="x-small")
        ax.text(0.5,-offset-0.1,"Horizontal position",
                ha="center",fontsize="small")
    
    plt.tight_layout()
    
    ax.set_position([-0.01,0.1,0.74,0.85])
    
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_scpot_prespike_amplitude(data,
                                 target_dir,
                                 name="scpot_prespike_amplitude"):
    """
    Make a plot to compare prespike amplitude to the spacecraft potential.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_prespike_amplitude.png".

    Returns
    -------
    None.

    """
    #sc potential vs prespike amlitudes    
    fig,ax=plt.subplots()
    #mask for a resonable range of sc potentials and those that have prespikes
    mask = (data["scpots"]>0)*(data["scpots"]<20)*(~np.isnan(data["negative_prespikes"]))
    ax.scatter(data["scpots"][mask],
               -data["negative_prespikes"][mask],
               alpha=0.5,linewidth=0,color="maroon")
    ax.set_yscale('log')
    ax.set_xlabel("sc potenatial [V]")
    ax.set_ylabel("-prespike amplitude [V]")
    ax.set_xlim(0,20)
    ax.set_ylim(1e-4,5e-1)
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_scpot_overshoot(data,
                              target_dir,
                              name="scpot_overshoot"):
    """
    Make a plot to compare overshoot amplitude to the spacecraft potential.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_overshoot".

    Returns
    -------
    None.

    """
    
    
    #sc potential vs overshoot amlitudes    
    fig,ax=plt.subplots()
    #mask for a resonable range of sc potentials
    mask = (data["scpots"]>0)*(data["scpots"]<20)
    ax.scatter(data["scpots"][mask],
               -data["overshoot_1"][mask],
               alpha=0.3,linewidth=0,s=3,color=u"#357266",label="ant1")
    ax.scatter(data["scpots"][mask],
               -data["overshoot_2"][mask],
               alpha=0.3,linewidth=0,s=3,color=u"#CE4257",label="ant2")
    ax.scatter(data["scpots"][mask],
               -data["overshoot_3"][mask],
               alpha=0.3,linewidth=0,s=3,color=u"#312509",label="ant3")
    ax.set_yscale('log')
    ax.set_xlabel("sc potenatial [V]")
    ax.set_ylabel("-overshoot amplitude [V]")
    ax.set_xlim(0,20)
    ax.set_ylim(1e-4,5e-1)
    ax.legend()
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_primary_overshoot(data,
                                target_dir,
                                name="primary_overshoot"):
    """
    Make a plot to compare overshoot amplitude to the body peak.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_overshoot".

    Returns
    -------
    None.

    """    
    
    #mask for a resonable range of sc potentials
    mask1_max = (data["scpots"]>0)*(data["scpots"]<20)*(
        data["body_amplitudes"]>data["sec_present_v1"])
    mask2_max = (data["scpots"]>0)*(data["scpots"]<20)*(
        data["body_amplitudes"]>data["sec_present_v2"])
    mask3_max = (data["scpots"]>0)*(data["scpots"]<20)*(
        data["body_amplitudes"]>data["sec_present_v3"])
    
    mask1_nomax = (data["scpots"]>0)*(data["scpots"]<20)*(
        data["body_amplitudes"]<=data["sec_present_v1"])
    mask2_nomax = (data["scpots"]>0)*(data["scpots"]<20)*(
        data["body_amplitudes"]<=data["sec_present_v2"])
    mask3_nomax = (data["scpots"]>0)*(data["scpots"]<20)*(
        data["body_amplitudes"]<=data["sec_present_v3"])
    
    #sc potential vs overshoot amlitudes    
    fig,ax=plt.subplots()

    ax.scatter(data["body_amplitudes"][mask1_nomax],
               -data["overshoot_1"][mask1_nomax],
               alpha=0.3,linewidth=0,s=0.5,color=u"#0D3044",label="isn't dominant")
    ax.scatter(data["body_amplitudes"][mask2_nomax],
               -data["overshoot_2"][mask2_nomax],
               alpha=0.3,linewidth=0,s=0.5,color=u"#0D3044")
    ax.scatter(data["body_amplitudes"][mask3_nomax],
               -data["overshoot_3"][mask3_nomax],
               alpha=0.3,linewidth=0,s=0.5,color=u"#0D3044")
    
    ax.scatter(data["body_amplitudes"][mask1_max],
               -data["overshoot_1"][mask1_max],
               alpha=0.3,linewidth=0,s=0.5,color=u"#F35A3F",label="is dominant")
    ax.scatter(data["body_amplitudes"][mask2_max],
               -data["overshoot_2"][mask2_max],
               alpha=0.3,linewidth=0,s=0.5,color=u"#F35A3F")
    ax.scatter(data["body_amplitudes"][mask3_max],
               -data["overshoot_3"][mask3_max],
               alpha=0.3,linewidth=0,s=0.5,color=u"#F35A3F") 

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("body peak amplitude [V]")
    ax.set_ylabel("-overshoot amplitude [V]")
    ax.set_xlim(1e-4,1e0)
    ax.set_ylim(1e-4,5e-1)
    ax.legend()
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_secondary_overshoot(data,
                              target_dir,
                              name="secondary_overshoot"):
    """
    Make a plot to compare overshoot amplitude to the secondary peak.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_overshoot".

    Returns
    -------
    None.

    """    
    
    #mask for a resonable range of sc potentials
    mask1 = (data["scpots"]>0)*(data["scpots"]<20)*(data["sec_present_v1"]>0)
    mask2 = (data["scpots"]>0)*(data["scpots"]<20)*(data["sec_present_v2"]>0)
    mask3 = (data["scpots"]>0)*(data["scpots"]<20)*(data["sec_present_v3"]>0)
    
    #sc potential vs overshoot amlitudes    
    fig,ax=plt.subplots()
    
    ax.scatter(data["sec_present_v1"][mask1],
               -data["overshoot_1"][mask1],
               alpha=0.15,linewidth=0,s=4,color=u"#BA1200",label="ant1")
    ax.scatter(data["sec_present_v2"][mask2],
               -data["overshoot_2"][mask2],
               alpha=0.15,linewidth=0,s=4,color=u"#8BA12B",label="ant2")
    ax.scatter(data["sec_present_v3"][mask3],
               -data["overshoot_3"][mask3],
               alpha=0.15,linewidth=0,s=4,color=u"#DEC02B",label="ant3")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("antenna peak amplitude [V]")
    ax.set_ylabel("-overshoot amplitude [V]")
    ax.set_xlim(1e-4,1e0)
    ax.set_ylim(1e-4,5e-1)
    ax.legend()
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_stupid_max_overshoot(data,
                              target_dir,
                              name="stupid_max_overshoot"):
    """
    Make a plot to compare overshoot amplitude to the maximum peak.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_overshoot".

    Returns
    -------
    None.

    """    
    
    #mask for a resonable range of sc potentials
    mask = (data["scpots"]>0)*(data["scpots"]<20)
    
    #sc potential vs overshoot amlitudes    
    fig,ax=plt.subplots()
    
    ax.scatter(data["stupid_max_v1"][mask],
               -data["overshoot_1"][mask],
               alpha=0.15,linewidth=0,s=3,color=u"#BA1200",label="ant1")
    ax.scatter(data["stupid_max_v2"][mask],
               -data["overshoot_2"][mask],
               alpha=0.15,linewidth=0,s=3,color=u"#8BA12B",label="ant2")
    ax.scatter(data["stupid_max_v3"][mask],
               -data["overshoot_3"][mask],
               alpha=0.15,linewidth=0,s=3,color=u"#DEC02B",label="ant3")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("channel maximum [V]")
    ax.set_ylabel("-overshoot amplitude [V]")
    ax.set_xlim(1e-4,1e0)
    ax.set_ylim(1e-4,5e-1)
    ax.legend()
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_overshoot_amplitude_delay(data,
                              target_dir,
                              name="overshoot_amplitude_delay"):
    """
    Make a plot to compare secondary amplitude to the secondary delay.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_overshoot".

    Returns
    -------
    None.

    """

    #sc potential vs overshoot amlitudes    
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    (ax2,ax) = gs.subplots(sharex=True)
    
    #mask for a resonable range of sc potentials
    mask = (data["scpots"]!=0)
    ax.scatter(data["overshoot_delay_1"][mask]/1000,
               -data["overshoot_1"][mask],
               alpha=0.3,linewidth=0,s=2,color=u"#DB5461",label="ant1")
    ax.scatter(data["overshoot_delay_2"][mask]/1000,
               -data["overshoot_2"][mask],
               alpha=0.3,linewidth=0,s=2,color=u"#9395D3",label="ant2")
    ax.scatter(data["overshoot_delay_3"][mask]/1000,
               -data["overshoot_3"][mask],
               alpha=0.3,linewidth=0,s=2,color=u"#CAFE48",label="ant3")
    ax.set_yscale('log')
    ax.set_xlabel(r"overshoot delay [$\mu s$]")
    ax.set_ylabel(r"-overshoot amplitude [$V$]")
    ax.set_xlim(0,3000)
    ax.set_ylim(1e-4,5e-1)
    
    #histrogams
    bins = np.arange(0,3100,150)
    ax2.hist(data["overshoot_delay_1"][mask]/1000,bins=bins,density=True,
             lw=1,histtype='step',color=u"#DB5461")
    ax2.hist(data["overshoot_delay_2"][mask]/1000,bins=bins,density=True,
             lw=1,histtype='step',color=u"#9395D3")
    ax2.hist(data["overshoot_delay_3"][mask]/1000,bins=bins,density=True,
             lw=1,histtype='step',color=u"#CAFE48")
    
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_secondary_amplitude_delay(data,
                              target_dir,
                              name="overshoot_amplitude_delay"):
    """
    Make a plot to compare overshoot amplitude to the overshoot delay.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_overshoot".

    Returns
    -------
    None.

    """

    #sc potential vs overshoot amlitudes    
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    (ax2,ax) = gs.subplots(sharex=True)
    
    #mask for a resonable range of sc potentials
    mask1 = (data["sec_present_v1"]>0)
    mask2 = (data["sec_present_v2"]>0)
    mask3 = (data["sec_present_v3"]>0)
    
    #trends
    bins = np.arange(0,810,100)
    segments = bins
    medians1 = np.zeros(len(segments)-1)
    medians2 = np.zeros(len(segments)-1)
    medians3 = np.zeros(len(segments)-1)
    for i in range(len(segments)-1):
        medians1[i] = np.median(data["sec_present_v1"][mask1][
            (segments[i]<data["delay_v1"][mask1])*
            (data["delay_v1"][mask1]<segments[i+1])])
        medians2[i] = np.median(data["sec_present_v2"][mask2][
            (segments[i]<data["delay_v2"][mask2])*
            (data["delay_v2"][mask2]<segments[i+1])])
        medians3[i] = np.median(data["sec_present_v3"][mask3][
            (segments[i]<data["delay_v3"][mask3])*
            (data["delay_v3"][mask3]<segments[i+1])])
    
    #plot
    ax.scatter(data["delay_v1"][mask1],
               data["sec_present_v1"][mask1],
               alpha=0.3,linewidth=0,s=2,color=u"#B1436C",label="ant1")
    ax.scatter(data["delay_v2"][mask2],
               data["sec_present_v2"][mask2],
               alpha=0.3,linewidth=0,s=2,color=u"#08B9D9",label="ant2")
    ax.scatter(data["delay_v3"][mask3],
               data["sec_present_v3"][mask3],
               alpha=0.3,linewidth=0,s=2,color=u"#E2C928",label="ant3")
    ax.hlines(medians1,segments[:-1],segments[1:],color=u"#B1436C")
    ax.hlines(medians2,segments[:-1],segments[1:],color=u"#08B9D9")
    ax.hlines(medians3,segments[:-1],segments[1:],color=u"#E2C928")
    ax.set_yscale('log')
    ax.set_xlabel(r"secondary delay [$\mu s$]")
    ax.set_ylabel(r"secondary amplitude [$V$]")
    ax.set_xlim(0,800)
    ax.set_ylim(5e-4,4e-1)
    
    #histrogams
    bins = np.arange(0,820,40)
    ax2.hist(data["delay_v1"][mask1],bins=bins,density=0,
             lw=1,histtype='step',color=u"#DB5461")
    ax2.hist(data["delay_v2"][mask2],bins=bins,density=0,
             lw=1,histtype='step',color=u"#9395D3")
    ax2.hist(data["delay_v3"][mask3],bins=bins,density=0,
             lw=1,histtype='step',color=u"#CAFE48")
    ax2.vlines(np.mean(data["delay_v1"][mask1]),0,250,color=u"#DB5461")
    ax2.vlines(np.mean(data["delay_v2"][mask2]),0,250,color=u"#9395D3")
    ax2.vlines(np.mean(data["delay_v3"][mask3]),0,250,color=u"#CAFE48")
    
    
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_secondary_overshoot_delay(data,
                              target_dir,
                              name="overshoot_amplitude_delay"):
    """
    Make a plot to compare secondary delay to the overshoot delay.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "scpot_overshoot".

    Returns
    -------
    None.

    """

    #sc potential vs overshoot amlitudes    
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    (ax2,ax) = gs.subplots(sharex=True)
    
    #mask for a resonable range of sc potentials
    mask1 = (data["sec_present_v1"]>0)
    mask2 = (data["sec_present_v2"]>0)
    mask3 = (data["sec_present_v3"]>0)
    ax.scatter(data["delay_v1"][mask1],
               data["overshoot_delay_1"][mask1]/1000,
               alpha=0.3,linewidth=0,s=2,color=u"#201E50",label="ant1")
    ax.scatter(data["delay_v2"][mask2],
               data["overshoot_delay_2"][mask2]/1000,
               alpha=0.3,linewidth=0,s=2,color=u"#FF934F",label="ant2")
    ax.scatter(data["delay_v3"][mask3],
               data["overshoot_delay_3"][mask3]/1000,
               alpha=0.3,linewidth=0,s=2,color=u"#75DF68",label="ant3")
    ax.set_xlabel(r"secondary delay [$\mu s$]")
    ax.set_ylabel(r"overshoot delay [$\mu s$]")
    ax.set_xlim(0,800)
    ax.set_ylim(0,3000)
    
    #histrogams
    bins = np.arange(0,820,40)
    ax2.hist(data["delay_v1"][mask1],bins=bins,density=0,
             lw=1,histtype='step',color=u"#201E50")
    ax2.hist(data["delay_v2"][mask2],bins=bins,density=0,
             lw=1,histtype='step',color=u"#FF934F")
    ax2.hist(data["delay_v3"][mask3],bins=bins,density=0,
             lw=1,histtype='step',color=u"#75DF68")
    
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_primary_secondary_amplitude(data,
                                 target_dir,
                                 name="primary_secondary_amplitude"):
    """
    A plot to compare the amplitudes of the primary (body) 
    vs the highest of the secondary (antenna) peaks.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "primary_secondary_amplitude.png".

    Returns
    -------
    None.

    """
    
    #only those that show an antenna peak
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_secondary = number_of_secondary_present > 0
    data = data[mask_secondary]
    
    #prepare antenna-wise maximum
    total_max = np.max(np.vstack((data["stupid_max_v1"],
                                  data["stupid_max_v2"],
                                  data["stupid_max_v3"])).transpose(),
                       axis=1)
    
    fig,ax=plt.subplots()
    ax.scatter(data["body_amplitudes"],
               total_max,
               c="royalblue",
               alpha=0.2,linewidth=0,label="Events")
    #fit
    coef = np.polyfit(np.log10(
        data["body_amplitudes"][(data["body_amplitudes"]>1e-3)]),
                     np.log10(
        total_max[(data["body_amplitudes"]>1e-3)]),1)
    poly1d_fn = np.poly1d(coef)
    #print(10**(poly1d_fn(np.array([-3,-1])))/np.array([1e-3,1e-1]))
    ax.plot([1e-3,1e-1], 10**(poly1d_fn(np.array([-3,-1]))), '-r', lw=1,
            label=r"Fit, exp. = qqq".replace('qqq', str(np.round(coef[0],2))))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Primary peak amplitude [V]")
    ax.set_ylabel("Secondary peak amplitude [V]")
    ax.set_xlim(1e-4,5e-1)
    ax.set_ylim(1e-3,5e-1)
    ax.plot([1e-5,10],[1e-5,10],c="black",ls=(0,(1,0.7)),label="Identity (1:1)",alpha=0.5)
    ax.plot([3.333e-6,3.333],[1e-5,10],c="darkviolet",ls=(0,(4,1.5)),label="3:1",alpha=0.5)
    ax.plot([1e-6,1],[1e-5,10],c="crimson",ls=(0,(8,2)),label="10:1",alpha=0.5)
    ax.plot([1e0,1e-6],[1e1,1e-3],c="green",ls=(0,(1.5,2)),label="Eq. 18", lw=1, alpha=1)
    ax.vlines([1e-3,1e-1],1e-5,1,color="darkgrey",ls=(0,(12,6)),alpha=0.4)
    ax.hlines(10**(poly1d_fn(np.array([-3,-1]))),1e-4,1,
              color="darkgrey",ls=(0,(12,6)),alpha=0.4)

    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_secondary_max_min(data,
                                legacy,
                                target_dir,
                                threshold = 20,
                                name="secondary_max_min",
                                ):
    """
    A function to plot the ratio of highest peak global maximum 
    vs the lowes peak blobal maximum as a function of heliocentric distance.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    legacy : bool
        If true, the old approach will be used for comatibility.
    target_dir : str
        An absolute path wher to put the resulting plots.
    threshold : float
        An upper y-axis bound.
    name : str, optional
        Output png name. The default is "secondary_max_min".

    Returns
    -------
    None.

    """
    
    #only those that show an antenna peak
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_secondary = number_of_secondary_present > 0
    data["seecondary"] = mask_secondary
    data = data[mask_secondary]
    
    if legacy:    
        total_max = np.max(np.vstack((data["stupid_max_v1"],
                                      data["stupid_max_v2"],
                                      data["stupid_max_v3"])).transpose(),
                           axis=1)
        total_min = np.min(np.vstack((data["stupid_max_v1"],
                                      data["stupid_max_v2"],
                                      data["stupid_max_v3"])).transpose(),
                           axis=1)
        
        maxima_each_other = total_max / total_min
    else:
        total_max = np.max(np.vstack((data["sec_present_v1"],
                                      data["sec_present_v2"],
                                      data["sec_present_v3"])).transpose(),
                           axis=1)
        
        maxima_each_other = total_max / data["body_amplitudes"]
    
    segments = np.arange(0.55,1.05,0.05)
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(maxima_each_other[
            (maxima_each_other<threshold)*
            (segments[i]<data["heliocentric_distances"])*
            (data["heliocentric_distances"]<segments[i+1])])
        medians[i] = np.median(maxima_each_other[
            (maxima_each_other<threshold)*
            (segments[i]<data["heliocentric_distances"])*
            (data["heliocentric_distances"]<segments[i+1])])
    
    
    binsy = np.arange(0,threshold,1)
    
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    (ax,ax2) = gs.subplots(sharey=True)
    
    ax2.hist(maxima_each_other, bins=binsy, orientation = "horizontal",
             color="rebeccapurple", histtype = "step")
    ax2.hlines(np.mean(maxima_each_other[maxima_each_other<threshold]),0,200,color="tomato")
    ax2.hlines(np.median(maxima_each_other[maxima_each_other<threshold]),0,200,color="black")
    ax2.set_xlabel("Frequency [1]")

    ax.scatter(data["heliocentric_distances"],maxima_each_other,
               alpha=0.2,edgecolor="none",color="rebeccapurple")
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="black")
    ax.set_xlabel(r"Heliocentric distance [$AU$]")
    ax.set_ylabel(r"Sec. peak / prim. peak [V/V]")
    ax.set_xlim(0.49,1.03)
    ax.set_ylim(0,threshold)
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax2.xaxis.set_label_coords(0.5, -0.15)
    
    ax.legend(fontsize="small",loc=2)
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_secondary_max_min_root(data,
                                     target_dir,
                                     name="secondary_max_min_root",
                                     exponent = 0.667):
    """
    A function to plot the ratio of highest peak global maximum 
    vs the cube root of the lowest peak global maximum 
    as a function of heliocentric distance.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    legacy : bool
        If true, the old approach will be used for comatibility.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "secondary_max_min_cuberoot".
    exponent : float, potional
        The exponent to relate the primary and the secondary peak. 
        The default is 0.667.

    Returns
    -------
    None.

    """
    
    #only those that show an antenna peak
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_secondary = number_of_secondary_present > 0
    data["seecondary"] = mask_secondary
    data = data[mask_secondary]
    
    total_max = np.max(np.vstack((data["sec_present_v1"],
                                  data["sec_present_v2"],
                                  data["sec_present_v3"])).transpose(),
                       axis=1)
    
    maxima_each_other = total_max / (data["body_amplitudes"]**exponent)
    
    segments = np.arange(0.55,1.05,0.05)
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(maxima_each_other[
            (maxima_each_other<20)*
            (segments[i]<data["heliocentric_distances"])*
            (data["heliocentric_distances"]<segments[i+1])])
        medians[i] = np.median(maxima_each_other[
            (maxima_each_other<20)*
            (segments[i]<data["heliocentric_distances"])*
            (data["heliocentric_distances"]<segments[i+1])])
    
    
    binsy = np.arange(0,5,0.2)
    
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    (ax,ax2) = gs.subplots(sharey=True)
    
    ax2.hist(maxima_each_other, bins=binsy, orientation = "horizontal",
             color="rebeccapurple", histtype = "step")
    ax2.hlines(np.mean(maxima_each_other[maxima_each_other<20]),0,200,color="tomato")
    ax2.hlines(np.median(maxima_each_other[maxima_each_other<20]),0,200,color="black")
    ax2.set_xlabel("Frequency [1]")

    ax.scatter(data["heliocentric_distances"],maxima_each_other,
               alpha=0.2,edgecolor="none",color="rebeccapurple")
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="black")
    ax.set_xlabel(r"Heliocentric distance [$AU$]")
    ax.set_ylabel(r"Sec. peak / prim. peak$^{qqq}$ [arb.u.]".replace(
        "qqq",str(np.round(exponent,2))))
    ax.set_xlim(0.49,1.03)
    ax.set_ylim(0,5)
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax2.xaxis.set_label_coords(0.5, -0.15)
    
    ax.legend(fontsize="small",loc=2)
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_secondary_max_min_vuv(data,
                                    target_dir,
                                    name="secondary_max_min_vuv",
                                    ):
    """
    A function to plot the ratio of highest peak global maximum 
    vs the lowes peak blobal maximum as a function of vuv illumination.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "secondary_max_min".

    Returns
    -------
    None.

    """
    
    #only those that show an antenna peak
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_secondary = number_of_secondary_present > 0
    data["seecondary"] = mask_secondary
    data = data[mask_secondary]
    
    
    total_max = np.max(np.vstack((data["sec_present_v1"],
                                  data["sec_present_v2"],
                                  data["sec_present_v3"])).transpose(),
                       axis=1)
    
    maxima_each_other = total_max / data["body_amplitudes"]
    
    segments = np.logspace(-0.5,0.6,10)#np.arange(0.5,4.05,0.5)
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(maxima_each_other[
            (maxima_each_other<20)*
            (segments[i]<data["vuv_illumination"])*
            (data["vuv_illumination"]<segments[i+1])])
        medians[i] = np.median(maxima_each_other[
            (maxima_each_other<20)*
            (segments[i]<data["vuv_illumination"])*
            (data["vuv_illumination"]<segments[i+1])])
    
    
    binsy = np.arange(0,20,1)
    
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    (ax,ax2) = gs.subplots(sharey=True)
    
    ax2.hist(maxima_each_other, bins=binsy, orientation = "horizontal",
             color="yellowgreen", histtype = "step")
    ax2.hlines(np.mean(maxima_each_other[maxima_each_other<20]),0,200,color="C0")
    ax2.hlines(np.median(maxima_each_other[maxima_each_other<20]),0,200,color="C1")
    ax2.set_xlabel("Frequency [1]")

    ax.scatter(data["vuv_illumination"],maxima_each_other,
               alpha=0.2,edgecolor="none",color="yellowgreen")
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="C0")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="C1")
    ax.set_xlabel(r"VUV illumination [$arb. u.$]")
    ax.set_ylabel(r"Sec. peak / prim. peak [V/V]")
    ax.set_xlim(0.4,4.03)
    ax.set_ylim(0,20)
    ax.set_xscale("log")
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax2.xaxis.set_label_coords(0.5, -0.15)
    
    ax.legend(fontsize="small",loc=2)
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_antenna_ampltiude_delay(data,
                              target_dir,
                              name="antenna_amplitude_delay",
                              showhist = True):
    """
    A function to make two plots that intend to compare the relative 
    primary / secondary peak amplitude and the delay of the maximum secondary
    vs the body peak.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "antenna_amplitude_delay".
    showhist : bool, optional
        Whether to make y-marginal histogram or not. 

    Returns
    -------
    None.

    """
    
    #prepare antenna-wise maximum
    total_max = np.max(np.vstack((data["sec_present_v1"],
                                  data["sec_present_v2"],
                                  data["sec_present_v3"])).transpose(),
                       axis=1)
    rel_max = total_max/data["body_amplitudes"]
    
    #prepare the time delay of the antenna-wise maximum
    total_max_delays = np.zeros(len(data.index))
    for i in range(len(data.index)):
        df = data.iloc[i]
        total_max_delays[i] = [df["delay_v1"],
                               df["delay_v2"],
                               df["delay_v3"]][
                                   np.argmax([df["sec_present_v1"],
                                              df["sec_present_v2"],
                                              df["sec_present_v3"]])]
          
    #colorscale delay
    fig,ax=plt.subplots()
    d = ax.scatter(data["body_amplitudes"][total_max_delays>0],
               total_max[total_max_delays>0],
               c=np.log10(total_max_delays[total_max_delays>0]),
               alpha=0.5,linewidth=0)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("el amp [V]")
    ax.set_ylabel("max amp [V]")
    ax.set_xlim(1e-4,5e-1)
    ax.set_ylim(1e-3,5e-1)
    ax.plot([1e-5,10],[1e-5,10],
            c="black",ls="dashed",label="identity",alpha=0.3)
    ax.plot([3.333e-6,3.333],[1e-5,10],
            c="darkviolet",ls="dashed",label="3:1",alpha=0.3)
    ax.plot([1e-6,1],[1e-5,10],
            c="crimson",ls="dashed",label="10:1",alpha=0.3)
    ax.legend(fontsize="small")
    cbar = fig.colorbar(d)
    cbar.set_label("Delay of the strongest peak [$log_{10}(\mu s)$]")
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)
    
    
    
    #relative amp vs delay
    rel_max_cuberoot = total_max/(data["body_amplitudes"]**(1/3))
    logbins =np.arange(-2,2,0.2)
    bins_mids = 10**((logbins[1:]+logbins[:-1])/2)
    segments = 10**logbins
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<rel_max_cuberoot)*
            (rel_max_cuberoot<segments[i+1])])
        medians[i] = np.median(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<rel_max_cuberoot)*
            (rel_max_cuberoot<segments[i+1])])
        
    binsy = 10**np.arange(0.8,3.1,0.1)
    
    
    
    
    if showhist:
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        (ax,ax2) = gs.subplots(sharey=True)
        
        ax2.hist(total_max_delays[total_max>1*data["body_amplitudes"]],
                 bins=binsy, orientation = "horizontal",
                 color=u"#CD7A98", histtype = "step")
        ax2.hlines(np.mean(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="tomato")
        ax2.hlines(np.median(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="black")
        ax2.set_xlabel("Frequency [1]")
        ax2.xaxis.set_label_coords(0.5, -0.15)
    else:
        fig,ax=plt.subplots()

    ax.scatter(rel_max_cuberoot[total_max>1*data["body_amplitudes"]],
               total_max_delays[total_max>1*data["body_amplitudes"]],
               c=u"#CD7A98",alpha=0.3,linewidth=0)
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="black") 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"Strongest peak / primary peak$^\frac{1}{3}$ [arb. u.]")
    ax.set_ylabel(r"Delay of the strongest peak [$\mu s$]")
    ax.set_xlim(2e-2,2e0)
    ax.set_ylim(7,1e3)
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    
    
    ax.legend(fontsize="small",loc=3)
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+"_simple_cuberoot"+".pdf", format='pdf')
    
    
    
    #relative amp vs delay
    logbins =np.arange(0,2,0.2)
    bins_mids = 10**((logbins[1:]+logbins[:-1])/2)
    segments = 10**logbins
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<rel_max)*
            (rel_max<segments[i+1])])
        medians[i] = np.median(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<rel_max)*
            (rel_max<segments[i+1])])
        
    binsy = 10**np.arange(0.8,3.1,0.1)

    if showhist:
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        (ax,ax2) = gs.subplots(sharey=True)
        
        ax2.hist(total_max_delays[total_max>1*data["body_amplitudes"]],
                 bins=binsy, orientation = "horizontal",
                 color=u"#CD7A98", histtype = "step")
        ax2.hlines(np.mean(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="tomato")
        ax2.hlines(np.median(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="black")
        ax2.set_xlabel("Frequency [1]")
        ax2.xaxis.set_label_coords(0.5, -0.15)
    else:
        fig,ax=plt.subplots()

    ax.scatter(rel_max[total_max>1*data["body_amplitudes"]],
               total_max_delays[total_max>1*data["body_amplitudes"]],
               c=u"#CD7A98",alpha=0.3,linewidth=0)
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="black") 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"Strongest peak / primary peak [1]")
    ax.set_ylabel(r"Delay of the strongest peak [$\mu s$]")
    ax.set_xlim(8e-1,1.5e2)
    ax.set_ylim(7,1e3)
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    
    ax.legend(fontsize="small",loc=4)
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+"_simple_relative"+".pdf", format='pdf')
    
    
    
    #abslute amp vs delay
    logbins =np.arange(-2.6,0,0.2)
    bins_mids = 10**((logbins[1:]+logbins[:-1])/2)
    segments = 10**logbins
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<total_max)*
            (total_max<segments[i+1])])
        medians[i] = np.median(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<total_max)*
            (total_max<segments[i+1])])
        
    binsy = 10**np.arange(0.8,3.1,0.1)
    
    if showhist:
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        (ax,ax2) = gs.subplots(sharey=True)
        
        ax2.hist(total_max_delays[total_max>1*data["body_amplitudes"]],
                 bins=binsy, orientation = "horizontal",
                 color=u"#CD7A98", histtype = "step")
        ax2.hlines(np.mean(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="tomato")
        ax2.hlines(np.median(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="black")
        ax2.set_xlabel("Frequency [1]")
        ax2.xaxis.set_label_coords(0.5, -0.15)
    else:
        fig,ax=plt.subplots()

    ax.scatter(total_max[total_max>1*data["body_amplitudes"]],
               total_max_delays[total_max>1*data["body_amplitudes"]],
               c=u"#CD7A98",alpha=0.3,linewidth=0)
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="black") 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"Strongest peak [$V$]")
    ax.set_ylabel(r"Delay of the strongest peak [$\mu s$]")
    ax.set_xlim(2.5e-3,4e-1)
    ax.set_ylim(7,1e3)
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    
    ax.legend(fontsize="small",loc=3)
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+"_simple"+".pdf", format='pdf')


def make_plot_heliocentric_antenna_delay(data,
                                         target_dir,
                                         name="heliocentric_delay",
                                         showhist = True):
    """
    A function to plot the delay of the highest antenna peak with respect to 
    the body peak as a function of heliocentric distance.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "secondary_max_min".
    showhist : bool, optional
        Whether to make y-marginal histogram or not. 

    Returns
    -------
    None.

    """
    
    #only those that show an antenna peak
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_secondary = number_of_secondary_present > 0
    data["seecondary"] = mask_secondary
    data = data[mask_secondary]
    

    total_max_delays = np.zeros(len(data.index))
    for i in range(len(data.index)):
        df = data.iloc[i]
        total_max_delays[i] = [df["delay_v1"],
                               df["delay_v2"],
                               df["delay_v3"]][
                                   np.argmax([df["sec_present_v1"],
                                              df["sec_present_v2"],
                                              df["sec_present_v3"]])]
                                   
    total_max = np.max(np.vstack((data["sec_present_v1"],
                                  data["sec_present_v2"],
                                  data["sec_present_v3"])).transpose(),
                       axis=1)
             
    segments = np.arange(0.55,1.05,0.05)
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<data["heliocentric_distances"])*
            (data["heliocentric_distances"]<segments[i+1])])
        medians[i] = np.median(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<data["heliocentric_distances"])*
            (data["heliocentric_distances"]<segments[i+1])])
    
    
    #binsy = np.arange(0,1000,50)
    binsy = 10**np.arange(0.8,3.1,0.1)

    if showhist:
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        (ax,ax2) = gs.subplots(sharey=True)
        
        
        ax2.hist(total_max_delays[total_max>1*data["body_amplitudes"]],
                 bins=binsy, orientation = "horizontal",
                 color=u"#CD7A98", histtype = "step")
        ax2.hlines(np.mean(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="tomato")
        ax2.hlines(np.median(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="black")
        ax2.set_xlabel("Frequency [1]")
        ax2.xaxis.set_label_coords(0.5, -0.15)
    else:
        fig,ax=plt.subplots()
    
    ax.set_yscale('log')

    ax.scatter(data["heliocentric_distances"][total_max>1*data["body_amplitudes"]],
               total_max_delays[total_max>1*data["body_amplitudes"]],
               alpha=0.3,edgecolor="none",color=u"#CD7A98")
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="black")
    ax.set_xlabel(r"Heliocentric distance [$AU$]")
    ax.set_ylabel(r"Delay of the strongest peak [$\mu s$]")
    ax.set_xlim(0.49,1.03)
    ax.set_ylim(7,1e3)
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    
    ax.legend(fontsize="small",loc=3) 
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_vuv_antenna_delay(data,
                                target_dir,
                                name="vuv_delay",
                                showhist = True):
    """
    A function to plot the delay of the highest antenna peak with respect to 
    the body peak as a function of vuv irradiance.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "secondary_max_min".
    showhist : bool, optional
        Whether to make y-marginal histogram or not. 

    Returns
    -------
    None.

    """
    
    #only those that show an antenna peak
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_secondary = number_of_secondary_present > 0
    data["seecondary"] = mask_secondary
    data = data[mask_secondary]
    

    total_max_delays = np.zeros(len(data.index))
    for i in range(len(data.index)):
        df = data.iloc[i]
        total_max_delays[i] = [df["delay_v1"],
                               df["delay_v2"],
                               df["delay_v3"]][
                                   np.argmax([df["sec_present_v1"],
                                              df["sec_present_v2"],
                                              df["sec_present_v3"]])]
                                   
    total_max = np.max(np.vstack((data["sec_present_v1"],
                                  data["sec_present_v2"],
                                  data["sec_present_v3"])).transpose(),
                       axis=1)
             
    segments = np.logspace(-0.5,0.6,10)#np.arange(0.,5.05,0.5)
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<data["vuv_illumination"])*
            (data["vuv_illumination"]<segments[i+1])])
        medians[i] = np.median(total_max_delays[
            (total_max>1*data["body_amplitudes"])*
            (segments[i]<data["vuv_illumination"])*
            (data["vuv_illumination"]<segments[i+1])])
    
    
    #binsy = np.arange(0,1000,50)
    binsy = 10**np.arange(0.8,3.1,0.1)
    
    if showhist:
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        (ax,ax2) = gs.subplots(sharey=True)
        
        ax2.hist(total_max_delays[total_max>1*data["body_amplitudes"]],
                 bins=binsy, orientation = "horizontal",
                 color=u"#CD7A98", histtype = "step")
        ax2.hlines(np.mean(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="tomato")
        ax2.hlines(np.median(total_max_delays[total_max>1*data["body_amplitudes"]]),
                   0,100,color="black")
        ax2.set_xlabel("Frequency [1]")
        ax2.xaxis.set_label_coords(0.5, -0.15)
    else:
        fig,ax=plt.subplots()
        
    ax.set_yscale('log')

    ax.scatter(data["vuv_illumination"][total_max>1*data["body_amplitudes"]],
               total_max_delays[total_max>1*data["body_amplitudes"]],
               alpha=0.3,edgecolor="none",color=u"#CD7A98")
    ax.hlines(means,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians,segments[:-1],segments[1:],label="Median",color="black")
    ax.set_xlabel(r"VUV illumination [$arb. u.$]")
    ax.set_ylabel(r"Delay of the strongest peak [$\mu s$]")
    ax.set_xlim(0.3,6.05)
    ax.set_ylim(7,1e3)
    ax.set_xscale("log")
    
    ax.xaxis.set_label_coords(0.5, -0.15)
    
    ax.legend(fontsize="small",loc=3) 
    plt.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_hist_delay(data,
                         target_dir,
                         name="hist_delay"):
    """
    The function to make a hitogram of delay of the secondary peak behind the 
    primary peak. 

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    target_dir : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is "hist_delay".

    Returns
    -------
    None.

    """
    
    #only those that show an antenna peak
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_secondary = number_of_secondary_present > 0
    data["seecondary"] = mask_secondary
    data = data[mask_secondary]
    

    total_max_delays = np.zeros(len(data.index))
    for i in range(len(data.index)):
        df = data.iloc[i]
        total_max_delays[i] = [df["delay_v1"],
                               df["delay_v2"],
                               df["delay_v3"]][
                                   np.argmax([df["sec_present_v1"],
                                              df["sec_present_v2"],
                                              df["sec_present_v3"]])]
                                   
    total_max = np.max(np.vstack((data["sec_present_v1"],
                                  data["sec_present_v2"],
                                  data["sec_present_v3"])).transpose(),
                       axis=1)
    
    fig,ax=plt.subplots()
    binsy = 10**np.arange(0.8,3.1,0.1)
    ax.set_xscale('log')
    ax.hist(total_max_delays[total_max>1*data["body_amplitudes"]],
             bins=binsy, orientation = "vertical",
             color=u"#CD7A98", histtype = "step")
    ax.vlines(np.mean(total_max_delays[total_max>1*data["body_amplitudes"]]),
               0,400,color="tomato",label="Mean")
    ax.vlines(np.median(total_max_delays[total_max>1*data["body_amplitudes"]]),
               0,400,color="black",label="Median")
    ax.set_ylabel("Frequency [1]")
    ax.set_xlabel(r"Delay of the strongest peak [$\mu s$]")
    ax.legend(loc=2)
    
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')

def make_plot_hist_body_amplitude(data,
                       target_dir,
                       name="hist_body_amplitude",
                       bootstrap=False):
    """
    The function to make a plot of thehistogram of body amplitudes,
    which is believed to be a better measure of the actual charge 
    generated than the ampltiude of global max.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "hist_body_amplitude".
    bootstrap: bool, optional
        Whether to show the bootstrap shadows. Default is False.

    Returns
    -------
    None.

    """
    
    #histogram of true impact amplitudes
    fig,ax=plt.subplots()
    #histogram
    logbins = np.arange(-3.3,0,0.1)
    bins = 10**(logbins)
    bincenters = 10**((logbins[1:]+logbins[:-1])/2)
    #binwidths = bins[1:]-bins[:-1]
    hstgrm = np.histogram(data["body_amplitudes"], bins=bins, density=True)[0]
    #bootstrap
    b=1000
    bootstrap_body_amp = np.random.choice(
                                    data["body_amplitudes"],
                                    size=b*len(data["body_amplitudes"]),
                                    replace=True)
    bootstrap_body_amp = np.reshape(bootstrap_body_amp, 
                                    (b,len(data["body_amplitudes"])))
    bootstrap_hstgrm = np.zeros((0,len(bincenters)))
    for i in range(b):
        bootstrap_hstgrm = np.vstack(
                            (bootstrap_hstgrm,
                             np.histogram(bootstrap_body_amp[i,:], 
                                          bins=bins, 
                                          density=True)[0]))
    #plot
    ax.step(bincenters,hstgrm,where="mid",c="black")
    if bootstrap:
        for i in range(b):
            ax.step(bincenters,bootstrap_hstgrm[i,:],
                    where="mid",c="gray",zorder=0,alpha=0.01)
    ax.errorbar(bincenters,hstgrm,
                yerr=np.vstack((
                        hstgrm-np.quantile(bootstrap_hstgrm,0.05,axis=0),
                        np.quantile(bootstrap_hstgrm,0.95,axis=0)-hstgrm)),
                fmt='none',ecolor="black")
    ax.text(6e-2,1e2,r"$\mu = $"+str(
                                np.round(1000*np.mean(data["body_amplitudes"]),1))
                                +" $mV$",
            color="black")
    ax.text(6e-2,2e1,r"$med = $"+str(np.round(
                                    1000*np.median(data["body_amplitudes"]),1))
                                +" $mV$",
            color="purple")
    ax.vlines(np.median(data["body_amplitudes"]),1e-5,1e5,
              alpha=0.3,color="purple")
    ax.vlines(np.mean(data["body_amplitudes"]),1e-5,1e5,
              alpha=0.3,color="black")
    ax.set_yscale("log")
    ax.set_xscale("log")
    x_lo = 2e-4
    x_hi = 1
    conversion = 1e3
    ax.set_xlim(x_lo,x_hi)
    ax.set_ylim(1e-3,1e3)
    ax.set_xlabel(r"Primary peak voltage [$V$]")
    ax.set_ylabel(r"Prob. density [$V^{-1}$]")
    ax2 = ax.twiny()
    ax2.set_xlim(x_lo*conversion,x_hi*conversion)
    ax2.set_xscale("log")
    ax2.set_xlabel('Impact charge [pC]')
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_hist_antenna_amplitude(data,
                      target_dir,
                      name="hist_antenna_amplitude",
                      bootstrap=False):
    """
    The function to make a plot of the histogram of ant2 amplitudes,
    which is a poor measure of the actual charge.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "hist_antenna_amplitude".
    bootstrap: bool, optional
        Whether to show the bootstrap shadows. Default is False.

    Returns
    -------
    None.

    """
    
    #histogram of true impact amplitudes
    fig,ax=plt.subplots()
    #histogram
    logbins = np.arange(-3.3,0,0.1)
    bins = 10**(logbins)
    bincenters = 10**((logbins[1:]+logbins[:-1])/2)
    #binwidths = bins[1:]-bins[:-1]
    hstgrm = np.histogram(data["stupid_max_v2"], bins=bins, density=True)[0]
    #bootstrap
    b=1000
    bootstrap_body_amp = np.random.choice(
                                    data["stupid_max_v2"],
                                    size=b*len(data["stupid_max_v2"]),
                                    replace=True)
    bootstrap_body_amp = np.reshape(bootstrap_body_amp, 
                                    (b,len(data["stupid_max_v2"])))
    bootstrap_hstgrm = np.zeros((0,len(bincenters)))
    for i in range(b):
        bootstrap_hstgrm = np.vstack(
                            (bootstrap_hstgrm,
                             np.histogram(bootstrap_body_amp[i,:], 
                                          bins=bins, 
                                          density=True)[0]))
    #plot
    ax.step(bincenters,hstgrm,where="mid",c="black")
    if bootstrap:
        for i in range(b):
            ax.step(bincenters,bootstrap_hstgrm[i,:],
                    where="mid",c="gray",zorder=0,alpha=0.01)
    ax.errorbar(bincenters,hstgrm,
                yerr=np.vstack((
                        hstgrm-np.quantile(bootstrap_hstgrm,0.05,axis=0),
                        np.quantile(bootstrap_hstgrm,0.95,axis=0)-hstgrm)),
                fmt='none',ecolor="limegreen")
    ax.text(6e-2,1e2,r"$\mu = $"+str(
                                np.round(1000*np.mean(data["stupid_max_v2"]),1))
                                +" $mV$",
            color="black")
    ax.text(6e-2,2e1,r"$med = $"+str(np.round(
                                    1000*np.median(data["stupid_max_v2"]),1))
                                +" $mV$",
            color="purple")
    ax.vlines(np.median(data["stupid_max_v2"]),1e-5,1e5,
              alpha=0.3,color="purple")
    ax.vlines(np.mean(data["stupid_max_v2"]),1e-5,1e5,
              alpha=0.3,color="black")
    ax.set_yscale("log")
    ax.set_xscale("log")
    x_lo = 2e-4
    x_hi = 1
    conversion = 1e99
    ax.set_xlim(x_lo,x_hi)
    ax.set_ylim(1e-3,1e3)
    ax.set_xlabel(r"Ant2 peak voltage [$V$]")
    ax.set_ylabel(r"Prob. density [$V^{-1}$]")
    ax2 = ax.twiny()
    ax2.set_xlim(x_lo*conversion,x_hi*conversion)
    ax2.set_xscale("log")
    ax2.set_xlabel('Ant2 charge [pC]')
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_hist_body_both(data,
                       target_dir,
                       name="hist_body_amplitude_both",
                       bootstrap=False):
    """
    The function to make a plot of the histogram of body amplitudes,
    overplotted with the histogram of only triple hits.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "hist_body_amplitude".
    bootstrap: bool, optional
        Whether to show the bootstrap shadows. Default is False.

    Returns
    -------
    None.

    """
    #triple hits selection
    subselection = data[(data["alright"]==True) & (
                                    (data["sec_present_v1"]<1e-5) &
                                    (data["sec_present_v2"]<1e-5) &
                                    (data["sec_present_v3"]<1e-5) )]
    
    
    #histogram of true impact amplitudes
    fig,ax=plt.subplots()
    #histogram
    logbins = np.arange(-3.3,0,0.1)
    bins = 10**(logbins)
    bincenters = 10**((logbins[1:]+logbins[:-1])/2)
    hstgrm = np.histogram(data["body_amplitudes"], bins=bins, density=True)[0]
    
    logbins_sub = np.arange(-3.3,0,0.2)
    bins_sub = 10**(logbins_sub)
    bincenters_sub = 10**((logbins_sub[1:]+logbins_sub[:-1])/2)
    hstgrm_sub = np.histogram(subselection["body_amplitudes"], bins=bins_sub, density=True)[0]
    
    
    
    
    
    #bootstrap all
    b=1000
    bootstrap_body_amp = np.random.choice(
                                    data["body_amplitudes"],
                                    size=b*len(data["body_amplitudes"]),
                                    replace=True)
    bootstrap_body_amp = np.reshape(bootstrap_body_amp, 
                                    (b,len(data["body_amplitudes"])))
    bootstrap_hstgrm = np.zeros((0,len(bincenters)))
    for i in range(b):
        bootstrap_hstgrm = np.vstack(
                            (bootstrap_hstgrm,
                             np.histogram(bootstrap_body_amp[i,:], 
                                          bins=bins, 
                                          density=True)[0]))
    
    #bootstrap triple hits
    b=1000
    bootstrap_body_amp_sub = np.random.choice(
                                    subselection["body_amplitudes"],
                                    size=b*len(subselection["body_amplitudes"]),
                                    replace=True)
    bootstrap_body_amp_sub = np.reshape(bootstrap_body_amp_sub, 
                                    (b,len(subselection["body_amplitudes"])))
    bootstrap_hstgrm_sub = np.zeros((0,len(bincenters_sub)))
    for i in range(b):
        bootstrap_hstgrm_sub = np.vstack(
                            (bootstrap_hstgrm_sub,
                             np.histogram(bootstrap_body_amp_sub[i,:], 
                                          bins=bins_sub, 
                                          density=True)[0]))

    
    #plot
    ax.step(bincenters,hstgrm,where="mid",c="black")
    ax.step(bincenters_sub,hstgrm_sub,where="mid",c="skyblue",alpha=0.8)
    
    if bootstrap:
        for i in range(b):
            ax.step(bincenters,bootstrap_hstgrm[i,:],
                    where="mid",c="gray",zorder=0,alpha=0.01)
    ax.errorbar(bincenters,hstgrm,
                yerr=np.vstack((
                        hstgrm-np.quantile(bootstrap_hstgrm,0.05,axis=0),
                        np.quantile(bootstrap_hstgrm,0.95,axis=0)-hstgrm)),
                fmt='none',ecolor="black")
    ax.errorbar(bincenters_sub,hstgrm_sub,
                yerr=np.vstack((
                        hstgrm_sub-np.quantile(bootstrap_hstgrm_sub,0.05,axis=0),
                        np.quantile(bootstrap_hstgrm_sub,0.95,axis=0)-hstgrm_sub)),
                fmt='none',ecolor="skyblue",alpha=0.8)
    ax.text(6e-2,1e2,r"$\mu = $"+str(
                                np.round(1000*np.mean(data["body_amplitudes"]),1))
                                +" $mV$",
            color="tomato")
    ax.text(6e-2,2e1,r"$med = $"+str(np.round(
                                    1000*np.median(data["body_amplitudes"]),1))
                                +" $mV$",
            color="black")
    ax.vlines(np.mean(data["body_amplitudes"]),1e-5,1e5,
              alpha=0.8,color="tomato")
    ax.vlines(np.median(data["body_amplitudes"]),1e-5,1e5,
              alpha=0.8,color="black")
    ax.set_yscale("log")
    ax.set_xscale("log")
    x_lo = 2e-4
    x_hi = 1
    conversion = 1e3
    ax.set_xlim(x_lo,x_hi)
    ax.set_ylim(1e-3,1e3)
    ax.set_xlabel(r"Primary peak voltage [$V$]")
    ax.set_ylabel(r"Prob. density [$V^{-1}$]")
    ax2 = ax.twiny()
    ax2.set_xlim(x_lo*conversion,x_hi*conversion)
    ax2.set_xscale("log")
    ax2.set_xlabel('Impact charge [pC]')
    fig.tight_layout()
    ax.set_position([0.16,0.17,0.75,0.662])
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_heliocentric_body_decay(data,
                                 target_dir,
                                 name="heliocentric_body_decay"):
    """
    The function to plot the primary (body) peak decay time
    as a function of heliocentric distance.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "heliocentric_body_decay".

    Returns
    -------
    None.

    """

    filtered_body_decay = data["body_decay_times"][
                                (data["body_decay_times"]>0)*
                                (data["body_decay_times"]<2.01e-4)]  
    filtered_helio_r = data["heliocentric_distances"][
                                    (data["body_decay_times"]>0)*
                                    (data["body_decay_times"]<2.01e-4)]
    segments = np.arange(0.5,1.01,0.1)
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(filtered_body_decay[
            (segments[i]<filtered_helio_r)*
            (filtered_helio_r<segments[i+1])])
        medians[i] = np.median(filtered_body_decay[
            (segments[i]<filtered_helio_r)*
            (filtered_helio_r<segments[i+1])])
    segments_mids = (segments[1:]+segments[:-1])/2
    fig,ax=plt.subplots()
    ax.scatter(filtered_helio_r,filtered_body_decay*1e6,
               alpha=0.5,edgecolor="none",color="goldenrod")
    ax.hlines(means*1e6,segments[:-1],segments[1:],label="Mean",color="tomato")
    ax.hlines(medians*1e6,segments[:-1],segments[1:],label="Median",color="black") 
    ax.plot(np.arange(0.4,1.1,0.01),93*(np.arange(0.4,1.1,0.01))**(2),
            label="Theory",ls="dashed",color="C2")
    ax.plot(np.arange(0.4,1.1,0.01),2*93*(np.arange(0.4,1.1,0.01))**(2),
            label="2x Theory",ls="dashed",color="C3")
    ax.set_xlim(0.45,1.05)
    ax.set_ylim(0,200)
    ax.set_xlabel(r"Heliocentric distance [$AU$]")
    ax.set_ylabel(r"Primary peak decay time [$\mu s$]")
    ax.legend(fontsize="small",loc=2)
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')
    

def make_plot_heliocentric_body_risetime(data,
                                    target_dir,
                                    name="heliocentric_body_risetime"):
    """
    The function to plot the primary (body) peak risetime as 
    a function of heliocentric distance.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "heliocentric_body_risetime".

    Returns
    -------
    None.

    """
    
    filtered_body_rise = data["body_rise_times"][
                                (data["body_rise_times"]>0)*
                                (data["body_rise_times"]<5e-5)]  
    filtered_helio_r = data["heliocentric_distances"][
                                    (data["body_rise_times"]>0)*
                                    (data["body_rise_times"]<5e-5)]
    segments = np.arange(0.5,1.01,0.1)
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(filtered_body_rise[
            (segments[i]<filtered_helio_r)*
            (filtered_helio_r<segments[i+1])])
        medians[i] = np.median(filtered_body_rise[
            (segments[i]<filtered_helio_r)*
            (filtered_helio_r<segments[i+1])])
    segments_mids = (segments[1:]+segments[:-1])/2
    fig,ax=plt.subplots()
    ax.scatter(filtered_helio_r,filtered_body_rise*1e6,
               alpha=0.2,edgecolor="none",color=u"#3BB273")
    ax.hlines(means*1e6,segments[:-1],segments[1:],label="Mean",color="tomato",
              ls="solid",lw=0.7)
    ax.hlines(medians*1e6,segments[:-1],segments[1:],label="Median",color="black",
              ls="solid",lw=0.7) 
    ax.plot(segments_mids,(8.1/1.6)**(0.333)*12*segments_mids**0.666*(20/10)**(-2/3),
            color="blue",ls="dashed",label="Meyer-Vernet et al. 2017 - sunlit")
    ax.plot(segments_mids,(8.1/1.6)**(0.333)*27*segments_mids**0.666*(20/10)**(-2/3),
            color="grey",ls="dashed",label="Meyer-Vernet et al. 2017 - shade")
    ax.set_xlim(0.45,1.05)
    ax.set_ylim(0,50)
    ax.set_xlabel(r"Heliocentric distance [$AU$]")
    ax.set_ylabel(r"Primary peak rise time [$\mu s$]")
    ax.legend(fontsize="small",loc=2)
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')


def make_plot_charge_body_risetime(data,
                              target_dir,
                              name="charge_body_risetime"):
    """
    The function to plot the primary (body) peak risetime as 
    a function of primary (body) peak amplitude. 

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "charge_body_risetime".

    Returns
    -------
    None.

    """
    
    filtered_body_rise = data["body_rise_times"][
                                (data["body_rise_times"]>0)*
                                (data["body_rise_times"]<5e-5)]  
    filtered_charge = data["body_amplitudes"][
                                    (data["body_rise_times"]>0)*
                                    (data["body_rise_times"]<5e-5)]
    logbins =np.arange(-3,-0.8,0.2)
    bins_mids = 10**((logbins[1:]+logbins[:-1])/2)
    segments = 10**logbins
    means = np.zeros(len(segments)-1)
    medians = np.zeros(len(segments)-1)
    for i in range(len(means)):
        means[i] = np.mean(filtered_body_rise[
            (segments[i]<filtered_charge)*
            (filtered_charge<segments[i+1])])
        medians[i] = np.median(filtered_body_rise[
            (segments[i]<filtered_charge)*
            (filtered_charge<segments[i+1])])
    fig,ax=plt.subplots()
    ax.scatter(filtered_charge,filtered_body_rise*1e6,
               alpha=0.2,edgecolor="none",color=u"#3BB273")
    ax.hlines(means*1e6,segments[:-1],segments[1:],label="Mean",color="tomato",
              ls="solid",lw=0.7)
    ax.hlines(medians*1e6,segments[:-1],segments[1:],label="Median",color="black",
              ls="solid",lw=0.7) 
    ax.plot(bins_mids,(1000*bins_mids/1.6)**(0.333)*12*(0.75)**0.666*(20/10)**(-2/3),
            ls="dashed",color="blue", label="Meyer-Vernet et al. 2017 - sunlit")
    ax.plot(bins_mids,(1000*bins_mids/1.6)**(0.333)*27*(0.75)**0.666*(20/10)**(-2/3),
            ls="dashed",color="grey", label="Meyer-Vernet et al. 2017 - shade")
    ax.set_xlim(5e-4,2e-1)
    ax.set_ylim(0,50)
    #ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"Primary peak amplitude [$V$]")
    ax.set_ylabel(r"Primary peak rise time [$\mu s$]")
    ax.legend(fontsize="small",loc=2)
    fig.tight_layout()
    plt.savefig(target_dir+name+"_simple"+".pdf", format='pdf')


def make_plot_heliocentric_charge(data,
                                  target_dir,
                                  name="heliocentric_charge_correlation"):
    """
    The function to plot the charge released 
    as a function of heliocentric distance.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "heliocentric_charge_correlation".

    Returns
    -------
    None.

    """
    
    
    filtered_helio_r = data["heliocentric_distances"][
                                    (data["body_rise_times"]>0)*
                                    (data["body_rise_times"]<5e-5)]
    filtered_charge = data["body_amplitudes"][
                                    (data["body_rise_times"]>0)*
                                    (data["body_rise_times"]<5e-5)]

    fig,ax=plt.subplots()
    ax.scatter(filtered_helio_r,filtered_charge,
               alpha=0.2,edgecolor="none",color="olivedrab")
    
    ax.set_xlim(0.45,1.05)
    ax.set_ylim(1e-4,1e0)
    
    ax.set_yscale("log")
    
    ax.set_xlabel(r"Heliocentric distance [$AU$]")
    ax.set_ylabel(r"Primary peak amplitude [$V$]")
   
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_plot_hist_asymmetry(data,
                             target_dir,
                             name="hist_asymmetry"):
    """
    The asymmetry in the body peak due to a charge remote pickup.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "hist_asymmetry".

    Returns
    -------
    None.

    """
    
    number_of_secondary_present = (data["sec_present_v1"]+
                                   data["sec_present_v2"]+
                                   data["sec_present_v3"])
    mask_no_secondary = number_of_secondary_present == 0
    
    prespike_correction = np.nan_to_num(data["negative_prespikes"])
    
    v1_over_v2 = (data["stupid_max_v1"]-prespike_correction
                  ) / (data["stupid_max_v2"]-prespike_correction)
    v2_over_v3 = (data["stupid_max_v2"]-prespike_correction
                  ) / (data["stupid_max_v3"]-prespike_correction)
    v3_over_v1 = (data["stupid_max_v3"]-prespike_correction
                  ) / (data["stupid_max_v1"]-prespike_correction)
    
    v1_over_v2_primary_only = v1_over_v2[mask_no_secondary]
    v2_over_v3_primary_only = v2_over_v3[mask_no_secondary]
    v3_over_v1_primary_only = v3_over_v1[mask_no_secondary]
    
    bins = np.arange(0,2.2,0.1)
    
    fig,ax = plt.subplots()
    ax.hist(v1_over_v2_primary_only,bins=bins,
            label="V1 / V2",alpha=1,density=True,histtype="step",
            ls=(0,(1,0.7)),lw=0.7)
    ax.hist(v2_over_v3_primary_only,bins=bins,
            label="V2 / V3",alpha=1,density=True,histtype="step",
            ls=(0,(6,2)),lw=0.7)
    ax.hist(v3_over_v1_primary_only,bins=bins,
            label="V3 / V1",alpha=1,density=True,histtype="step",
            ls=(0,(5,1,1,1)),lw=0.7)
    ax.set_xlabel("Ratio [1]")
    ax.set_ylabel("Density [1]")
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.savefig(target_dir+name+".pdf", format='pdf')
    
    print(data["ids"][mask_no_secondary][
        v1_over_v2_primary_only>2].astype('int64'))


def make_plot_risetime_monopole_dipole(data,
                                       target_dir,
                                       name="monopole_dipole_risetime"):
    """
    Make a plot to compare monopole and dipole risetimes.

    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.
    name : str, optional
        Output png name. The default is "monopole_dipole_risetime.png".

    Returns
    -------
    None.

    """
    
    mask = (data["sec_present_v1"]
             +data["sec_present_v2"]
             +data["sec_present_v3"]==0)
   
    fig = plt.figure()
    gs = fig.add_gridspec(3, 1, hspace=0.1)
    (ax1,ax2,ax3) = gs.subplots(sharex=True)
    bins = np.arange(0,100,20)
    ax1.hist(data["xld_risetime_1"][mask]*1e6,bins=bins,density=0,
             lw=1,color=u"#44355B", alpha=0.3)
    ax2.hist(data["xld_risetime_2"][mask]*1e6,bins=bins,density=0,
             lw=1,color=u"#44355B", alpha=0.3)
    ax3.hist(data["xld_risetime_3"][mask]*1e6,bins=bins,density=0,
             lw=1,color=u"#44355B", alpha=0.3)
    ax3.set_xlabel(r"channel risetime [$\mu s$]")
    ax1.set_ylabel("wf1")
    ax2.set_ylabel("wf2")
    ax3.set_ylabel("wf3 \n (monopole)")
    fig.tight_layout()
    plt.savefig(target_dir+name+".png", format='png', dpi=300)


def make_all_plots(data,
                   target_dir):
    """
    A container for all the plotting routines to be done.
    
    Parameters
    ----------
    data : pd.dataframe
        All the data to base the plots on.
    target_dir : str
        An absolute path wher to put the resulting plots.

    Returns
    -------
    None.

    """
    make_plot_hist_prespike_potential(data[data["alright"]==True],
                                 target_dir)
    
    make_plot_hist_heliocentric_nano(data[data["alright"]==True],
                                     target_dir)
    
    make_plot_ternary(data[data["saturated"]==False],
                      target_dir,colorcode="steelblue")
    
    make_plot_ternary_heatmap(data[data["saturated"]==False],
                      target_dir,
                      nbins = 50,
                      datapointcount=False) 
    
    make_plot_ternary(data[(data["saturated"]==False) * (
                            data["heliocentric_distances"]<0.7)],
                      target_dir,
                      note = r"$r<0.7AU$",
                      name="ternary_inner")
    
    make_plot_ternary(data[(data["saturated"]==False) * (
                            data["heliocentric_distances"]>0.9)],
                      target_dir,
                      note = r"$r>0.9AU$",
                      name="ternary_outer")
    
    make_plot_ternary(data[data["saturated"]==False],
                      target_dir,
                      colorcode=np.log10(data[
                          data["saturated"]==False
                          ]["body_amplitudes"]),
                      colorscale="body amplitudes [log10(V)]",
                      name="ternary_amp.png")
    
    make_plot_ternary(data[data["alright"]==True],
                      target_dir,
                      colorcode=make_plot_assignment_jakub(
                          data[data["alright"]==True],
                          target_dir,
                          hand_colors=True),
                      name="ternary_jakub_colors.png")
        
    make_plot_ternary_heatmap(data[(data["saturated"]==False) & (
                                    (data["sec_present_v1"]>0) |
                                    (data["sec_present_v2"]>0) |
                                    (data["sec_present_v3"]>0) )],
                      target_dir,
                      nbins = 40,
                      name="ternary_secondary_present")
    
    make_plot_ternary_heatmap(data[(data["saturated"]==False) & (
                                    (data["sec_present_v1"]<1e-3) &
                                    (data["sec_present_v2"]<1e-3) &
                                    (data["sec_present_v3"]<1e-3) )],
                      target_dir,
                      nbins = 40,
                      colorbarticks = [0,2,4,6,8,10,12,14,16],
                      name="ternary_secondary_not_present")
    
    make_plot_assignment_jakub(data[data["alright"]==True],
                    target_dir)
    
    make_plot_scpot_prespike_amplitude(data[data["alright"]==True],
                                 target_dir)
    
    make_plot_scpot_overshoot(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_primary_overshoot(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_stupid_max_overshoot(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_secondary_overshoot(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_overshoot_amplitude_delay(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_secondary_amplitude_delay(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_secondary_overshoot_delay(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_primary_secondary_amplitude(data[data["alright"]==True],
                                                 target_dir)

    make_plot_antenna_ampltiude_delay(data[data["alright"]==True],
                              target_dir)
    
    make_plot_heliocentric_antenna_delay(data[data["alright"]==True],
                                         target_dir)
    
    make_plot_vuv_antenna_delay(data[data["alright"]==True],
                                  target_dir)
    
    make_plot_hist_delay(data[data["alright"]==True],
                              target_dir)
    
    make_plot_antenna_ampltiude_delay(data[data["alright"]==True],
                              target_dir,
                              showhist=False,
                              name="antenna_amplitude_delay_nohist")
    
    make_plot_heliocentric_antenna_delay(data[data["alright"]==True],
                                         target_dir,
                                         showhist=False,
                                         name="heliocentric_delay_nohist")
    
    make_plot_vuv_antenna_delay(data[data["alright"]==True],
                                  target_dir,
                                  showhist=False,
                                  name="vuv_delay_nohist")

    make_plot_hist_body_amplitude(data[data["alright"]==True],
                              target_dir)
    
    make_plot_hist_body_amplitude(data[(data["alright"]==True) & (
                                    (data["sec_present_v1"]<1e-5) &
                                    (data["sec_present_v2"]<1e-5) &
                                    (data["sec_present_v3"]<1e-5) )],
                              target_dir,
                              name="hist_body_amplitude_pure_triple")

    make_plot_hist_antenna_amplitude(data[data["alright"]==True],
                              target_dir)

    make_plot_hist_body_both(data[data["alright"]==True],
                              target_dir)

    make_plot_heliocentric_body_decay(data[data["alright"]==True],
                              target_dir)

    make_plot_heliocentric_body_risetime(data[data["alright"]==True],
                              target_dir)
    
    make_plot_charge_body_risetime(data[data["alright"]==True],
                              target_dir)
    
    make_plot_heliocentric_charge(data[data["alright"]==True],
                              target_dir)

    make_plot_secondary_max_min(data[data["alright"]==True],
                                False,
                                target_dir,
                                threshold = 20)
    
    make_plot_secondary_max_min_root(data[data["alright"]==True],
                                         target_dir)
    
    make_plot_secondary_max_min_vuv(data[data["alright"]==True],
                                target_dir)
    
    make_plot_hist_asymmetry(data[data["alright"]==True],
                              target_dir)
    
    make_plot_risetime_monopole_dipole(data[data["alright"]==True],
                              target_dir)
    

def main(source_dir,target_dir,hand_data=False):
    """
    The main function, first loads the data, then makes the plots. 
    Optionally, may output the dataframe it used to produce the plots.

    Parameters
    ----------
    source_dir : str
        Absolute path to get the datafiles from.
    target_dir : str
        Absolute path where to put the generated png files.
    hand_data : bool, optional
        Whether to return the dataframe used. The default is False.

    Returns
    -------
    data : pd.dataframe
        The dataframe used to generate the plots.

    """
    files_to_analyze = npz_files_to_analyze(source_dir)
    
    data = create_df(arrays_to_define)
    
    for f in files_to_analyze:
        data = pd.concat([data,load_npz_into_df(f)])
        
    if hand_data:
        return data
    else:
        make_all_plots(data,target_dir)
    
    
#feed source data and target plot directory  
main("/997_data/solo_features/",
     "/998_generated/solo_statistics/") 
    

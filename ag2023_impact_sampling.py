import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600
import commons_figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')


def plot_solo(ants, shield, shield_points=[]):
    #plot how SolO looks
    
    ant1 = ants[0]
    ant2 = ants[1]
    ant3 = ants[2]
    
    fig, ax = plt.subplots()
    ax.plot(ant1[0],ant1[1],c="red",ls="dashed") 
    ax.plot(ant2[0],ant2[1],c="red",ls="dashed") 
    ax.plot(ant3[0],ant3[1],c="red",ls="dashed")
    ax.plot([shield[1],shield[1],shield[0],shield[0],shield[1]],[shield[2],shield[3],shield[3],shield[2],shield[2]],c="black") 
    if len(shield_points)>0:
        if len(shield_points)>1000:
            points = np.random.choice(np.arange(len(shield_points)),150)
            plt.scatter(shield_points[points,0],shield_points[points,1],s=0.1)
        else:
            plt.scatter(shield_points[:,0],shield_points[:,1],s=0.1)
    ax.text(1,9,"1",c="red")
    ax.text(-7,-4,"2",c="red")
    ax.text(7,-4,"3",c="red")
    ax.set_xlim(-9,9)
    ax.set_ylim(-7,11)
    ax.set_aspect(1)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.text(-7.5,6.5,r"$\otimes$ to Sun")
    ax.text(-7.5,8.5,r"$\uparrow$ HAE Z")
    ax.text(3.65,0.5,"ram",fontsize="small")
    ax.annotate("", xy=(8, 0), xytext=(2, 0),arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    fig.savefig("/998_generated/solo_diagram.pdf", format='pdf')
    fig.show() 
    

#define the function for a potential from a single point charge
def phi(loc,loc_q,debye=7):
    #function to return potential in point x,y,z caused by a single charge Q
    #loc is 2D, 3*n element array [x,y,z] - point on the antenna
    #loc_q is 2D, 3*n element array of the charge position
    #debye is the debye length
    #everything is in meters
    
    eps0 = 8.85e-12 #permittivity, SI
    
    if len(np.shape(loc)) != len(np.shape(loc_q)):
        raise Exception("array dimensions not matching")
        
    if len(np.shape(loc)) == 2:  
        x = loc[0,:]
        y = loc[1,:]
        z = loc[2,:]
        xq = loc_q[0,:]
        yq = loc_q[1,:]
        zq = loc_q[2,:]      
        if len(x)!=len(xq):
            raise Exception("array sizes not matching")
        
    if len(np.shape(loc)) == 1:
        x = loc[0]
        y = loc[1]
        z = loc[2]
        xq = loc_q[0]
        yq = loc_q[1]
        zq = loc_q[2]
    
    #distance from impact to the point on the antena    
    R = np.sqrt((x-xq)**2+(y-yq)**2+(z-zq)**2) 
    return (1/(4*np.pi*eps0))*(1/R)*np.exp(-R/debye) #Volt


def elastance(ant_start,ant_end,loc_q,sampling=500,idebye=7):
    #ant_start and ant_end are 1D, 3-position vector of antenna start and end
    #loc_q is the location of a single free charge
    #sampling is in how many points the function will be averaged
    #everything is in meters
    #voltage is elastance * charge
        
    loc_q_tiled = np.transpose(np.tile(np.array(loc_q),(sampling,1)))
    
    loc_tiled = np.zeros((3,sampling))
    for i in range(3):
        span = ant_end[i]-ant_start[i]
        for j in range(sampling):    
            loc_tiled[i,j] = ant_start[i] + (span/sampling)*(j+0.5)
    
    return np.average(phi(loc_tiled,loc_q_tiled,debye=idebye)) #Volt


def shield_grid(shield,shield_plane,density=10):
    #make grid of impacts onto the shield
    #shield is [-1.2,1.2]*[-1.15,1.95], whih is a rather rough approx
    shield_sampling = (int(np.round((shield[1]-shield[0])*density)),
                       int(np.round((shield[3]-shield[2])*density)))
    shield_points = np.zeros(((shield_sampling[0])*(shield_sampling[1]),3))
    for i in range(shield_sampling[0]):
        for j in range(shield_sampling[1]):
           shield_points[i*shield_sampling[1]+j,0] = shield[0] + ((np.random.random()+i+0.5)/shield_sampling[0])*(shield[1]-shield[0])
           shield_points[i*shield_sampling[1]+j,1] = shield[2] + ((np.random.random()+j+0.5)/shield_sampling[1])*(shield[3]-shield[2])
           shield_points[i*shield_sampling[1]+j,2] = shield_plane #just assumption that the impact happens on the shield
   
    return shield_points


def responses(shield_points,
              ants,
              test_charge,
              sc_sensitivity,
              debye_length,
              sampling):
    #now make 3 arrays that represent voltages left on 3 antennas
    # and one more for the body response
    v1 = np.zeros(len(shield_points))
    v2 = np.zeros(len(shield_points))
    v3 = np.zeros(len(shield_points))
    body = np.zeros(len(shield_points))
    
    ant1 = ants[0]
    ant2 = ants[1]
    ant3 = ants[2]
    
    for i in range(len(v1)):
        v1[i] = test_charge*(elastance([ant1[0][0],ant1[1][0],0],
                                   [ant1[0][1],ant1[1][1],0],
                                   shield_points[i],idebye=debye_length,
                                   sampling=sampling))
        v2[i] = test_charge*(elastance([ant2[0][0],ant2[1][0],0],
                                   [ant2[0][1],ant2[1][1],0],
                                   shield_points[i],idebye=debye_length,
                                   sampling=sampling))
        v3[i] = test_charge*(elastance([ant3[0][0],ant3[1][0],0],
                                   [ant3[0][1],ant3[1][1],0],
                                   shield_points[i],idebye=debye_length,
                                   sampling=sampling))
        body[i] = test_charge*sc_sensitivity
        
    return np.vstack((body,v1,v2,v3))
    
    
def main():
    #xmin, xmax, ymin, ymax
    shield = [-1.2,1.2,-1.15,1.95]

    #dimensions
    x1,y1 = 0       , 2.8371 #antenna 1  [m], positive x is prograde
    x2,y2 = -1.8823 , -1.4185 #antenna 2  [m], positive y is gocentric north
    x3,y3 = 1.8823  , -1.4185 #antenna 3  [m]
    l = 6.5 #length of antenna past heat shield

    #assumptions
    test_charge = 1 #C
    charge_distance = 1 #in front of the antenna plane, [m]
    sc_sensitivity = 1e9 #V/C, equation: Gamma/C_sc
    debye_length = 5
    antenna_points = 100
    shield_points_density = 30

    #xmin, xmax, ymin, ymax
    ant1 = [[x1,x1+l*x1/np.sqrt(x1**2+y1**2)],[y1,y1+l*y1/np.sqrt(x1**2+y1**2)]]
    ant2 = [[x2,x2+l*x2/np.sqrt(x2**2+y2**2)],[y2,y2+l*y2/np.sqrt(x2**2+y2**2)]]
    ant3 = [[x3,x3+l*x3/np.sqrt(x3**2+y3**2)],[y3,y3+l*y3/np.sqrt(x3**2+y3**2)]]
    ants = [ant1,ant2,ant3]
    
    shield_points = shield_grid(shield,
                                charge_distance,
                                density=shield_points_density)   
    plot_solo(ants,
              shield,
              shield_points=shield_points)
    
    amplitudes = responses(shield_points,
                           ants,
                           test_charge,
                           sc_sensitivity,
                           debye_length,
                           sampling=antenna_points)
    
    bins = np.arange(0,2,0.02)
    
    fig,ax=plt.subplots()
    plt.hist((amplitudes[0]+amplitudes[1])/(amplitudes[0]+amplitudes[2]),
             bins=bins,
             label="V1 / V2",
             alpha=1,density=True,histtype="step",ls=(0,(1,0.7)),lw=0.7)
    plt.hist((amplitudes[0]+amplitudes[2])/(amplitudes[0]+amplitudes[3]),
             bins=bins,label="V2 / V3",
             alpha=1,density=True,histtype="step",ls=(0,(6,2)),lw=0.7)
    plt.hist((amplitudes[0]+amplitudes[3])/(amplitudes[0]+amplitudes[1]),
             bins=bins,label="V3 / V1",
             alpha=1,density=True,histtype="step",ls=(0,(5,1,1,1)),lw=0.7)
    plt.hist((amplitudes[1])/(amplitudes[0]),
             bins=bins,label=r"$\Phi_{ant1} / \Phi_{body}$",
             alpha=1,density=True,histtype="step",color="black",lw=0.7)
    ax.legend(fontsize="small")
    ax.set_xlabel("Ratio [1]")
    ax.set_ylabel("Density [1]")
    fig.tight_layout()
    fig.savefig("/998_generated/solo_asymmetry_hist.pdf", format='pdf')
    fig.show() 
    
main()
    
    
    
    
    
    
    
    
import sys
from sunpy.net import Fido
import sunpy.net.attrs as a
import glob
from commons_conversions import date2jd
from astropy.io import fits
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


def download(dt_from,
             dt_to,
             data_instrument,
             data_level,
             data_product,
             folder):
    """
    A fetching function for spice data from SOAR.

    Parameters
    ----------
    dt_from : datetime.datetime()
        Starting date of the selection.
    dt_to : datetime.datetime()
        The last downloaded day.
    data_instrument : str
        Such as "EUI" of "SPICE".
    data_level : int
        0 : raw, 1 : engineering, 2 : scientific, 3 : higher
    data_product : str
        Such as "EUI-FSI304-IMAGE".
    folder : str
        Where to place the downloaded files.

    Returns
    -------
    files : parfive.Results
        The download report.

    """
    
    time_from = str(dt_from.date())
    time_to = str(dt_to.date())
    
    instrument = a.Instrument(data_instrument)
    time = a.Time(time_from, time_to)
    level = a.Level(data_level)
    product = a.soar.Product(data_product)
    
    result = Fido.search( instrument & time & level & product )
    files = Fido.fetch(result,path=folder)
    return files


def read_fits_datamean(filepath):
    """
    Reads the given EUI FSI L2 FITS file for the mean value 
    and the time of the capture.

    Parameters
    ----------
    filepath : str
        The FITS file to read.

    Returns
    -------
    time : datetime.datetime()
        The time of capture.
    mean : float
        The mean value of the image.
    heliocentric_distance : float
        tbd
    solar_disc_radius : float
        in arcsec
    xposure : float
        exposure in sec
    """
    
    file = fits.open(filepath)
    
    #mean = file[1].header["DATAMEAN"]
    #mean = np.mean(file[1].data[200:-200,200:-200])
    
    pixel = file[1].header["CDELT1"]*file[1].header["CDELT2"]
    pixel_count = ((np.shape(file[1].data[:,:])[0]-400)*
           (np.shape(file[1].data[:,:])[1]-400))
    
    mean = (np.sum(file[1].data[200:-200,200:-200])/pixel_count)*pixel
    #actually in something like "photon / ( time * arcsec**2 )"
    
    yyyy = int(file[1].header["DATE-BEG"][:4])
    mm = int(file[1].header["DATE-BEG"][5:7])
    dd = int(file[1].header["DATE-BEG"][8:10])
    h = int(file[1].header["DATE-BEG"][11:13])
    mi = int(file[1].header["DATE-BEG"][14:16])
    s = int(file[1].header["DATE-BEG"][17:19])
    ms = int(file[1].header["DATE-BEG"][20:23])
    time = dt.datetime(yyyy,mm,dd,h,mi,s,ms*1000)
    
    heliocentric_distance = file[1].header["DSUN_AU"]
    
    solar_disc_radius = file[1].header["RSUN_ARC"]
    
    xposure = file[1].header["XPOSURE"]
    
    nbin = file[1].header["NBIN"]
    
    filcpos = file[1].header["FILCPOS"]
    
    cdelt1 = file[1].header["CDELT1"]
    
    return time,mean,heliocentric_distance,solar_disc_radius,xposure,nbin,filcpos,cdelt1


def get_file_list(datapath = "/997_data/solo_eui/fsi304/"):
    """
    Make a list of fits files that are donwloaded.
    
    Parameters
    ----------
    datapath : str
        Directory in which to look for the fits files.

    Returns
    -------
    fitsfiles : numpy.ndarray(1,:) of str
        An array of fits files, absolute paths.

    """
    fitsfiles = glob.glob(datapath+"*.fits")
    return fitsfiles


def quality_criteria(filepath,dt_from,dt_to):
    """
    The function to check whether a given fits file with EUI data 
    meets the minimum quality requirements.

    Parameters
    ----------
    filepath : str
        absolut path of a file to be checked for quality.
    dt_from : datetime.datetime
        Lowest accepted date.
    dt_to : datetime.datetime
        highest accepted date.

    Returns
    -------
    met : bool
        File passed quality check.
    """
    
    year = int(filepath[-27:-14][:4])
    month = int(filepath[-27:-14][4:6])
    day = int(filepath[-27:-14][6:8])
    hour = int(filepath[-27:-14][9:11])
    minute = int(filepath[-27:-14][11:13])
    time = dt.datetime(year,month,day,hour,minute,0,0)
    if time<dt_from or time>dt_to:
        return False   
    
    file = fits.open(filepath)
    
    if len(file) != 2:
        return False    
    
    else:
    
        yyyy = int(file[1].header["DATE-BEG"][:4])
        mm = int(file[1].header["DATE-BEG"][5:7])
        dd = int(file[1].header["DATE-BEG"][8:10])
        h = int(file[1].header["DATE-BEG"][11:13])
        mi = int(file[1].header["DATE-BEG"][14:16])
        s = int(file[1].header["DATE-BEG"][17:19])
        ms = int(file[1].header["DATE-BEG"][20:23])
        time = dt.datetime(yyyy,mm,dd,h,mi,s,ms*1000)
        
        met = 1
        
        #if file[1].header["DATAMEAN"]>1000:
        #    met = met*0
        
        if time<dt_from or time>dt_to:
            met = met*0
            
        if np.abs(file[1].header["WAVELNTH"]-304)>1:
            met = met*0    
            
        #if np.abs(file[1].header["EUXCEN"]-750)>200:
        #    met = met*0 
            
        #if np.abs(file[1].header["EUYCEN"]-750)>200:
        #    met = met*0 
        
        #if np.abs(file[1].header["XPOSURE"]-10)>0.2:
        #    met = met*0    
            
        #if np.abs(file[1].header["FILCPOS"]-1)>0.2 and np.abs(file[1].header["FILCPOS"]-3)>0.2:
        #    met = met*0
            
        #if np.abs(file[1].header["NBIN"]-1)>0.2:
        #    met = met*0
    
    
        return bool(met)


def make_index(dt_from,
               dt_to,
               datapath = "/997_data/solo_eui/fsi304/",
               targetpath = "/997_data/solo_eui/indexed/",
               chunks = 1,
               section = 0):
    """
    Take the downloaded data and make an index npz out of them.

    Parameters
    ----------
    dt_from : datetime.datetime
        minimum acceptable datetime.
    dt_to : datetime.datetime
        maximum acceptable datetime.
    datapath : str, optional
        The absolute path where to look for available fits files. 
        Default is "/997_data/solo_eui/fsi304/"
    targetpath : str, optional
        The absolute path where to save the indexed information. 
        Default is "/997_data/solo_eui/indexed/"  
    chunks : int, optional
        The number of sections to break the analysis into. The default is 1.
    section : int, optional
        Number of chunk to analyze. The default is 1.

    Returns
    -------
    filecount : int
        number of converted files.

    """
    filecount = 0
    exceptions = 0
    fitsfiles = get_file_list(datapath)
    
    #chunking
    per_chunk = len(fitsfiles)//chunks + 1
    fitsfiles = fitsfiles[(section)*per_chunk:(section+1)*per_chunk]
    
    for filepath in fitsfiles:
        filename_short = filepath[filepath.find("\\solo_L2_eui")+1:-5]
        try:
            if quality_criteria(filepath,dt_from,dt_to):
                times = []
                means = []
                helio_rs = []
                disc_rs = []
                xposures = []
                nbins = []
                filcposs = []
                cdelt1s = []
                
                time,mean,helio_r,disc_r,xposure,nbin,filcpos,cdelt1 = read_fits_datamean(filepath)
                
                times.append(date2jd(time))
                means.append(mean)
                helio_rs.append(helio_r)
                disc_rs.append(disc_r)
                xposures.append(xposure)
                nbins.append(nbin)
                filcposs.append(filcpos)
                cdelt1s.append(cdelt1)
                
                target_output_npz = filename_short+".npz"     
                np.savez(targetpath+target_output_npz,
                         times = np.array(times),
                         means = np.array(means),
                         helio_rs = np.array(helio_rs),
                         disc_rs = np.array(disc_rs),
                         xposures = np.array(xposures),
                         nbins = np.array(nbins),
                         filcposs = np.array(filcposs),
                         cdelt1s = np.array(cdelt1s))
                print("saved index to "+target_output_npz)
                filecount += 1
        except:
            print("exception @ "+filepath)
            exceptions += 1
            
    
    return filecount, exceptions


def main(dt_from,
         dt_to,
         access_online=False,
         datapath = "/997_data/solo_eui/fsi304/",
         chunks = 1,
         section = 0):
    """
    The main function to load the data and to produce the npz index.

    Parameters
    ----------
    dt_from : datetime.datetime()
        Lowest accepted date.
    dt_to : datetime.datetime()
        Highest accepted date.
    access_online : bool, optional
        Whether to acces SOAR. If False, only local files will be used. 
        The default is False.
    datapath : str, optional
        The absolute folder of the fits data.
    chunks : int, optional
        The number of sections to break the analysis into. The default is 1.
    section : int, optional
        Number of chunk to analyze. The default is 1.

    Returns
    -------
    None.

    """
    
    if access_online:
        files = download(dt_from,
                 dt_to,
                 "EUI",
                 2,
                 "EUI-FSI304-IMAGE",
                 datapath)

    filecount, exceptions = make_index(dt_from,
                                       dt_to,
                                       datapath=datapath,
                                       chunks=chunks,
                                       section=section)
    
    print("files saved :"+str(filecount))
    print("exceptions :"+str(exceptions))
  
    
def main_multi(source,target):
    """
    The function to process the .fits files one by one and save 
    them as .npz pre-processed, much smaller files.

    Parameters
    ----------
    source : str
        Sorce file (.fits).
    target : str
        Target file (.npz).

    Returns
    -------
    None.

    """
    quality = quality_criteria(source,
                               dt.datetime(2000,1,1,0,0,0,1),
                               dt.datetime(2100,1,1,0,0,0,1))
    
    if quality:
        time,mean,helio_r,disc_r,xposure,nbin,filcpos,cdelt1 = read_fits_datamean(source)
        np.savez(target,
                 times = np.array(date2jd(time)),
                 means = np.array(mean),
                 helio_rs = np.array(helio_r),
                 disc_rs = np.array(disc_r),
                 xposures = np.array(xposure),
                 nbins = np.array(nbin),
                 filcposs = np.array(filcpos),
                 cdelt1s = np.array(cdelt1))
    else:
        np.savez(target,
                 times = np.zeros(0),
                 means = np.zeros(0),
                 helio_rs = np.zeros(0),
                 disc_rs = np.zeros(0),
                 xposures = np.zeros(0),
                 nbins = np.zeros(0),
                 filcposs = np.zeros(0),
                 cdelt1s = np.zeros(0))
    
    

#single-thread analysis
"""  
main(dt_from = dt.datetime(2021,3,15,0,0,0,1),
     dt_to = dt.datetime(2024,3,20,23,59,59,999),
     access_online=False) 
"""    


main(dt_from = dt.datetime(2020,3,15,0,0,0,1),
     dt_to = dt.datetime(2024,3,14,23,59,59,999),
     access_online=False,
     chunks=int(sys.argv[1]),
     section=int(sys.argv[2])) 


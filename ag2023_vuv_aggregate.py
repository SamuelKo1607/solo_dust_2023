import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


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


def load_npz_into_df(target_npz,arrays_to_define):
    """
    Creates a single dataframe out of npz file
    
    Parameters
    ----------
    target_npz : str
        An absolute path to the npz that is to be unpacked and 
        made into dataframe.
        
    arrays_to_define : list of str
        Names of the columns used.

    Returns
    -------
    df : pd.dataframe
        Dataframe made of contents of target_npz .npz file.
        
    """
    npz_content = np.load(target_npz,allow_pickle=True)
    
    #the dictionary to translate from npz column names to local keys
    
    times = npz_content['times']
    means = npz_content['means']
    helio_rs = npz_content['helio_rs']
    disc_rs = npz_content['disc_rs']
    xposures = npz_content['xposures']
    nbins = npz_content['nbins']
    filcposs = npz_content['filcposs']
    cdelt1s = npz_content['cdelt1s']
        
    #shape them into one pd.df
    df = pd.DataFrame(data=np.vstack((target_npz,
                                      times,
                                      means,
                                      helio_rs,
                                      disc_rs,
                                      xposures,
                                      nbins,
                                      filcposs,
                                      cdelt1s
                                      )).transpose(),
                      columns=arrays_to_define)
    return df


def main(source_dir,target_dir):
    """
    A wrapper function to read all the .npz files and make one huge .csv.

    Parameters
    ----------
    source_dir : str
        The directory, in which .npz files are sought.
    target_dir : str
        The directory to put the aggregated .csv file. 

    Returns
    -------
    None.

    """
    
    files_to_analyze = glob.glob(source_dir+"*.npz")
    list_of_variables = ['files',
                         'times',
                         'means',
                         'helio_rs',
                         'disc_rs',
                         'xposures',
                         'nbins',
                         'filcposs',
                         'cdelt1s']
    data = create_df(list_of_variables)
    
    for f in files_to_analyze:
        data = pd.concat([data,load_npz_into_df(f,list_of_variables)])
        
    data.to_csv(target_dir+"eui_stats.csv")  
    
    
    
main("/997_data/solo_eui/indexed/",
     "/997_data/solo_eui/")
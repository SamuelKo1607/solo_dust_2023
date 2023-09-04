import cdflib
import numpy as np
import glob
from commons_conversions import jd_to_tt2000
from commons_conversions import jd2date

path = "/997_data/solo_rpw/bia_current"
bia_current_list = glob.glob(path+"/solo_L1_rpw-bia-current*.cdf")
start_YYYYMMDD = []

for file in bia_current_list:
    start_YYYYMMDD.append(file[file.index("_202")+1:file.index("_202")+9])

def find_bias_cdf(jd):
    #finds the right CDF file
    date = jd2date(jd)
    year = date.year
    month = date.month
    match = []
    
    for file in bia_current_list:
        fyear = int(file[file.index("_202")+1:file.index("_202")+5])
        fmonth = int(file[file.index("_202")+5:file.index("_202")+7])
        if year == fyear and month == fmonth:
            match.append(file)
    if len(match)==0:
        raise Exception("no match for "+str(date))
    elif len(match)>1:
        raise Exception("multiple files for "+str(date)+" : \n "+match)
    else: 
        return match[0]
    
    
def bias(jd,give_epoch=False): #test 2459411.78125
    #returns the bias current fed into the three antennas ata given julian date
    #optionally, returns the epoch of the last current change
        
    #load the file for the right month
    bias_file = cdflib.CDF(find_bias_cdf(jd))
    epoch = bias_file.varget("epoch") #ns
    #sometimes the required time predates the first entry of a month, in that case, load the preceding month
    if len(epoch[epoch<jd_to_tt2000(jd)])==0:
        bias_file = cdflib.CDF(find_bias_cdf(jd-27)) #to get the file of the preceding month
        epoch = bias_file.varget("epoch") #ns
    
    bias_1 = bias_file.varget("IBIAS_1")*1e-9 #nA -> A
    bias_2 = bias_file.varget("IBIAS_2")*1e-9 #nA -> A
    bias_3 = bias_file.varget("IBIAS_3")*1e-9 #nA -> A
    
    last_epoch = np.max(epoch[epoch<jd_to_tt2000(jd)])
    near_epochs_indices = np.arange(len(epoch))[np.abs(epoch-last_epoch)<1e10]
    
    b1 = bias_1[near_epochs_indices][np.abs(bias_1[near_epochs_indices])<1e9][0] #<1A
    b2 = bias_2[near_epochs_indices][np.abs(bias_2[near_epochs_indices])<1e9][0] #<1A
    b3 = bias_3[near_epochs_indices][np.abs(bias_3[near_epochs_indices])<1e9][0] #<1A
    
    if give_epoch:
        return np.array([b1, b2, b3]), last_epoch
    else:
        return np.array([b1, b2, b3])
        
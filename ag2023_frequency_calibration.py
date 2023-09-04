import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600
import commons_figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
from commons_solo_download import fetch
from scipy.signal import welch
mpl.rcParams['text.usetex'] = False

#correct the time-series for a high-pass filter of X Hz through Laplace domain as Amalia did
def hipass_correction(signal,epochstep,cutoff=200):
    #cutoff in Hz
    #epochstep in s
    cumsignal = np.cumsum(signal)
    return signal + (2*np.pi*cutoff)*(cumsignal*epochstep)

YYYYMMDD = "20210207"
waveform = fetch(YYYYMMDD,
                 '/997_data/solo_rpw/tds_wf_e',
                 "tds_wf_e",
                 "_rpw-tds-surv-tswf-e-cdag_",
                 ["V04.cdf","V03.cdf","V02.cdf","V01.cdf"],
                 True)
i=332   #1, 19, 20, 126, 143, 175, 214
channel = 0
irange = np.arange(0,16384,1)
cutoff = 370

waveform_quality = waveform.varget('QUALITY_FACT')
waveform_epoch = waveform.varget('EPOCH')
epoch_offset = waveform.varget("EPOCH_OFFSET")
indices = np.arange(len(waveform_quality))[waveform_quality>65000]
waveforms_raw = waveform.varget('WAVEFORM_DATA_VOLTAGE')
waveforms = waveform.varget('WAVEFORM_DATA')

for j in indices:
    for k in [0,1,2]:
        waveforms_raw[j][k,irange] -= np.mean(waveforms_raw[j][k,irange])
        waveforms_raw[j][k,irange] /= np.var(waveforms_raw[j][k,irange])**0.5
    
        waveforms[j][k,irange] -= np.mean(waveforms[j][k,irange])
        waveforms[j][k,irange] /= np.var(waveforms[j][k,irange])**0.5


time_step = epoch_offset[i][1]-epoch_offset[i][0]
nperseg=2048

psd_orig=welch(waveforms_raw[i][channel,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_calibrated=welch(waveforms[i][channel,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)

fig,ax = plt.subplots()
ax.loglog(psd_orig[0][1:]/(2*np.pi),psd_orig[1][1:],"black",label="orig")
ax.loglog(psd_calibrated[0][1:]/(2*np.pi),psd_calibrated[1][1:],"blue",label="calibrated")
ax.legend()
ax.vlines([16,250,50000,131000],1e-20,1e5,color="red",ls="dashed")
ax.set_xlabel("freq [Hz]")
ax.set_ylim(1e-22,1e-1)
ax.set_xlim(10,2e5)

psd_orig_0=welch(waveforms_raw[i][0,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_calibrated_0=welch(waveforms[i][0,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_orig_1=welch(waveforms_raw[i][1,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_calibrated_1=welch(waveforms[i][1,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_orig_2=welch(waveforms_raw[i][2,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_calibrated_2=welch(waveforms[i][2,irange],fs=(1/(time_step/1e9)),nperseg=nperseg)

#attempt to a correction
corrected_0 = hipass_correction(waveforms_raw[i][0,irange],time_step/1e9,cutoff=cutoff)
corrected_1 = hipass_correction(waveforms_raw[i][1,irange],time_step/1e9,cutoff=cutoff)
corrected_2 = hipass_correction(waveforms_raw[i][2,irange],time_step/1e9,cutoff=cutoff)

psd_corrected_0=welch(corrected_0,fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_corrected_1=welch(corrected_1,fs=(1/(time_step/1e9)),nperseg=nperseg)
psd_corrected_2=welch(corrected_2,fs=(1/(time_step/1e9)),nperseg=nperseg)

if channel == 0:
    calibration_average = ((psd_calibrated_0[1][50]/psd_orig_0[1][50])*(psd_orig_0[1][1:]/psd_calibrated_0[1][1:]))**-1
    correction_average =  ((psd_corrected_0[1][50]/psd_orig_0[1][50])*(psd_orig_0[1][1:]/psd_corrected_0[1][1:]))**-1
elif channel == 1:
    calibration_average = ((psd_calibrated_1[1][50]/psd_orig_1[1][50])*(psd_orig_1[1][1:]/psd_calibrated_1[1][1:]))**-1
    correction_average =  ((psd_corrected_1[1][50]/psd_orig_1[1][50])*(psd_orig_1[1][1:]/psd_corrected_1[1][1:]))**-1
elif channel == 2:
    calibration_average = ((psd_calibrated_2[1][50]/psd_orig_2[1][50])*(psd_orig_2[1][1:]/psd_calibrated_2[1][1:]))**-1
    correction_average =  ((psd_corrected_2[1][50]/psd_orig_2[1][50])*(psd_orig_2[1][1:]/psd_corrected_2[1][1:]))**-1
else:
    calibration_average = (((psd_orig_0[1][1:]/psd_calibrated_0[1][1:])+(psd_orig_1[1][1:]/psd_calibrated_1[1][1:])+(psd_orig_2[1][1:]/(psd_calibrated_2[1][1:])))/3)**-1
    correction_average = (((10.8*psd_orig_0[1][1:]/(psd_corrected_0[1][1:]))+(10.8*psd_orig_1[1][1:]/(psd_corrected_1[1][1:]))+(10.8*psd_orig_2[1][1:]/(psd_corrected_2[1][1:])))/3)**-1


fig,ax = plt.subplots()
#ax.loglog(psd_orig_0[0][1:],(psd_orig_0[1][1:]/psd_calibrated_0[1][1:])**-1,"green",label="calibration_ch0")
#ax.loglog(psd_orig_1[0][1:],psd_orig_1[1][1:]/psd_calibrated_1[1][1:],"firebrick",label="calibration_ch1")
#ax.loglog(psd_orig_2[0][1:],2.97*psd_orig_2[1][1:]/(psd_calibrated_2[1][1:]),"navy",label="calibration_ch2")
ax.plot(psd_orig_2[0][1:-166],calibration_average[:-166],label="Inverse response",linestyle=(0,(5,2)))
#ax.loglog(psd_corrected_0[0][1:],11*psd_orig_0[1][1:]/(psd_corrected_0[1][1:]),"red",lw=2,ls="dashed",label="correction_ch0")
#ax.loglog(psd_corrected_1[0][1:],11*psd_orig_1[1][1:]/(psd_corrected_1[1][1:]),"green",lw=2,ls="dashed",label="correction_ch1")
#ax.loglog(psd_corrected_2[0][1:],11*psd_orig_2[1][1:]/(psd_corrected_2[1][1:]),"blue",lw=2,ls="dashed",label="correction_ch2")
ax.plot(psd_corrected_2[0][1:-166],correction_average[:-166],lw=1,label="Laplace correction",linestyle=(0,(1,1)))
ax.plot(psd_orig_2[0][1:-166],correction_average[:-166]/calibration_average[:-166],lw=1,ls="solid",label="Corrected")
ax.set_ylim(6e-1,5e0)
ax.set_xlim(5e1,2e5)
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()
#ax.vlines([16,250,50000,131000],1e-10,1e10,color="red",ls="dashed")
ax.hlines(1,1e-10,1e10,color="grey",ls="dashed")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Response [1]")
fig.tight_layout()
plt.savefig("/998_generated/laplace.pdf", format='pdf')


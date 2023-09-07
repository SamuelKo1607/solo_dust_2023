# Impact Ionization Double Peaks Analyzed in High Temporal Resolution on Solar Orbiter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8325050.svg)](https://doi.org/10.5281/zenodo.8325050)


This is a public archive of the code and, to some extent, the data, used in the article: Impact Ionization Double Peaks Analyzed in High Temporal Resolution on Solar Orbiter. 

## The workflow:

The starting piece of data are the convolutional neural network classification flags files in *\997_data\solo_amplitude_data*, one file per day. The main workflow is included in the Snakefile. First, each of the *RPW_TDS_TSFW_E* files (*.cdf*) is accessed or downloaded, as necessary. For mode about the data, access metadata in the files, or visit [LESIA](https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/docs/DPDD/html/sections/4-data_product_descriptions.html). Then the files are loaded, waveforms are processed, analyzed, printed, and aggreagates are saved in *\997_data\solo_features* in *.npz* files. This is done in paralled with Snakefile, but can be run for individual files as well. When *.npz* files are present, most important plots are produced with *ag2023_delay_plots.py*. Note that VUV data should be present as well, which is achieved with *build_vuv_data.bat* (data download) and *ag2023_vuv_aggregate.py*. Other files, namely *ag2023_download.py*, *ag2023_bias.py* and *commons_XXX* files are imported as assets for the described process to run. 

Aside from the main workflow, *ag2023_frequency_calibration.py* and *ag2023_impact_sampling.py* are used to produce standalone plots.

## Data inputs are included:

1. All the classification files for the Solar Orbiter impacts until NOV 2021, as classified with the convolutional network developed at UiT, see  [ML_dust_detection](https://github.com/AndreasKvammen/ML_dust_detection). Placed in *\997_data\solo_amplitude_data*. The ordering of the impacts is perhaps counterintuitive: first all the impacts classified on-board as *dust* are in the first lines, then all the other (on-board classified as *no dust*) impacts follow.

2. A sample of L2 RPW/TDS triggered snapshot electrical data, as accessed at [LESIA](https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L2/tds_wf_e/). Placed in *\997_data\solo_rpw\tds_wf_e*. The rest can be accessed at the aformentioned link, or downloaded using *ag2023_download.py*. Alternatively, *sunpy.net.Fido* comes recommended. 

3. A sample of EUI fs304 regular image data, as accessed using SoAr (see *ag2023_vuv.py*).  Placed in *\997_data\solo_eui\fsi304*. The outputs produced using these data were not used in the paper, but were used in *ag2023_delay_plots.py*. 

4. A sample of L1 RPW/BIAS data, as access at [LESIA](https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L1). Placed in *\997_data\solo_rpw\bia_current*. 

5. A sample of L3 RPW/SCPOT data as accessed at [LESIA](https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L3/lfr_scpot/). Placed in *\997_data\solo_rpw\lfr_scpot*.

## Outputs are included: 

1. Plots of all the impacts classified as dust with the convolutional network developed at UiT, see  [ML_dust_detection](https://github.com/AndreasKvammen/ML_dust_detection). Both waveforms (*\997_data\solo_features\plots\waveforms_minimal*) and the spectra (*\997_data\solo_features\plots\spectra*). These are produced with *ag2023_feature_extraction.py*, either individually, or making use of *Snakefile*. 

2. All the statistical files, used in paper to describe the features of the impacts. Placed in *\998_generated\solo_statistics*, produced with *ag2023_delay_plots.py*. A pre-requisite for that is the presence of *.npz* files in *\997_data\solo_features*, produced individually with *ag2023_feature_extraction.py* or in batch, making use of *Snakefile*. Another pre-requisite is the presence of */997_data/solo_eui/eui_stats.csv*, produced with *ag2023_vuv_aggregate.py*, but this is a prerequisite for a few of the plots. 

3. The plot of frequency calibration example, placed in *\998_generated\laplace.pdf*. The plot is produced with *ag2023_frequency_calibration.py*. 

5. The output of the Monte Carlo model of the antenna electrostatic response ratios, plots *\998_generated\solo_asymmetry_hist.pdf* and *\998_generated\solo_diagram.pdf*. Both are produced with *ag2023_impact_sampling.py*. 

6. The preprint of the paper describing the procedure and the findings. 


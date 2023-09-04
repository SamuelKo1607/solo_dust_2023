# solo_dust_2023
This is a public archive of the code and, to some extent, the data, used in the article: Impact Ionization Double Peaks Analyzed in High Temporal Resolution on Solar Orbiter. 

Several data inputs are included:

1. All the classification files for the Solar Orbiter impacts until NOV 2021, as classified with the convolutional network developed at UiT, see  [ML_dust_detection](https://github.com/AndreasKvammen/ML_dust_detection). Placed in *\997_data\solo_amplitude_data*. The ordering of the impacts is perhaps counterintuitive: first all the impacts classified on-board as *dust* are in the first lines, then all the other (on-board classified as *no dust*) impacts follow.

2. A sample of original RPW triggered snapshot electrical data, as accessed at [LESIA](https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L2/tds_wf_e/). Placed in *\997_data\solo_rpw\tds_wf_e*. The rest can be accessed at the aformentioned link, or downloaded using *ag2023_download.py*. Alternatively, *sunpy.net.Fido* comes recommended. 

Several outputs are included: 

1. 

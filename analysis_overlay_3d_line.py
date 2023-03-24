'''
    Script for importing tracks from trackmate (FIJI) output (csv or xml) and
    outputting step-size distribution
    Created by Darren McAffee and Laura Nocka, Groves lab
    Toolbox to plot and fit single molecule tracks on 2D surface

'''

from imtrack.trackmate import tmxml, tmcsv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import chi
from scipy.stats import expon
from scipy.integrate import trapz
#from stepfilled import hard_edge
from mpl_toolkits.mplot3d import Axes3D
import mimetypes



### SCRIPT PARAMETERS

# scale = 0.107  # these xmls are already converted
scale = 1
t = 0.020  # time lapse in seconds if not using metadata
N = 3  # diffusional components
NE = 3 # number of exponential components

# given initial parameter estimates, multiply the lowest by this number
# prevents div 0 errors with bounds=[(0,1),...]
SENSITIVITY = 1e-4

btknames1 = ["15_Btk+LAT_1","15_Btk+LAT_2","15_Btk+LAT_3",
"15_Btk+LAT_5","15_Btk+LAT_6"] # name of xml/csv files

btkfolder1 = 'LAT+Btk/' # path to data folder

btknames2 = ["16_Btk+LAT+Grb2_1","16_Btk+LAT+Grb2_2","16_Btk+LAT+Grb2_3",
"16_Btk+LAT+Grb2_4","16_Btk+LAT+Grb2_5","16_Btk+LAT+Grb2_6"] # name of xml/csv files

btkfolder2 = 'LAT+Grb2+Btk/' # path to data folder

btknames3 = ["14_Btk+LATphase_1","14_Btk+LATphase_2","14_Btk+LATphase_3",
"14_Btk+LATphase_4","14_Btk+LATphase_5","14_Btk+LATphase_6","14_Btk+LATphase_7",
"13_Btk+LATphase_6","13_Btk+LATphase_7","13_Btk+LATphase_8","13_Btk+LATphase_9",
"13_Btk+LATphase_10","13_Btk+LATphase_11"] # name of xml/csv files

btkfolder3 = 'Phase+Btk/' # path to data folder

# btknames4 = ["0nM_1","0nM_3","0nM_4","0nM_5","0nM_6","0nM_7","0nM_8"]
#
# btkfolder4 = 'Btk/'

names = [btknames1, btknames2, btknames3]

folders = [btkfolder1, btkfolder2, btkfolder3]

filetypes = ['xml','xml','xml']

figname = "3D_overlay"

logfile = open('log_'+figname+'.txt','w+')

stepbins = 150
dwellbins = 25
tl = [1, 2, 3, 4] # steps to skip

# colors 'Facebook Messenger 1' from color-hex.com
c0 = '#0084ff'
c1 = '#44c789'
c2 = '#ffc300'
c3 = '#fa3c4c'
c4 = '#ff598f'
colors = [c4, c2, c0, c1]

# OME-TIF format for parsing metadata
#dwelltifs = ['21_30nMBtk_150pMGrb2_stream_tirf647_20ms_em999_10MHz_1xg_1x.ome.tif',
#'21_30nMBtk_150pMGrb2_stream_tirf647_20ms_em999_10MHz_1xg_1x001.ome.tif',
#'21_30nMBtk_150pMGrb2_stream_tirf647_20ms_em999_10MHz_1xg_1x002.ome.tif']

### END SCRIPT PARAMTERS

# Import data
# If using filters in trackmate, import data with tmcsv and supply track, link,
# spot files

fig = plt.figure()
ax = plt.axes(projection ='3d')


# find appropriate bins for all files and delays
for btknames in names:
    pos = names.index(btknames)
    btkfolder = folders[pos]
    type = filetypes[pos]
    color = colors[pos]
    grp = []
    if type == 'xml':
        for f in btknames:
            track, spot = tmxml(btkfolder, f)
            grp.append(spot)
    else:
        for f in btknames:
            track, spot = tmcsv(btkfolder, f)
            grp.append(spot)

    #### DIFFUSION COEFFICIENT ANALYSIS

    def step(df, stride=2):
        df = df[::stride]
        if len(df) < 2:
            return np.array([])
        f = df[1:-1]
        i = df[:-2]
        dx = (f.position_x.values - i.position_x.values)**2
        dy = (f.position_y.values - i.position_y.values)**2
        d = np.sqrt(dx + dy)
        return d

    cols = ['position_x', 'position_y', 'frame']

    spot.groupby('track_id').size().idxmax()
    track = spot[spot.track_id == 1][cols]
    track.head(20)

    # Separate into arrays skipping various numbers of steps (tl)
    steps = []
    for i in tl:
        st = []
        for a in grp:
            a = a[a.track_id >= 0]
            s = a.groupby('track_id').apply(step, stride=i)
            s = np.concatenate(s.values)
            st.extend(list(s))
            #print(i, len(st),file=logfile)
        steps.append(np.array(st)) # all steps from one tl appended to single axis
    steps = np.array(steps, dtype=object) # should not change the shape of steps, but makes it possibleto perform *
    # convert to um
    steps = steps * scale
    print(steps,file=logfile)

    # pool all steps (from all delays and samples) to get good bins for
    # plotting all histograms on same plot.
    #_, bins = np.histogram(np.concatenate(steps), stepbins) #step array flattened to determine bins
    #generate evenly spaced bins
    #hist_bins = np.linspace(bins.min(), bins.max(), stepbins*5)
    #bins = (hist_bins[:-1] + hist_bins[1:])/2 #compute bin centers

    ### PLOT THESE PUPPIES
    x = 0
    for s in steps:
        i = tl[x]
        ys, hist_bins = np.histogram(s, bins=stepbins, range=(0,1), density=True)
        xs = (hist_bins[:-1] + hist_bins[1:])/2 #compute bin centers
        #print(len(ys),file=logfile)
        delay = i*20
        size = (hist_bins.max()-hist_bins.min())/(15*stepbins)
        #print(delay,file=logfile)
        ax.plot(xs, ys, zs=delay, zdir='y', color=color, ls='solid')
        x+=1

    # if only one time delay
    # for s in steps:
    #     ys, hist_bins = np.histogram(s, bins=stepbins, range=(0,1), density=True)
    #     xs = (hist_bins[:-1] + hist_bins[1:])/2 #compute bin centers
    #     #print(len(ys),file=logfile)
    #     delay = tl*20
    #     size = (hist_bins.max()-hist_bins.min())/(15*stepbins)
    #     #print(delay,file=logfile)
    #     ax.plot(xs, ys, color=color, ls='solid')

plt.rcParams.update({'mathtext.fontset': 'cm'})

ax.set_xlabel('r ($\mu$m)')
# ax.set_ylabel('Delay time (ms)')
ax.set_ylabel('Probability')
#plt.tight_layout()
plt.show()

fig.savefig('figures/stepsize_'+figname+'.pdf', dpi=300, transparent=True)

### CLOSE OUT
print('COMPLETE',file=logfile)
logfile.close()

'''
    Script for importing tracks from trackmate (FIJI) output (csv or xml)
    for diffsuion coefficient and dwell time analysis
    Created by Darren McAffee and Laura Nocka, Groves lab
    Toolbox to plot and fit single molecule tracks on 2D surface
    Some aspects still in progress (i.e. dwell time fit parameters)
'''




from imtrack.trackmate import tmcsv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import chi
from scipy.stats import expon
from scipy.integrate import trapz
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D



### SCRIPT PARAMETERS

# scale = 0.107  # these xmls are already converted
scale = 1
TIME = 0.02  # time lapse in seconds if not using metadata
N = 3  # diffusional components
IMMOBILE = True  # whether to model one immobile fraction if diffusion data
NE = 2 # number of exponential components
stepbins = 150
cutoff = 1.1  # diffusion step size cutoff in microns
dwellbins = 25
tl = [2, 3, 4, 5] # steps to skip
# given initial parameter estimates, multiply the lowest by this number
# prevents div 0 errors with bounds=[(0,1),...]
SENSITIVITY = 1e-4

btknames = [] # name of xml/csv files

btkfolder = '' # path to data folder containing data

figname = "" # name of output figure

logfile = open('log_'+figname+'.txt','w+')

# colors 'Facebook Messenger 1' from color-hex.com
c0 = '#0084ff'
c1 = '#44bec7'
c2 = '#ffc300'
c3 = '#fa3c4c'
c4 = '#d696bb'
colors = [c0, c1, c2, c3, c4]

### END SCRIPT PARAMTERS

# Import data
# If using filters in trackmate, import data with tmcsv and supply track, link,
# spot files
grp = []
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
        print(i, len(st),file=logfile)
    sta = np.array(st)
    steps.append(sta[sta < cutoff])
steps = np.array(steps, dtype = object)

# convert to um
steps = steps * scale

# pool all steps (from all time steps) to get good bins for
# plotting all histograms on same plot.
_, globalbins = np.histogram(np.concatenate(steps), stepbins)
globalxmp = (globalbins[1:] + globalbins[:-1])/2
xs = np.linspace(globalbins.min(), globalbins.max(), stepbins*5)

plt.rcParams.update({'mathtext.fontset': 'cm'})

#see Darren's notes on diffusion to see this derivation
def scale2D(scale, time):
    return scale**2/2/time

def D2scale(Dif, ts):
    return np.sqrt(abs(2 * Dif * ts))

# Mixture model as sum of multiple chis
# Number of chi determined by N denoted in script parameters
def NChi(x, *prms):
    """
    prms = [alpha1, alpha2, ..., alphaN, scale1, ..., scaleN]
    We need N-1 "alpha" parameters and N "scale" parameters.
    """
    nargs = len(prms)
    alphas = prms[:nargs//2]
    scales = prms[nargs//2:]
    lp = 0
    for a, s in zip(alphas, scales):
        lp += a * chi(loc=0, scale=s, df=2).pdf(x)
    return lp

def NChi_error(prms, x=None, ydata=None):
    y = NChi(x, *prms)
    return ((y - ydata)**2).sum()

def NChiT(r_arr, t_arr, *prms):
    """
    Gives the step size distribution at various time points.
    x_arr: array of distances for which to get PDF values
    t_arr: array of time "steps". E.g. 2-step, 4-step, etc.
    prms = [alpha1, alpha2, ..., alphaN, scale1, ..., scaleN]
    We need N-1 "alpha" parameters and N "scale" parameters.
    """
    assert r_arr.shape == t_arr.shape
    assert r_arr.min() >= 0
    p = np.zeros_like(r_arr)
    nargs = len(prms)
    alphas = prms[:nargs//2]
    D = prms[nargs//2:]
    dev = []
    for a, d in zip(alphas, D):
        distr = chi(loc=0, scale=D2scale(d, TIME), df=2)
        p += a * distr.pdf(r_arr)
        dev.append(distr.std())
    return p, dev

def NChiT_error(prms, r=None, t=None, ydata=None):
    """
    Calculate error for NChiT. Including penalties for non-normalized
    probabilities and negative diffusion coefficients.

    Begin by making guesses for parameters and evaluating error.

    Scale errors to approximately within the same order of magnitude.
    """
    y, _ = NChiT(r, t, *prms) # typical values ~100 for a "good guess"
    ye = ((y - ydata)**2).sum() * 1e-5 # residuals
    p = prms[:len(prms)//2] # sum alphas
    pe = (sum(p) - 1)**2 # typical values ~1, rescale, penalty for non-norm
    d = prms[len(prms)//2:] # scales
    de = sum(1e2*(abs(d) - d)**2) # penalty for negative diffusion coef.
    return (ye + pe + de)

# Create a norm function to ensure sum(alphas) = 1
def normChi(prms, chis=N):
    #print(prms[:chis])
    y = sum(prms[:chis])
    return (y-1)


nrm = lambda prms: normChi(prms, N)
cons = {'type': 'eq', 'fun': nrm}
yt = []
x0 = []
for j in range(N):
    x0.append(np.percentile(steps[len(tl)//2], (j+1)/(N+1)*100))
bounds = [[SENSITIVITY**2,1.0]]*N + [[min(x0)*SENSITIVITY, 5.0]]*N
bounds = np.array(bounds)
x0 = [1/N]*N + x0
x0 = np.array(x0)
for i,b in zip(x0, bounds):
    assert b[0] <= i and i <= b[1]
for i in range(len(tl)):
    data = steps[i]
    yvals, _, _ = plt.hist(data, bins=globalbins, density=1)
    yt.append(yvals)
yt = np.array(yt)
# Create a function (error) to minimize residuals of NChi
# Define constraints to ensure sum(alphas) = 1
R, T = np.meshgrid(globalxmp, np.array(tl)*TIME)
res = minimize(NChiT_error, x0=x0, args=(R, T, yt))
print(res, file=logfile)
print('initial diffusion guesses:', [f'{i:.04f}' for i in x0[N:]], file=logfile)
print('final diffusion parameters:', [f'{i:.04f}' for i in res.x[N:]], file=logfile)
print('final component parameters:', [f'{i:.04f}' for i in res.x[:N]], file=logfile)
# Compute the diffusion coefficients
# read diffusion.pdf for an explanation of the math below
# the last D (from farthest excursion is most accurate)
# NChiT_error(res.x, R, T, yt, False)
print('-----------------\n', file=logfile)


# ## Plot all time steps in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
dx = globalxmp[1] - globalxmp[0]
# for t, y, c in zip(tl, yt, colors):
#     ax.bar(globalxmp, y, zs=t*TIME, zdir='y', color=c, ec=c, alpha=0.3, width=dx)
# R, T = np.meshgrid(globalxmp, np.array(tl)*TIME)
# ax.plot_surface(R, T, yt, alpha=0.7)
R, T = np.meshgrid(globalxmp, np.linspace(min(tl)*TIME, max(tl)*TIME*1.2, 40))
Z, stdev = NChiT(R, T, *res.x)
for m in enumerate(stdev):
    stdev[m[0]] = scale2D(stdev[m[0]], TIME)
print('Standard deviation of diffusion parameter', stdev, file=logfile)
print('Ensure time slice is PDF, area:', np.trapz(Z[len(T)//2],R[len(T)//2]), file=logfile)
# ax.plot_trisurf(R.ravel(), T.ravel(), Z.ravel(), alpha=0.7, color='gray')
# ax.set_zlim(0, 14)
# ax.set_zlabel('Probability')
# ax.tick_params(labelsize=6)
# # Print D calculated from each chosen tl
# for i, d in enumerate(res.x[N:]):
#     fig.text(0.2+i*0.3, 0.95, f"$D_{i+1}=${d:.04f} $\mu m^2/s$", fontsize=11, transform=ax.transAxes)
# #fig.text(0.01, 0.95, r"$\frac{r}{2Dt} e^{-\frac{r^2}{4Dt}}$", fontsize=20, transform=ax.transAxes)
# plt.xlabel('r ($\mu$m)')
# #plt.tight_layout() #not sure why this attribute isn't working
# plt.show()
# fig.savefig('figures/stepsize_'+figname+'.pdf')

## Plot each time step separately in a stack
fig, ax = plt.subplots(nrows=len(tl), figsize=(5, N*3))
xv = np.linspace(globalbins.min(), globalbins.max(), 1000)
for i, a in enumerate(ax):
    y = yt[i]
    R, T = np.meshgrid(xv, TIME*tl[i])
    Z, _ = NChiT(R, T, *res.x)
    a.bar(globalxmp, y, width=dx, alpha=0.5)
    a.plot(R[0], Z[0], color='gray')
    a.set_title(f'Time steps: {tl[i]}',fontsize=11)
    a.set_xlim(right=1)
    a.set_ylim(top=14)
    a.tick_params(labelsize=8)
plt.ylabel('Probability')
plt.xlabel('r ($\mu$m)')
plt.tight_layout()
# plt.show()
fig.savefig('figures/stack_'+figname+'.pdf')

### LIFETIME ANALYSIS
# filter out tracks that touch border
def is_interior(df, pad=2, left=0, right=512, top=0, bottom=512):
    x = df['position_x']
    if (x < pad).any() or (x > right - pad).any():
        return False
    y = df['position_y']
    if (y < pad).any() or (y > bottom - pad).any():
        return False
    return True


within = []
r, b = 512, 512
for a in grp:
    within.append(a.groupby('track_id').filter(is_interior, right=r, bottom=b))

# Getting tracking lifetime in terms of frames
def get_frames(df):
    return df.frame.max() - df.frame.min()

frames = []
for w in within:
    frames.append(w.groupby('track_id').apply(get_frames).values)
frames = np.concatenate(frames)

# Plot Dwell Times
#Create xs, convert to real time and take 95% of data
time = frames*TIME
timec = time[time < np.percentile(time, 95)]

# PDF the values and calculate bin centers
density, bins, _ = plt.hist(timec, bins=dwellbins, density=1)
plt.close('all')
xpoints = (bins[1:] + bins[:-1])/2

# Mixture model as sum of multiple chis
# Number of chi determined by N denoted in script parameters
def NExp(x, *prms):
    """
    The probabilities of N exponential survival distributions.
    prms = [alpha1, alpha2, ..., alphaN, scale1, ..., scaleN]
    """
    nargs = len(prms)
    alphas = prms[:nargs//2]
    scales = prms[nargs//2:]
    lp = 0
    err = []
    for a, s in zip(alphas, scales):
        lp += a * expon(loc=0, scale=s).sf(x)
        # calculate the standard deviation for this fit
        err.append(expon(loc=0, scale=s).sf(x).std()) # something is wrong here
    return lp, err

def NExp_error(prms, xdata=None, ydata=None):
    y, _ = NExp(xdata, *prms)
    return sum((1e2*(y - ydata))**2)

def normE(prms, n=NE):
    return sum(prms[:n]) - 1


# normalize data to "survival probabilities"
xx = xpoints-xpoints[0]
dd = density/density[0]

x0 = []
for i in range(NE):
    x0.append(np.percentile(timec, (i+1)/(NE+1)*100))
pLB = min(x0)*SENSITIVITY
bounds = [[0,1]]*NE + [[pLB, np.inf]]*NE  # minimize
x0 = [1/NE]*NE + x0
cons = {'type': 'eq', 'fun': normE}
res = minimize(NExp_error, x0, args=(xx, dd),bounds=bounds, method='SLSQP', constraints=cons)
_, conf = NExp(x0, *res.x)
print(res, file=logfile)
print('initial dwell guesses:', [f'{i:.04f}' for i in x0[NE:]], file=logfile)
print('final dwell parameters:', [f'{i:.04f}' for i in res.x[NE:]], file=logfile)
print('This is the standard deviation:', conf, file=logfile)
print('-----------------\n', file=logfile)


for ys in ['linear', 'log']:
    fig, ax = plt.subplots(figsize=(5,3))
    ax.scatter(xx, dd, label='data')
    xa = np.linspace(xx.min(), xx.max(), 1000)
    fits, _ = NExp(xa, *res.x)
    ax.plot(xa, fits, lw=2, color=c1, ls='--',label='fit')
    ax.set_title('Btk Dwell Time')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Rel. Probability')
    ax.set_xlim(right=1.2)
    ax.legend()
    plt.yscale(ys)
    plt.tight_layout()
    # plt.show()
    if ys == 'linear':
        fig.savefig('figures/dwell_log_'+figname+'.pdf')
    if ys == 'log':
        fig.savefig('figures/dwell_log_'+figname+'.pdf')

print('COMPLETE', file=logfile)
logfile.close()

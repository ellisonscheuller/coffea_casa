from cycler import cycler
from functools import reduce
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import operator

################################################################################################

'''
draws score thresholds as vertical lines on matplotlib axis "ax" using dictionary threshold_map, 
which has keys given by threshold names and values which are dicts that provide threshold name
for printing, threshold value, and color
'''
def draw_thresholds(ax, threshold_map):
    for th in threshold_map.values():
        ax.axvline(x=th['score'], 
                   label=th['name'], 
                   color=th['color'],
                   linestyle='--',
                   linewidth=2,
        )
      
    
'''
returns hist.Hist object from coffea return dictionary, accessed by keys in order given by items 
in the list mapList
mapList example: ['Scouting_2024G', 'hists', 'dimuon_m']
'''
def getHist(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)
   
        
'''
sets matplotlib color cycle according to colors, a list of string hex values (maintains color 
order between plots)
'''
def setColorCycle(colors = [
    '#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6', 
    '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd'
]):
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)

'''
draws 1d histogram from hist.Hist object "hist_in" for trigger "trigger" on mpl axis "ax".
"label": optional label of errorbar object
"rebin": rebinning factor for histogram, 1 be default
"norm": whether to normalize histograms
'''
def draw_hist1d(hist_in, ax, trigger, label='', rebin=1, norm=False):
    hist_in = hist_in[:, trigger, hist.rebin(rebin)] # slice histogram and rebin
    counts, _, bins = hist_in.to_numpy() # get bin information from hist object
    
    if len(counts)>0: # check that there are events to plot
        _counts = counts[0]/(np.sum(counts[0])*np.diff(bins)) if norm else counts[0]
        errs = (np.sqrt(counts[0])/(np.sum(counts[0])*np.diff(bins)) 
                if norm else np.sqrt(counts[0]))
        _errs = np.where(_counts==0, 0, errs)
        bin_centres = 0.5*(bins[1:] + bins[:-1])
        l = ax.errorbar(x=bin_centres,y=_counts,yerr=_errs,linestyle='')
        color = l[0].get_color()
        ax.errorbar(x=bins[:-1],y=_counts,drawstyle='steps-post',label=label,color=color)
        
    else:        
        l = ax.errorbar(x=[],y=[],yerr=[],drawstyle='steps-post') # plot nothing
        color = l[0].get_color()
        ax.errorbar(x=[],y=[],drawstyle='steps-post',label=label,color=color)

        
'''
draws 1d histogram from hist.Hist object "hist_in" for trigger "trigger" on matplotlib axis "ax".
"label": optional label of errorbar object
"rebin": rebinning factor for histogram, 1 is default
"obj": if the histogram has an object axis, select this particular object
"norm": whether to normalize histograms
'''
def draw_hist1d(hist_in=None, ax=None, trigger=None, label='', rebin=1, obj=None, norm=False):
    # slice histogram and rebin
    # if rebin == None: # rebin automatically according to Terrel-Scott rule
    #     if obj:
    #         counts = hist_in.values()[...,trigger,obj].sum()
    #         nbins = hist_in[...,trigger,obj].size()
    #     else:
    #         counts = hist_in.integrate("trigger", trigger).values().sum()
    #         nbins = hist_in.size
    #     nbins_new = np.power(2*counts,1/3)
    #     if nbins_new < nbins:
    #         rebin = int(nbins/nbins_new)
    if obj:
        try:
            hist_in = hist_in[:, trigger, obj, hist.rebin(rebin)] 
        except:
            return
    else:
        try:
            hist_in = hist_in[:, trigger, hist.rebin(rebin)] 
        except:
            return
        
    counts, _, bins = hist_in.to_numpy() # get bin information from hist object
    
    if len(counts)>0: # check that there are events to plot
        _counts = counts[0]/(np.sum(counts[0])*np.diff(bins)) if norm else counts[0]
        errs = (np.sqrt(counts[0])/(np.sum(counts[0])*np.diff(bins)) 
                if norm else np.sqrt(counts[0]))
        _errs = np.where(_counts==0, 0, errs)
        bin_centres = 0.5*(bins[1:] + bins[:-1])
        l = ax.errorbar(x=bin_centres,y=_counts,yerr=_errs,linestyle='')
        color = l[0].get_color()
        ax.errorbar(x=bins[:-1],y=_counts,drawstyle='steps-post',label=label,color=color)
        
    else:        
        l = ax.errorbar(x=[],y=[],yerr=[],drawstyle='steps-post') # plot nothing
        color = l[0].get_color()
        ax.errorbar(x=[],y=[],drawstyle='steps-post',label=label,color=color)
    return
    
'''
draws 2d histogram from hist.Hist object "hist_in" for trigger "trigger" on matplotlib axis "ax".
"x_rebin": rebinning factor for histogram x-axis, 1 is default
"y_rebin": rebinning factor for histogram y-axis, 1 is default
"obj": if the histogram has an object axis, select this particular object
"norm": whether to normalize histograms
"log": whether z-axis should be log-scale, False is default
returns array given by pcolormesh method
'''
def draw_hist2d(hist_in, ax, trigger, x_var, y_var, x_rebin=1, y_rebin=1, obj=None, norm=False, log=False):
    if log: norm_method = 'log'
    else: norm_method = 'linear'
    
    cmap = plt.get_cmap('plasma')

    if obj:
        h = hist_in[:, trigger, obj, hist.rebin(x_rebin), hist.rebin(y_rebin)]
    else: 
        h = hist_in[:, trigger, hist.rebin(x_rebin), hist.rebin(y_rebin)]
        
    w, x, y = h.project(x_var,y_var).to_numpy()
    
    if norm:
        mesh = ax.pcolormesh(x, y, np.where(w.T==0, np.min(w.T[w.T!=0]), w.T),  cmap=cmap, norm=norm_method)
    else:
        mesh = ax.pcolormesh(x, y, w.T, cmap=cmap, norm=norm_method)

    return mesh

def multipage(filename, figs=None, dpi=200):
    """Creates a pdf with one page per plot"""
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        print("No figures handed")
    for fig in figs:
        plt.figure(fig).savefig(pp, format='pdf')
    pp.close()


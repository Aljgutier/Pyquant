
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import Normalize
import datetime as dt
import logging
from matplotlib.cbook import boxplot_stats


def get_plot_defaults():

	plot_defaults = {
	'pltstyle': 'seaborn',
	'figsize' : (18,9) ,
	'plottypes': '',
	'title' :  '',               #  list ot titles, 1 per axis  #  list of x,y tuples
	'legendloc':'best',
	'marker_label':'',
	'ylims' :None,               # list of two-tuples ylims (lower, upper)
	'xlims' :None,
	'figsize' :  None,
	'wspace': 0.1,
	'hspace': 0.1,
	'sharex':False,
	'xlabelfontsize':16,
	'xticklabelsize':16,
	'xtickfontsize': 16,
	'xtickrotation':0,
	'ylabelfontsize':16,
	'ytickfontsize':16,
	'ytickrotation':0,
	'titlefontsize':18,
	'legendsize':16,
	'legend':True,
	'xlabel':'',
	'ylabel':'',
	'title':''
	}

	return plot_defaults

def set_axisparams(options_dict,ax):

	title=options_dict['title']
	titlefontsize=options_dict['titlefontsize']
	marker_label = options_dict['marker_label']
	legendloc=options_dict['legendloc']
	legendsize=options_dict['legendsize']
	xlabel=options_dict['xlabel']
	xlabelfontsize=options_dict['xlabelfontsize']
	xlims=options_dict['xlims']
	xtickfontsize=options_dict['xtickfontsize']
	xtickrotation=options_dict['xtickrotation']
	ylabel=options_dict['ylabel']
	ylabelfontsize=options_dict['ylabelfontsize']
	ytickfontsize=options_dict['ytickfontsize']
	ytickrotation=options_dict['ytickrotation']
	ylims=options_dict['ylims']


	ax.set_xlabel(xlabel,fontsize=xlabelfontsize)
	ax.set_ylabel(ylabel,fontsize=ylabelfontsize)

	for x_tick in ax.get_xticklabels():
		x_tick.set_fontsize(xtickfontsize)
		x_tick.set_rotation(xtickrotation)

	for y_tick in ax.get_yticklabels():
		y_tick.set_fontsize(ytickfontsize)
		y_tick.set_rotation(ytickrotation)

	
	ax.set_title(title, fontsize=titlefontsize)

	if marker_label != '': ax.legend(loc=legendloc,prop={'size':legendsize})

	if ylims != None:
		ax.set_ylim(ylims[0],ylims[1])
	    
	if xlims != None:
	 	ax.set_xlim(xlims[0],xlims[1])

	if options_dict['legend'] == True:
		ax.legend( loc=options_dict['legendloc'], prop={'size': options_dict['legendsize']})

	return None


# Easy Bar Plot

def barplot(x, names, ax ,HorizontalBars=True):
    
    if HorizontalBars == True:  # horizontal looks good with many variable, 
        indices = list(np.argsort(x))
    else:                   # vertical for fewer variables
        # sort importances descending
        indices = list(np.argsort(x)[::-1])
        
    n = [ names[i] for i in indices]
    
    if HorizontalBars == True:
        
        print(names)
        print(x)
        
        ax.barh(n, x[indices], color = 'blue', align ='center')
        xticklabel_rotation = 0
    else:
        ax.bar(n, x[indices], color = 'blue')
        xticklabel_rotation = 60

    plt.xticks(rotation=xticklabel_rotation,fontsize=14)
    plt.yticks(fontsize=14)
    ax.grid(color='b', ls = '-.', lw = 0.25)
    ax.set_title("Feature Importance",fontsize=18)
    
    return
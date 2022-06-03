import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


"""
.. module:: fmplot.py
    :Python version 3.7 or greater
    :synopsis: functions for plotting financial time-series data including sub-plots, line plots, and market cycle (stem) plots.

.. moduleauthor:: Alberto Gutierrez <aljgutier@yahoo.com>
"""


def fmplot(df,variables,**kwargs):

    """
    Plot financial market timeseries data. 

    **Parameters**:
        **df** (datatfrme): dataframe conntaining price variable to be analyzed for market cycles

        **variables** (string): the variable or list of variables to be plotted. Each item in the list will be placed on its own axis. If the variables are a list of lists. Then each sub list is placed on its own access with each sub list representing multiple variables for each axis.


    **Kwargs**: 
        
        **df** (dataframe): input data frame containing the variables (columns) to be plotted. The df.index must be in datatime.
        
        **fb** (datetime_1,datetime_2): fill between two datetime horizontal values. For example, gray recession periods

        **variables** (string): a string indicating the variable (i.e., column) to be plotted. If variabes is a list, then an axis will be created for each variable. Variables can be a list of lists, in which case each sublist corresponds to the variables to be plotted on the corresponding axis.

        **startdate** (datetime): the start date (x-axis) for all axis. The default is the first date in the df.index.

        **enddate** (datetime): the end date (x-axis) for all axis. The default is the last date in the df.index.

        **ylims** (2-tuple, numeric): the lower and upper limit of the vertical axis.

        **titles** (string): a string title for the axis, or a list of titles for each axis.

        **titledate** (boolean): If True append the start and end date, derived from the df index, to the chart title

        **titlein** (boolean): If True (default False) the title is positioned above the axis. If True, the title is positioned to be centered inside the axis centered at x = 0.5 (50%) and height of 0.9 (90%) of the graph area.

        **titlexy** (2-tuple, numberic): If titlein = True,  titlexy is the x and y position of the title. x and y range from 0 to 1 (fraction) of the graph areaa. The title is centered at x.

        **figsize** (2-tuple, integers): Is the a 2-tuple (Width, Hieght) of the total plotting area in inches area for all axis.

        **plottypes** (string): Default line, mktcycle (stem chart, without the dot/bubble on each stem).

        **labels** (string): the string label names to be used for identifying the plotted labels. The string indicated by label will be used in the chart legend.

        **llocs** (string): the legend location for the axis, specified according to matplotlib, for example "upper right" etc.

        **ncols** (int): designates the number of columns for the legend listing.

        **stemlw** (numeric): the stem line width corresponding to the mktcycle chart style. List of colors, corresponding to the variables plotted on the axis.

        **sharex** (boolean): If True, share the x-axis for all charts. Default is False.

        **hspace** (numberic): The amout of horizontal space between axis. Defalut = 0.1. Make this 0 to hide any space between axis. In this case (hspace = 0) the axis is shared for all charts.

        **linecolors** (string): color or list of colors to use for the line type charts. For multiple axis and multiple *variables* per axis *colors* is a list of lists, with each sublist a set of colors corresponding an axis.

        **pltstyle** (string): matplotlib style, default 'seaborn'

        **hlines** (numeric): draw horizontal line at y value = hlines for axis. Default is '' indicating no line. For a multi-plot (multiple axis) hlines is a list with multiple numeric values or '' indicating no line for the corresponding axis. 

        **vlines** (numeric): draw vertical line at (x_dt:datetimes,...) vlines for axis. Default is '' indicating no line. For a multi-plot (multiple axis) vlines is a list with multiple numeric values or '' indicating no line for the corresponding axis. 
        
        **xaxis_fontsize** (numeric): x-axis label font size

        **yaxis_fontsize** (numeric): y-axis label font size

        **title_fontsize** (numeric): title font size

        **legend_fontsize** (numeric): legend font size

        **xtick_labelsize** (numeric): xtick labels font size

        **ytick_labelsize** (numeric): ytick labels font size

        **annotations** (n-tuple): Tuple of parameters including at least a 3-tuple containing (x,y,text) designating text to be placed on the graph. A 4-th tuple item for color, and 5-th item for fontsize can also be included. annotations can be a list corresponding to text on different axis or list of lists corresponding to multiple annotations per axis.

        **height_ratios** (List, int): a numberic list or relative height ratios (integers) wherein the height of each axis is proportial to the total sum of height_ratios.


    **Additional Functionality**:
        For multiple axis, *variables* is a list of lists, optional parameters, such as *hlines*, must match the length of the *variables* argument. 
        For example, if the *variables* parameter indicates 3 axis, then *hlines* can either be default = '' or *hlines* must be a list containing 3 
        arguments, e.g., *hlines* = [0,'',1], an *hline* for each axis. The number of axis plotted will be the shortest length the *variables* list or the *hlines* list. 
        Similarly, the length of other lists, when not default, must match the *variables* length 
        including *titles*, *ylims*, *labels*, *llocs*, *linecolors*, *plottypes*, and *annotations*.

    |

    """
    pltstyle = kwargs.get('pltstyle', 'seaborn')
    hlines = kwargs.get('hlines','')
    vlines = kwargs.get('vlines','')
    figsize = kwargs.get('figsize', (18,9))
    plottypes = kwargs.get('plottypes', '')
    linecolors = kwargs.get('linecolors', '')
    startdate = kwargs.get('startdate', '')
    enddate = kwargs.get('enddate', '')
    titles = kwargs.get('titles', '')                   #  list ot titles, 1 per axis
    titlexy = kwargs.get('titlexy','')                  #  list of x,y tuples
    titlein = kwargs.get('titlein',False)
    titledate = kwargs.get('titledate',False)
    labels = kwargs.get('labels', '')
    llocs = kwargs.get('llocs','')
    ncols = kwargs.get('ncols','')
    ylims = kwargs.get('ylims','')                    # list of two-tuples ylims (lower, upper)
    figsize = kwargs.get('figsize', [24,12])
    stemlw = kwargs.get('stemlw', 2)
    wspace = kwargs.get('wspace',0.1)
    hspace = kwargs.get('hspace',0.1)
    sharex = kwargs.get('sharex',False)
    height_ratios = kwargs.get('height_ratios','')
    xaxis_fontsize= kwargs.get('xaxis_fontsize',12)
    xtick_labelsize= kwargs.get('xtick_labelsize',12)
    yaxis_fontsize= kwargs.get('yaxis_fontsize',12)
    ytick_labelsize= kwargs.get('ytick_labelsize',12)
    title_fontsize= kwargs.get('title_fontsize',14)
    legend_fontsize= kwargs.get('legend_fontsize',12)
    annotations= kwargs.get('annotations','')
    xlabel=kwargs.get('xlabel','')
    xlabelloc=kwargs.get('xlabelloc','')
    xlabelfontsize=kwargs.get('xlabelfontsize',14)



    # process Kwargs turn into lists if necessary

    # fill between
    fb = kwargs.get('fb','') # fill betweeen tuples

    if startdate == '':
        startdate=df.index[0]

    if enddate == '':
        enddate=df.index[df.index.size-1]

    if not isinstance(variables, list): 
        variables=[variables]

    if plottypes == '' :
        plottypes = ['line']*len(variables)

    if linecolors == '' :
        linecolors = ['']*len(variables)

    if titles == '' :
        titles = ['']*len(variables)

    if llocs == '' :
        llocs = ['upper left']*len(variables)

    if ncols == '' :
        ncols = [1]*len(variables)

    if hlines == '' :
        hlines=['']*len(variables)
        
    if vlines == '' :
        vlines=['']*len(variables)

    if ylims == '' :
        ylims=['']*len(variables)


    if labels=='':
        labels=variables

    if titlexy=='':
        titlexy=['']*len(variables)

    if height_ratios =='':
        height_ratios = [1]*len(variables)

    if annotations== '' :
        annotations = ['']*len(variables)

    if not isinstance(ylims, list): 
        ylims=[ylims]

    if not isinstance(plottypes, list): 
        plottypes=[plottypes]

    if not isinstance(titles, list): 
        titles=[titles]

    if not isinstance(hlines, list): 
        hlines=[hlines]

    if not isinstance(llocs, list): 
        llocs=[llocs]

    if not isinstance(titlexy, list): 
        titlexy=[titlexy]

    titlein=[titlein]*len(variables)  


    # create axis
    plt.style.use(pltstyle)
    fig, axs = plt.subplots(nrows=len(variables),ncols=1,figsize=figsize,sharex=sharex,gridspec_kw={'height_ratios': height_ratios})
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    #plt.tick_params(axis="x",labelsize = xtick_labelsize)
    #plt.tick_params(axis="y",labelsize = ytick_labelsize)


    if not isinstance(axs, np.ndarray): 
        axs=[axs]


    # get x, xindex (datetime), xticks an labels 

    x,xindex,xticks,xlabels=_getx_for_fmplot(df,startdate=startdate,enddate=enddate)

    # fill between
    if fb != '':
        fb2=[]
        for f in fb:
            f0=xindex.get_loc(f[0],method='nearest')
            f1=xindex.get_loc(f[1],method='nearest')
            fb2.append((f0,f1))

    # plot variables on each axis according to the plot type

    for variable,ptype,ax,label,loc,ncol,hline,vline,ylim,title,txy,tin,linecolor,an in zip(variables,plottypes,axs,labels,llocs,ncols,hlines,vlines,ylims,titles,titlexy,titlein,linecolors,annotations):
        # line graph
        
        if ptype == 'line' or ptype =='':

            # ensure varibles are iterable
            vars=variable
            if not isinstance(vars, list): 
                vars=[vars]

            labs=label
            if not isinstance(labs, list): 
                labs = [labs]

            cs=linecolor
            if not isinstance(cs, list): 
                cs = [cs]*len(vars)

            # loop through variables for this axis
            for v,l,c in zip(vars,labs,cs):
     
                y=_gety_for_fmplot(df,v,startdate=startdate,enddate=enddate)

                if (c != '') :
                    ax.plot(x,y,c = c, label=l)
                else:
                    ax.plot(x,y, label=l)

                plt.setp(ax, xticks=xticks, xticklabels=xlabels)

            if an != '':
                for a in an:
                    x_an=xindex.get_loc(a[0],method='nearest')
                    #x=t[0]*xindex.sized
                    color = 'b'
                    fsize = 12
                    if len(a)>=4:
                        color = a[3]
                    if len(a)==5:
                        fsize=a[4]
                    
                    ax.annotate(a[2],xy=(x_an,a[1]), ha='left', rotation=0, wrap=True, 
                    fontsize=fsize, family='serif', style='normal', color=color) # styles are normal, italic, oblique

        # mkt cycle stem plot
        if ptype == 'mktcycle':
            y=_gety_for_fmplot(df,variable,startdate=startdate,enddate=enddate)

            y_ltzero=y[y<0]
            x_foryltzero=x[y<0]
            y_gtzero=y[y>=0]
            x_forygtzero=x[y>=0]

            #markerline, stemlines, baseline=ax.stem(x,y,markerfmt=" ", label='bull market', basefmt=' ')
            markerline, stemlines, baseline=ax.stem(x_forygtzero,y_gtzero,markerfmt=" ", label='bull market', basefmt=' ',use_line_collection=True)
            #if you want to change each line, then loop through stemplines
            #for stem in stemlines:
            #    stem.set_linewidth(0.1)

            plt.setp(baseline, color='red')
            plt.setp(stemlines, linestyle="-", color="blue", linewidth=stemlw,alpha=1 )

            markerline, stemlines, baseline=ax.stem(x_foryltzero,y_ltzero,markerfmt=" ", label='bear market',basefmt=' ', use_line_collection=True)
            #if you want to change each line, then loop through stemplines
             #for stem in stemlines:
            #    stem.set_linewidth(0.1)
            plt.setp(baseline, color='red')
            plt.setp(stemlines, linestyle="-", color="red", linewidth=stemlw,alpha=1 )



            plt.setp(ax, xticks=xticks, xticklabels=xlabels)

            ax.tick_params(axis='y',labelsize=ytick_labelsize)
            ax.tick_params(axis='x',labelsize=xtick_labelsize)

            #plt.annotate('Testing\nThis\nOut', xy=(0.5, 0.5), style='normal', color='b', fontsize=12, family='serif')

            if an != '':
                for a in an:
                    x_an=xindex.get_loc(a[0],method='nearest')
                    #x=t[0]*xindex.size
                    color = 'b'
                    fsize = 12
                    if len(a)>=4:
                        color = a[3]
                    if len(a)==5:
                        fsize=a[4]
                    

                    ax.annotate(a[2],xy=(x_an,a[1]), ha='left', rotation=0, wrap=True, 
                    fontsize=fsize, family='serif', style='normal', color=color) # styles are normal, italic, oblique


        # post loop parameters that apply to all types of graphs
        # fill between
        if fb != '':
            for f in fb2:
                ax.axvspan(f[0], f[1], alpha=0.5, color='grey')

        if titledate:
            title = title  + ': ' + str(startdate.year)+ '.'+str(startdate.month)+ '.' +str(startdate.day) \
                    + ' - ' + str(enddate.year)+ '.' + str(enddate.month) + '.'+ str(enddate.day)
        if title != '':
            tx = 0.5
            ty = 0.8
            if not tin:
                ax.set_title(title,fontsize=title_fontsize)
            else:
                if txy == '':
                    ax.set_title(title,x=tx, y=ty,fontsize=title_fontsize)
                else:
                    ax.set_title(title,x=txy[0], y=txy[1],fontsize=title_fontsize)

        if hline != '':
            ax.axhline(y=hline, xmin=0.0, xmax=1.0, color='k',lw=1)
            
        if vline != '':
            for vl in vline:
                x_dt=xindex.get_loc(vl,method='nearest')
                ax.axvline(x=x_dt, ymin=0, ymax=1, color='k',lw=1)

        if ylim != '':
            ax.set_ylim([ylim[0],ylim[1]])



        if loc == '':
            ax.legend(fontsize=legend_fontsize,markerscale=14)
        else:
            ax.legend(loc=loc,fontsize=legend_fontsize,markerscale=14)

    if xlabel != '':
        if xlabelloc == '':
            xy = (0,-0.2)
        else:
            xy =(xlabelloc[0],xlabelloc[1])
        if xlabelfontsize =='':
            xlabelfontsize = 14
        ax.annotate(xlabel, xy=xy , xycoords='axes fraction', fontsize=xlabelfontsize)

    ax.tick_params(axis='y',labelsize=ytick_labelsize)
    ax.tick_params(axis='x',labelsize=xtick_labelsize)

    plt.show()

    return
    # end plot_fits


def _getx_for_fmplot(df,**kwargs):
    startdate = kwargs.get('startdate','')
    enddate = kwargs.get('enddate','')

    if startdate=='' :
        startdate=df.index[0]
        
    if enddate=='' :
        enddate=df.index[df.index.size-1]


    first_year = startdate.year
    first_month = startdate.month
    first_day = startdate.day

    xindex=df.loc[startdate:enddate].index
    x=np.arange(0,len(xindex))
    xticks=[]
    xlabels=[]

    if (enddate.year - startdate.year <=10):
        ystep=1
    elif (enddate.year - startdate.year <=20):
        ystep=2 
    else:
        ystep=5
    # setup xticks ... yticks = year ticks
    # yticks correspond to "year" ticks for setting up x labels

    yearticks=list(range(startdate.year,enddate.year+1,ystep))


    mticks=[]
    months =  (enddate.year-startdate.year)*12 + (enddate.month - startdate.month) + 1

    days = df.loc[startdate:enddate].index.size 


    if days >32:

        if ( months <= 18) :
            mticks=[1,2,3,4,5,6,7,8,9,10,11,12]


        for year in yearticks:
            if len(mticks)==0:  # in this case no month ticks, only years
                loc=xindex.get_loc(dt.datetime(year,1,1),method='nearest')
                xticks.append(loc)
                if (dt.datetime(year,1,1) > enddate):
                    break
                xlabels.append(str(year))
            for month in mticks:
                loc=xindex.get_loc(dt.datetime(year,month,1),method='nearest')
                if (dt.datetime(year,month,1) > enddate):
                    break
                if (year > first_year):
                    xticks.append(loc)
                    xlabels.append(str(year) + '-' + str(month))
                if ((year == first_year) & (month >= first_month)):
                    xticks.append(loc)
                    xlabels.append(str(year) + '-' + str(month))
    else:
        xticks = list(range(0,days))
        xlabels = [date_obj.strftime('%m%d') for date_obj in df.loc[startdate:enddate].index ] # list of strings


    return x,xindex,xticks,xlabels

def _gety_for_fmplot(df,variable,**kwargs):
    startdate = kwargs.get('startdate','')
    enddate = kwargs.get('enddate','')

    if startdate=='' :
        startdate=df.index[0]
        
    if enddate=='' :
        enddate=df.index[len(df.index)-1]

    y=df.loc[startdate:enddate,variable].reset_index()[variable].values

    return y


# US Recessions
#xhttps://en.wikipedia.org/wiki/List_of_recessions_in_the_United_States
def get_recessions():
    recessions=[(dt.datetime(1953,7,1),dt.datetime(1954,5,1)),
        (dt.datetime(1957,7,1),dt.datetime(1958,4,1)),
        (dt.datetime(1960,4,1),dt.datetime(1961,2,1)),
        (dt.datetime(1969,12,1),dt.datetime(1970,11,1)),
        (dt.datetime(1973,11,1),dt.datetime(1975,3,1)),
        (dt.datetime(1980,1,1),dt.datetime(1980,7,1)),
        (dt.datetime(1981,7,1),dt.datetime(1982,11,1)),
        (dt.datetime(1990,7,1),dt.datetime(1991,3,1)),
        (dt.datetime(2001,3,1),dt.datetime(2001,11,1)),
        (dt.datetime(2007,12,1),dt.datetime(2009,6,1)),
        (dt.datetime(2020,3,1),dt.datetime(2020,4,13))
      ]
    return recessions    
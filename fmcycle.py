import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import time


"""
.. module:: fmcycle.py
   :Python version 3.7 or greater
   :synopsis: derive financial market cycles ("Bull", "Bear") from stock data

.. moduleauthor:: Alberto Gutierrez <aljgutier@yahoo.com>

"""

def get_market_cycles(df='',compute=0,symbol='mcycle',variable='Close',mc_filename='',mcs_filename='',mcdown_p=20,mcup_p=21,initmarket=1,save=1,v=1,savedir='./'):

    """

    Derive financial market cycles ("Bull", "Bear") from stock data. 

    **Parameters**:
        **df** (datatfrme): dataframe conntaining price variable to be analyzed for market cycles

        **compute** (int): if = 1 compute market cycles from df, otherwise if = 0 read saved market cycle dataframes with filenames given by 'mc_filename' and 'mcs_filename' 

        **variable** (string): the variable (default 'Close') to be analyzed inside the dataframe df

        **mcdown_p** (numeric): The market cycle down percent (default 20), from market high point, for which to declare a down "Bear Market"

        **mcup_p** (numeric): The market cycle up pdercent (defualt 21), from market low point, for which to declare an up market "Bull Market"

        **symbol** (string): symbol name, defulat GSPC, used to formulate in the saved filenames

        **mc_filename** (int): If compute = 0, read the detailed market cycle information from the file indicated by 'mc_filename'

        **mcs_filename** (int): If compute = 0, read the summary market cycle information from the fiile indicated by 'mcs_filename'

    **Returns**:
        Returns two data frames

            **dfmcs** - market cycle summary data frame. Variables include the following:

                **mkt**  = 1 for market up cycle ("Bull"), = 0 for market down cycle ("Bear").

                **startTime** = start date of the market cycle

                **endTime** = end date of the market cycle

                **startPrice** = price of the markket variable, on startdate

                **endPrice** = price of the market variable, on enddate

                **mcnr** = market cycle normaalized return, set to zero at the beginning of the cycle

            **dfmc** - detailed market cycle dataframe. Includes all variables and dates contained in the input data frames plus additional variables as follows.

                **mkt**  = 1 for up market ("Bull"), = 0 for down market ("Bear")

                **mcupm** = the detected market cycle = 1 for up market cycle. This is delayed version of mkt. For example the market is detetected down, only after falling by mcdown_p.

                **sdm** - market switch detection marker. This is 1 on the day that the market switches from up to dowan and visa versa

                **mchlm** - the day when the market hits the high point (corresponding to up market) or low point (correspondng to down market)

                **mucdown** - when in an up market, the % that the market is below the latest detected market high. When a new market high point is found the subsequent mucdown is relative to the latest detected market high.
                
                **mdcup** - when in a down market, the % that the market is above the latest detected market low. When a new market low is found the subsequent mdcup is relative to the latest detected market low.

                **mcnr** - normalized cycle return, set to zero at the beginning of the cycle

    |

    """

    if compute == 0: # if compute = 0 then read from file
        save=0 # if compute = 0 then don't save
        print('compute ==0, read from file ...')
        #mcvariable='2021' # 2021, 2022, 2023, 2024, 2025
        print('  dfmc filename = ',mc_filename)
        print('  dfmcs filename = ',mcs_filename)
        dfmc = pd.read_csv(mc_filename,index_col=0,parse_dates=True)
        dfmcsummary = pd.read_csv(mcs_filename,index_col=0,parse_dates=True)
        print('mcdown =',mcdown_p,'mcup =',mcup_p)
        variable='Close'
    else:
        #dfmc,dfmcsummary=compute_market_cycle(df,dataStartDate,test_e,mcdown_p=mcycledown,mcup_p=mcycleup, savedir=savedir)
        (dfmc, dfmcsummary)=_market_cycles(df,initmarket,variable,mcdown_p,mcup_p,v)
    #if v==1:
    #    print(dfmcsummary.tail(10))

    #lastdate=df.index[len(df.index)-1]
 
    if save==1: # note, only save if compute == 1
        lastdate=df.index[len(df.index)-1]           # save market cycle
        startdate=df.index[0] 
        save_dfmc_filename=savedir+'/'+symbol+'_dfmc'+str(mcdown_p)+str(mcup_p)+'_'+str(startdate.year)+'_'+str(lastdate.year)+'-'+str(lastdate.month)+'-'+str(lastdate.day)+'.csv'
        save_dfmcs_filename=savedir+'/'+symbol+'_dfmcs'+str(mcdown_p)+str(mcup_p)+'_'+str(startdate.year)+'_'+str(lastdate.year)+'-'+str(lastdate.month)+'-'+str(lastdate.day)+'.csv'
        print('save dfmc file: ',save_dfmc_filename)
        print('save dfmcs file:',save_dfmcs_filename)
        dfmc.to_csv(save_dfmc_filename)
        dfmcsummary.to_csv(save_dfmcs_filename)
    return dfmc, dfmcsummary


def _market_cycles(df,initMarket,mcpricevariable,mcdown_p,mcup_p,v):

    # mcdp = mucdown/100
    # mcup = mdcup/100
    #initialize state variables

    mcdp=mcdown_p/100
    mcup=mcup_p/100
    initialMarket=initMarket
    n = 0
    if initialMarket == 1:
        muc = 1
        mdc = 0
        mcupm = 1

    elif initialMarket == -1:
        muc = 0
        mdc = 1
        mcupm = 0

    mkt = initialMarket
    # initialize data frames
    #  dfmc details
    #  dfmc summary

    dfmc = df.copy()
    newcols= ['mkt','mchlm', 'sdm', 'mcupm', 'mcnr','mucdown','mdcup','mcudthr']
    for newcol in newcols:
        dfmc[newcol]=''
    dfmc.loc[df.index[0],['mkt','mchlm','sdm']] = [mkt,0,0]

    # mchlm := market cycle high low marker
    # newmlm := new market low marker
    index=dfmc.index[0]
    #print("index = ",index)
    dfmcsummary = pd.DataFrame({'mkt': [], 'startTime': [], 'endTime': [], 'startPrice': [], 'endPrice': [], 'mcnr': []}, index=[])


    mdclowtime = df.index[0]
    mdchigh = df.iloc[0,0]
    mdchightime=0
    muclow = float(df.iloc[0,0])
    muclowtime = 0
    #muchp=0
    muchightime = 0
    switch = 0

    mdclow = float(dfmc.loc[dfmc.index[0],mcpricevariable])
    muchigh = float(dfmc.loc[dfmc.index[0],mcpricevariable])

    newhlm=0


    lastEndTime = dfmc.index[0]
    lastEndPrice = dfmc.loc[dfmc.index[0]]
    st = dfmc.index[0]
    et = dfmc.index[0]



    lastMarket = initialMarket
    first_switch=0
    n=0
    updatecols=['newmhlm','mdclp','mucdown','mdcup', 'mcupm','muc','mdc']
    for i in dfmc.index:
        #date = df.ix[i].name
        date = i
        #price = df.ix[i, 0]
        price =df.loc[i,mcpricevariable]


        (mcupm, muc,  mucdown, muclow, muclowtime, muchigh, muchightime, mdc, mdcup, mdclow, mdclowtime, mdchigh,
                    mdchightime, switch, mkt, st, et, sp, ep) =  \
                    _marketCycleLogic(n,price, date, mcupm, mcdp,mcup,
                    muc, muclow, muclowtime, muchigh, muchightime, mdc, mdclow,
                    mdclowtime, mdchigh, mdchightime)
        n=1

        #dfmc.loc[i,'newmhlm']=newmhlm
        #dfmc.loc[i, 'mcudthr'] = mcudthr
        dfmc.loc[i,'mucdown']= mucdown
        dfmc.loc[i,'mdcup']=mdcup
        dfmc.loc[i, 'mcupm'] = mcupm
        dfmc.loc[i,'mdc'] = mdc
        dfmc.loc[i,'muc'] = muc



        if switch == 1:
            d = {'mkt': [lastMarket],'startTime': [st], 'endTime': [et], 'startPrice': [sp], 'endPrice': [ep] }
            if v == 1:
                print('  ... mkt:', lastMarket,'Start:', st, 'Price:', sp,'End:', et,'Price', ep)
            lastMarket = -1 * lastMarket
            dftmp = pd.DataFrame(d, index=[st])
            lastEndTime = et
            lastEndPrice = ep
            dfmcsummary = dfmcsummary.append(dftmp,sort=False)
            dfmc.loc[st, 'mchlm'] = 1
            dfmc.loc[et, 'mchlm'] = 1

            #print('last_mcupm=', last_mcupm, 'mcupm =',mcupm)


        # fill in dfmc with
        #    sdm   - switch detection marker
        #    mchlm - market cycle high low mark

        last_mcupm=mcupm
        dfmc.loc[i,'sdm'] = switch

        # market cycle high/low mark
        if pd.isnull(dfmc.loc[i,'mchlm']):
            dfmc.loc[i, 'mchlm']=0


        #h2Price = h1Price
        #h1Price = dfmc.loc[i,mcpricevariable]

    ##### LOOP IS DONE ####
    #print(df mcsummary)
    # After the loop is complete, in most cases there will not be a switch detected at the end time,
    #   thus the tail end of the market will not be represented in the summary
    #   so, update dfmcsummary with the latest data ... include startTime, startPrice, endTime, endPrice
    if switch == 0:
        if muc == 1:
            mkt = 1
        else:
            mkt = -1
        #d = {'mkt': [mkt],'startTime': [lastEndTime], 'endTime': [date], 'startPrice': [lastEndPrice], 'endPrice': [ dfmc.loc[dfmc.index[dfmc.index.size-1], mcpricevariable] ] }
        d = {'mkt': [mkt],'startTime': [lastEndTime], 'endTime': [None], 'startPrice': [lastEndPrice], 'endPrice':  None }
        if v == 1:
            print('  ... mkt:', mkt,'Start:', lastEndTime, 'Price:', lastEndPrice,'End:', None,'Price', None )
        dftmp = pd.DataFrame(d, index = [lastEndTime])

        dfmcsummary = dfmcsummary.append(dftmp,sort=False)

    # At this point the only dfmc columns with data are
    #   S&P and sdm (switch detection point)
    #   from these all other entries can be determined
    #   fill in the market cycle columns
    if initialMarket == 1:
        lastMkt = 1
    elif initialMarket == -1:
        lastMkt = -1
    #dfmc.loc[dfmc.index[0], ['mkt']] = lastMkt
    mcStartPrice = float(dfmc.loc[dfmc.index[0], mcpricevariable])

    lastMkt = initialMarket
    for i in dfmc.index:
        dfmc.loc[i,'mkt'] = lastMkt

        mcnr=float(dfmc.loc[i, mcpricevariable]) / mcStartPrice - 1
        dfmc.loc[i,['mcnr']] = mcnr
        #
        #print(i, lastMkt,dfmc.loc[i,'mchlm'] )
        if dfmc.loc[i,'mchlm'] == 1 and lastMkt == 1:
            #     sprint(i)
            if i != dfmc.index[0]:
                lastMkt = -1
            mcStartPrice = float(dfmc.loc[i,mcpricevariable])
            #print(lastMkt)
        elif dfmc.loc[i,'mchlm'] == 1 and lastMkt == -1:
            #print(i)
            if i != dfmc.index[0]:
                lastMkt = 1
            mcStartPrice = float(dfmc.loc[i,mcpricevariable])
            #print(lastMkt)

    # Add normalized Returns to dfmcsummary


    for i in dfmcsummary.index:
        # https://docs.python.org/3/tutorial/errors.html
        try:
            dfmcsummary.loc[i, ['mcnr']] = dfmc.loc[dfmcsummary.loc[i,'endTime'], 'mcnr']
        # will get a type error on the very last call, because reality is the last day is in mid cycle
        #  and in that case the end time is "None" so the indexing will through a TypeError
        except KeyError:
            pass

    return (dfmc, dfmcsummary)


#################################################################
#  Market Cycle Logic


def _marketCycleLogic(n,price,date,mcupm, mcdp,mcup,muc,muclow,muclowtime,muchigh,muchightime,mdc,mdclow,mdclowtime,mdchigh,mdchightime):
    v = float(price)
    t = date
    switch = 0
    if muc ==1:
        mkt =1
    else:
        mkt =-1
    st=pd.NaT
    et=pd.NaT
    sp = float('nan')
    ep = float('nan')

    # mcupm := market up indicator ("buy")
    # mdc := market down cycle (1 or 0)
    # muc := market up cycle (1 or 0)
    # mcudp : market cycle up/down percent
    # mdclow := market down cycle low
    # mdchigh := high measured from market down low while in down market
    # muchp := market up percentage from last sdm switch point
    # mdclp := market down percentage from last sdm switch point

    newmhlm = 0

    mucdown=0
    mdcup=0
    if muc == 1:
        mdcup = 0
        #mdcup2 = 0
        mucdown = (muchigh - v) / muchigh
        #muchp = v/lswp - 1  # market cycle up percentage from last low point
        mcudthr = muchigh*(1-mcdp)

    if mdc == 1:
        mucdown = 0
        #mucdown2 = 0
        mdcup =  (v - mdclow) / mdclow
        #mdclp = v/lswp -1   # market cycle down percentage, from last high point ... lswp last high point
        mcudthr = mdclow*(1+mcup)

#    if (n==0 ) and (muc == 1):
#        mdclow = v
#        mdclowtime = t
#    if (n ==0) and (mdc == 1):
#        muchigh = v
#        muchightime = t

    if (n==0 ):
        mdclow = v
        mdclowtime = t
        muchigh = v
        muchightime = t


    #print('  mdc = ',mdc,'mdclow = ',mdclow , ' muc =', muc,'muchigh =', muchigh)

    if (muc == 1) and v > muchigh:
        newmhlm = 1
        mucdown = 0
        muchigh = v
        mcudthr = muchigh*(1-mcdp)  # market up cycle down threshold
        muchightime = t             # time corresponding to this down
        muclow = v                  # reset the bull low to the bull high
        muclowtime = t
        # print("    ",t, "up", "bullhigh = ",bullhigh, "bulllow =", bulllow, "adjclose =", v, "bullup =", bullup)

    elif (muc == 1) and v < muclow:
        mucdown =  (muchigh - v) / muchigh
        mcudthr = muchigh * (1 - mcdp)
        #if v < muclow:   # this conditional seems redundant ... fix later  is not needed
        muclow = v
        muclowtime = t
            # switch from up -> down
        if _mudLogic(mucdown,mcdp):
            switch = 1
            mcudthr = mdclow * (1 + mcup)
            #newmhlm = 1
            muc = 0
            mdc = 1
            mdchigh = v
            mkt = 1
            sp = mdclow
            ep = muchigh
            st = mdclowtime
            et = muchightime
            #lswp = v
            #muchp = 0
            mcupm = 0
            #print("switch = ", 1, "muc to mdc", "mdclow = ", mdclow, "muchigh= ", muclow, ", price = ", v)
            mdclow = v
            mcudthr = mdclow * (1 + mcup)
            mdclowtime = t
    elif (mdc == 1) and (v < mdclow):
        newmhlm = 1
        mdcup=0
        mdclow = v
        mcudthr=mdclow*(1+mcup)
        mkt = -1
        mdclowtime = t
        mdchigh = v
        mdchightime = t

    elif (mdc == 1) and (v > mdclow):   # the second part of this v > mdclow is not needed
        mcudthr = mdclow * (1 + mcup)
        if v > mdchigh:    # this could be moved up to the elif
            mdcup = (v - mdclow) / mdclow
            mdchigh = v
            mdchightime = t
            rbrh=mdchigh
            rbrht=mdchightime
        #### switch from down -> up ###

        if _mudLogic(mdcup,mcup) :
            # print(" bulllow = ", bulllow)
            switch = 1

            #newmhlm = 1
            muc = 1
            mdc = 0
            mdchigh = v
            mdchightime = t
            mkt = 1
            sp = muchigh
            ep = mdclow
            st = muchightime
            et = mdclowtime
            #lswp = v
            #mdclp=0
            mcupm = 1
            #print("switch = ", 1, "mdc to muc", "muchigh = ", muchigh, "mdclow = ", mdclow, ", price = ", v)
            muchigh = v
            mcudthr = muchigh * (1 - mcdp)
            muchightime = t

    return mcupm, muc, mucdown, muclow, muclowtime, muchigh, muchightime, mdc, mdcup, mdclow, mdclowtime, mdchigh, mdchightime, switch, mkt, st, et, sp, ep

def _mudLogic(mud,mudp):

    TF = False
    if mud > mudp:
        TF = True
    return TF


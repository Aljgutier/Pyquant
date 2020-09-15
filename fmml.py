import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
#from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
#from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier



"""
.. module:: finmktml.py
   :Python version 3.7 or greater
   :synopsis: Fin Market Machine Learning helpers and wrappers

.. moduleauthor:: Alberto Gutierrez <aljgutier@yahoo.com>
"""


def fmclftraintest(dfX,dfY,y, predict_s, predict_e,k=1, model='DT', traindays = 5040,trainsamples=5040,mc=0,
    smooth=3,trainall=False,posvalue=0,negvalue=1,v=1):

    """

    Financial market classifier training and test. This function makes use of the SciKit learn library
    Decision Tree Classifer, Random Forest Classifier, XGB classifiers, K Nearest Neighbor, Support Vector Machine, 
    and Logistic Regression models. 


    Args:
        dfXY(dataframe): dataframe of independent variable columns, and Y dependent variable column
        y (string): name of the dependent variable
        nday(numeric): predict ndays forward. The pridiction is for ndays forward. If nday = 1, then the prediciton is for the next day. If nday = 2 then the prediiciton is for 2 days forward and so on
        k(integer): train the model every k days
        ndayfeatures (list): if nday is a list, then for each n in nday list there is a list of features to align by n (see explanation below)
        traindays(integer): default = 5,040 (20 years). traindays will be reduced to the available days in the dataset (X, and Y) working back from test_et
        trainsamples(integer): default = 5,040 samples, combined with stratified sampling. The actual training samples could be sligtly less than trainsamples to accomodate stratified sampling.
        test_st(datetime): test start date. The first predicted date.
        test_et(datetime): test end deate. The last predicted date.
        model(string): indicates the type of ML model to use: "DT", "RF", "SVM', "XGB"
        v(integer): Verbosity level. if = 0 some initial diagnostic information is printed, if = 1 print helpful information indicating progress yearly, if = 2, print monthly.
        posvalue(integer): the integer value of a positive detection. Default 0, correpsonding to market down.
        negvalue(integer): the integer value of a negative detection. Default 1, corresponding to market up.


    Returns: 
        Training results dataframe dfTR from test_st to test_et and the classifier clf. 
        dfTR columns include the following   
        
            dfTR columns include 

            dfXY: all the columns from the dfXY input dataframe

            t: true 1 up or 0 down market indicator  

            p: predicted market indicator  

            t_1: true market indicator for the next day (t shifted back by one day)  

            p_1: predicted market indicator for the next day (p shifted back by one day)


    How does it work?   
        Example 1
        Predicting 3 days forward, nday=3. The table below focuses on Day 4. 
        Predict 3 days forward from Day 1 then Day 4 is the first prediction made
        with the Day 1, end of day data, X1. 

        +------------+-----------------+----------------+---------+---------+----------+
        |  Note      |                 |                |         |         |   Day4   | 
        +============+=================+================+=========+=========+==========+
        | EndOfDay   |  X0,t0          | X1,t1          | X2,t2   | X3,t3   |   X4,t4  |
        +------------+-----------------+----------------+---------+---------+----------+
        | EndOfDay3  | clf3=fit(X0,t3) | p4=clf3.prd(X1)|         |         |          |
        +------------+-----------------+----------------+---------+---------+----------+
        | EndOfDay4  |                 | clf4=fit(X1,t4)|         |         |          |
        +------------+-----------------+----------------+---------+---------+----------+
        | Prediction |                 |                |         |         |   p4     |
        +------------+-----------------+----------------+---------+---------+----------+



    |


    """

    #import sys

    # Decision Tree
    if model=='DT':
        ##trainndays = 400
        max_depth=8
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state= 99 )
        print("ML model type = DecisionTreeClassifier, max_depth =", max_depth)

    # Random Forest
    elif model=='RF':
        ne=100
        #max_depth=8
        #min_samples_leaf=5
        #trainndays = 1100
        clf = RandomForestClassifier(n_estimators=ne, random_state=2)

    # K Nearest Neighbor
    elif model=='KNN':
        k=5
        trainndays=500

        clf=KNeighborsClassifier(n_neighbors=k,algorithm='auto')

    # XG Boost ... https://www.datacamp.com/community/tutorials/xgboost-in-python
    elif model == 'XGB':

        #trainndays=1000
        ne=100
        clf = XGBClassifier()

    # Support Vector Machine
    elif model == 'SVM':
        trainndays = 2000

        clf = svm.SVC(kernel='rbf', C=1,gamma='auto')

    # Naive-Bayes
    elif model == 'NB':
        trainndays = 1100

        clf = GaussianNB()

    ##### Logistic Regression
    elif model == 'LR':
        trainndays = 1100
        clf = LogisticRegression()



    # output datafframe contains the X input variables
    # dfTR = X.loc[test_st:test_et, X.columns]


    # Create Trainig Results DataFrame
    # Pre loaded with the XY dataframe
    # dfTR inndex will correspondond dfXY index wtih x(t=index) corresponding with label(t_n) 
    # indexes correxpond to X training aligned with t_n



    ###########################################################
    ### Check that predict start and end are in dfXY dataframe #
    ############################################################


    id_s = dfX.index.searchsorted(predict_s)   # X (i-th row) corresponding to first prediction
    id_e = dfX.index.searchsorted(predict_e)   # X (i-th row) correspondint to last prediction


    if dfX.index[id_s] != predict_s:
        print('error: predict_s not in dfX index')
        return 
    
    if dfX.index[id_e] != predict_e:
        print('error: predict_e not in dfX index')
        return


    #############################################
    ### Create dfTR training results dataframe  #
    #############################################

    dfTR = dfX.iloc[id_s:id_e+1].copy()

    #dfTR[y] =dfY.iloc[id_s:id_e+1][y]
    #dfTR.loc[:,y]=[np.NAN]*dfTR.index.size   # add truth label col to dfTR

   
    dfTR.loc[dfY.index[id_s]:dfY.index[id_e],y] = dfY.loc[dfY.index[id_s]:dfY.index[id_e],y]
    dfTR['p_1'] = [np.NAN] * dfTR.index.size   # predictions default to NaN


    #############################################################
    # Print some information before training & prediction loop  #
    #############################################################
    print('train after every k =',k,'days')
    print('predict start date =', predict_s.strftime('%Y-%m-%d'))
    print('predict end date = ',predict_e.strftime('%Y-%m-%d'))
    print('train samples requested =', trainsamples)

    xysamplesize=dfX.iloc[0:id_s].index.size
    print('train samples available =',xysamplesize)
    print()


    ###################################################################
    ## Setup the training dataframes                                 ##
    ## Shift to align the ML features/predictors for max correlation ##
    ###################################################################


    # Seperate into Positive and Negative samples DataFrames

    ## shift the ML features 

    dfXY=dfX.join(dfY,lsuffix='x',rsuffix='y')
    dfXYp = dfXY[dfY[y] == posvalue]
    dfXYn = dfXY[dfY[y] == negvalue]




    #################################################################
    ## initialize variables for the prediction and training loop   ##
    #################################################################


    year = dfX.index[id_s].year
    month = dfX.index[id_e].month
    if trainall:
        trainsamples=xysamplesize

    kcount = 0   # train when kcount is = 0
    prev_i= dfX.index[id_s-1] # model training date concurrent with last_i



    #############################
    ## Train and Predict Loop ###
    #############################
    for i in dfTR.index:  # i corresponds to  x  index to predict t_n

        if v > 2:
            print('i=',i.strftime('%Y-%m-%d'),'last_i =',last_i.strftime('%Y-%m-%d'))

        #dfTR.loc[i,'t_n_date']=dfXY.index[i]
        dfTR.loc[i, 'xtrain_s'] = dfX.index[0]



        #try:            xysamplesize=dfXY.loc[dfXY.index[0]:last_i].index.size

        ######################################
        # train if kcount has been set to 0  #
        ######################################
        if kcount == 0:
            xysamplesize=dfX.loc[dfX.index[0]:prev_i].index.size
 


            if trainall:
                trainsamples=xysamplesize
                psamplesize= dfXYp.loc[dfXYp.index[0]:prev_i].index.size
                nsamplesize= dfXYn.loc[dfXYn.index[0]:prev_i].index.size
                samples=xysamplesize

            if (trainsamples < xysamplesize):

                ## prepare to create the training dataframe 

                # have an excess of training samples, so need to stratify sample

                # Simple Stratified Sampling .

                #psamplesize = 

                dfXYp2 = dfXYp.loc[dfXY.index[0]:prev_i]
                dfXYn2 = dfXYn.loc[dfXY.index[0]:prev_i]

                #print(dfXYn2.loc[pd.isnull(dfXYn2).any(1), :].index.values)


                samples=trainsamples
                
                psamples = psamplesize * trainsamples // (nsamplesize + psamplesize) # integer part of divsion
                nsamples = samples - psamples
                dfXYptrain = dfXYp2.sample(psamples)
                dfXYntrain = dfXYn2.sample(nsamples)


                dfXYtrain = pd.concat([dfXYptrain, dfXYntrain])  # concatenate
                dfXYtrain = dfXYtrain.sample(frac=1) # shuffle the rows

 

                dfXTrain  = dfXYtrain.drop(y,axis=1)
                dfYTrain  = dfXYtrain[y]



            else: # use all samples available

                if v > 2:
                    print("... all training samples")



                samples = xysamplesize
                psamples = psamplesize
                nsamples = nsamplesize
                
            
                dfXTrain=dfX.loc[dfX.index[0]:prev_i]    # start from nday sinde there will be NaN's before
                dfYTrain=dfY.loc[dfY.index[0]:prev_i]

                #display(dfYTrain.head(5))
                #display(dfXTrain.head(5))

            ########################
            #### Fit the Model #####
            ########################
            clf.fit(dfXTrain.values, dfYTrain.values.ravel())
            model_date=prev_i


            if v > 2:

                print('... train, prev_i =', prev_i.strftime('%Y-%m-%d'))


                #clf.fit(X2.as_matrix(), Y2.values.ravel())

#            except ValueError:
#                e = sys.exc_info()[0]
#                print("Exception occurred, ValueError")
##                print(i,last_i)
 #               #print(Y2)
 #               #print(X2)
 #               pass

        ####################
        ###### Predict #####
        ####################


        p = clf.predict(dfX.loc[i].values.reshape(1,-1))  # get a new row of data 


        if v > 2:
            print('... predict, i =', i.strftime('%Y-%m-%d') ,'p=',p[0])
            print()
        
        dfTR.loc[i, 'p_1'] = p  # prediction
        dfTR.loc[i,'model_date']=model_date  # model training date 

        #########################
        # Loop Housekeeping    ##
        #########################

        #  is it time to retrain?
        kcount += 1
        if kcount == k:
            kcount = 0

        # Print diagnostic information 
        #  yearly 
        print_flag = 0
        if v==1:
            if i.year != year:
                print_flag=1

        elif v > 1:
            if i.year != year or i.month != month:
                print_flag = 1

        if print_flag == 1:
                print(i.strftime('%Y-%m-%d'),'train samples requested =', trainsamples,'train samples available =',xysamplesize)
                print('  samples =',trainsamples, 'pos samples =', psamples,'neg samples =', nsamples)
                print('  pos samplesize =', psamplesize,'neg samplesize =', nsamplesize)
                (accuracy, precision, recall, fscore, dfcma, dfcmr, tp, fp, tn, fn)=fmclfperformance(dfTR.loc[dfXY.index[0]:i],y,'p_1',v=0)
                print('  accuracy = % 1.3f' %(accuracy))
                print('  precision (tp /(tp + fp)) = %1.3f' %(precision))
                print('  recall tp /(tp + fn) = %1.3f' %(recall))
                print('  fscore = 2*precision*recall / (precision + recall) = %1.3f' %(fscore))
                print('  tp =', tp,'fp =', fp,'tn =', tn, 'fn =', fn)

                print_flag = 0


        # house keeping, next loop variables
        year = i.year
        month = i.month
        prev_i=i  # use the last i, for training the clf ... ensures no leakage of current day prediction and label

        #### END LOOP ###


    ##############################################
    ## Put all dfTR variables back into dfXY    ##
    ## its convenient to have all in one place  ##
    ##############################################


    dfXYTR=dfXY
    cols=['p_1','model_date']
    for c in cols:
        dfXYTR.loc[dfTR.index[0]:dfTR.index[len(dfTR.index)-1],c]=dfTR.loc[dfTR.index[0]:dfTR.index[len(dfTR.index)-1],c]


    #####################################################################
    ###  Smooth with previous predictions, majority vote              ###
    ###  amount of smoothing is controlled with the input parameter   ###
    ###  ks, and placed in the "p_s" variable. See input argument     ###
    ######################################################################
#    dfXYTR['p_s']=dfXY['p_1']*2  - 1  # convert variable to be +/- 1
#    for ks in range(1,smooth):
#        dfXYTR['p_s']=dfXYTR['p_s']+dfXYTR['p_s'].shift(ks)
#    dfXYTR.loc[dfXY.p_s > 0 , 'p_s']= 1   # convert back to 0 and 1
#    dfXYTR.loc[dfXY.p_s < 0 , 'p_s']= 0

    dfXYTR['p']=dfXYTR['p_1'].shift(1)

    #dfTR.loc[test_et,'t_1']='NaN'
    #print ("Training month", i.strftime('%Y-%m-%d'), "train_st = ",train_st, 'train_st2 = ',train_st2)
    return dfXYTR, dfTR, clf

####################################################################################


def fmclfperformance(df,tcol,predcol,posvalue=0,negvalue=1,v=1):
# to-do examples
    # Market Classification Performance  

    """
    The classificaion performance for the binary classificaion (up  or down) of the financial market fund or equity. 
    The sample size for all the calculations is set to df.index.size. 

    Args:
        dfdataframe): dataframe containing classification results.
        tcol(string: namee of truth column. Make sure this column is free of Nulls or NAs.
        pred(string): name of predicted column
        posvalue(int): Positive value, default = 0. Think of this as detecting a rare event like cancer, or down market.
        negvalue(int): Negative value, default = 1, not affected, a good market "does not have cancer" 
        v(integer): Verbosity level. Defaults to 1. If = 1 print a summary of results. If = 0 do not print results.

    Returns:
        Tuple of variables as follows.    

            accuracy = (tp + tp) / (tp + fp + tn + fn)

            precision =  tp /(fp + fp)

            recall = tp / (tp + fn)

            fscore: precision * recall / ( precision + recal). The harmonic mean of precision and recall.

            dfcma =  dataframe comprised of the confusion matrix absolute numbers. 

            dfcmr =  dataframe comprised of the confusion matrix rates.

            tp = true positives

            fp = false positives

            tn = true negatives

            fn = false negatives

    |


    """

    errors  = df[df[tcol]!= df[predcol]]             # errors
    correct = df[df[tcol] == df[predcol]]            # correct
    pos = df[df[tcol] == posvalue].index.size       # positive ... affected with "with cancer" ... down market
    neg = df[df[tcol] == negvalue].index.size       # true ... up market 
    samplesize = df.index.size                      # number of samples
    nerrors = errors.index.size                     # number of errors
    ncorrect = correct.index.size                   # number of correct
    er = (nerrors)/(samplesize)                     # error rate

    #print('total =', samplesize, '\n  pos (up) = ', pos,'\n  neg (dwn)',neg)
    #print('errors =',errors,'correct = ',correct)


    tp = correct[correct[tcol] == posvalue].index.size     # true positives
    tn = correct[correct[tcol] == negvalue].index.size     # true negatives
    fn = errors[errors[predcol] == negvalue].index.size    # false negatives
    fp = errors[errors[predcol] == posvalue].index.size    # false positives


    

    # rates 
    if pos !=0:
        tpr = tp / pos
        fnr = fn / pos
    else:
        tpr = 0
        fnr = 0
    if neg != 0:
        fpr = fp / neg
        tnr = tn / neg
    else:
        fpr = 0
        tnr = 0


    if (tp + fp + tn + fn) != 0:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
    else:
        accuracy = np.NaN

    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = np.NaN

    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall = np.NaN

    if (precision >0 and recall >0):
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = np.NaN


    # Confusion Matrix Absolute Numbers
    dfcma=pd.DataFrame({'Predicted Positive':[tp,fp],'Predicted Negative':[fn,tn],'Totals':[pos, neg]}, index=['actual Positive','actual Negative'])
    # Confusion Matrix Rates
    dfcmr=pd.DataFrame({'Predicted Positive':[tpr,fpr],'Predicted Negative':[fnr,tnr],'Totals': [pos, neg]}, index=['actual Positive','actual Negative'])

    # if verbosity, v == 1 then print summary of results
    if v==1:
        display(dfcma)
        print('posvalue =',posvalue, 'negvalue =',negvalue)
        print('accuracy =',accuracy)
        print('precision (tp /(tp + fp))=',precision)
        print('recall tp /(tp + fn) =',recall)
        print('fscore = 2*precision*recall / (precision + recall) = ', fscore)
        print('tp =', tp,'fp =', fp,'tn =', tn, 'fn =', fn)
    

    # returning tuples in python
    # https://stackoverflow.com/questions/3136059/getting-one-value-from-a-tuple
    return accuracy,precision,recall,fscore,dfcma,dfcmr,tp,fp,tn,fn

#################################################################
def fmbacktest(df,price_variable,tradesignal):

# To-do ... examples,....


    # get rid of start_date and end_date ... we should be able to determine this directly from the input dataframe
    # ... note dfsp is not used so can delete 
    # .... change dfTR ... to df
    # predictor ... change this to tradesignal ... 
    # _SP := Strategy Price ("Value") ... value corresponding to following the strategy ... change this to SV (strategy Value)
    # _R  := return 

    """
    Strategy backtest receives a financial strategy up/down trade signal and backtests the financial performance
    of the strategy. It returns two dataframes one with daily trade returns and the other with a summary of trade returns
    each year.

    Args:
        df (dataframe): price_variable
        dfsp (dsfkadsl): dlsjlasdf
        price_variable (string): name of price_variable, for example, "close_price"
        tradesignal (string): the trade signal variable name
        start_date (datetime): 
        end_date (datetime): afdasf

    Returns: 
        Two dataframes with trade returns from startdate to enddate. 
        
            dftsummary: alf;sdf 

            dftreturns: af;lsf 


    |

    """


    dft=df.copy()
    dft = fmtradereturns(df, price_variable)  ### create the dft "df Trade dataframe"
    dft = fmstrategytrade(dft, price_variable, price_variable + '_R', tradesignal)

    start_date = dft.index[0]
    end_date = dft.index[len(dft.index)-1]

    start_strategy_trade=dt.datetime(start_date.year,start_date.month,start_date.day)
    end_strategy_trade=dt.datetime(end_date.year,end_date.month,end_date.day)
    startyr=start_strategy_trade.year
    endyr=dft.index[len(dft.index)-1].year

    # create empty dataframe, only containts index without columns
    dftsummary = pd.DataFrame(index=range(startyr, endyr))



    # variable name strings
    r = 'r'
    r_strategy = 'r_strategy'
    startdate="start_date"
    enddate='end_date'
    startprice='start_price'
    endprice='end_price'
    strategyvalue = price_variable + '_strategyvalue'
    startstrategyvalue = 'start_strategyvalue'
    endstrategyvalue = 'end_strategyvalue'


    last_ix = (len(dft)-1)
    # dftsummary has yearly returns 
    for year in range(startyr, endyr + 1):

        start_ix = dft.index.searchsorted(dt.datetime(year, 1, 1))
        if year != endyr:
            end_ix = dft.index.searchsorted(dt.datetime(year, 12, 31))
        else:
            end_ix = dft.index.searchsorted(dt.datetime(end_strategy_trade.year, end_strategy_trade.month, end_strategy_trade.day))

        six = start_ix
        eix = end_ix

        #if eix == last_ix:
        #    eix = end_ix - 1

        dftsummary.loc[year, startdate] = dft.index[six]
        dftsummary.loc[year, enddate] = dft.index[eix]
        dftsummary.loc[year, startprice] = dft.iloc[six][price_variable]
        dftsummary.loc[year, endprice] = dft.iloc[eix][price_variable]

        if year == startyr:
            dftsummary.loc[year, startstrategyvalue] = dft.iloc[six][price_variable]   #dft.ix[six, price_variable]
        else:
            dftsummary.loc[year, startstrategyvalue] = dft.iloc[six][strategyvalue]
        
        dftsummary.loc[year, endstrategyvalue] = dft.iloc[eix][strategyvalue]

        dftsummary.loc[year, r] = dftsummary.loc[year, endprice] / dftsummary.loc[year, startprice] - 1
        dftsummary.loc[year, r_strategy] = dftsummary.loc[year, endstrategyvalue] / dftsummary.loc[year, startstrategyvalue] - 1
       
        lastyear = year


    n = (end_strategy_trade - start_strategy_trade).days / (365)  # does this need to be changed to trade days????
    Rc = dftsummary.loc[endyr, endprice] / dftsummary.loc[startyr, startprice] - 1   # total return price_variable
    Rc_strat = dftsummary.loc[endyr, endstrategyvalue] / dftsummary.loc[
        startyr, startstrategyvalue] - 1   # total regurn strategy 

    Ra = ((Rc + 1) ** (1 / n)) - 1 # anualized return for price_variable
    Ra_strat = ((Rc_strat + 1) ** (1 / n)) - 1 # anualized return for strategy


    d={ 'nyear':n,  'Ra': Ra,'Ra_strat': Ra_strat,'Rc': Rc,'Rc_strat': Rc_strat}
    dfreturns=pd.DataFrame(d,index=[end_date])  # annualized returns


    return dftsummary, dfreturns, dft


# Trade Returns
#  computes returns
#   Input
#       DataFrame indexed by time sorted in ascending order
#       Columns must contain "P" Price Column
#           P := price column
#   Output
#       DataFrame same as input DataFrame + one additional column
#       Columns (in addition to input columns)
#           R := returns column
def fmtradereturns(df,ticker):
    """
    Trade returns function computes the returns corresponding to a market variable (e.g., equity or fund). Receives a dataframe, df, with the market
    variable "ticker" and returns the dataframe with the additional column "ticker_R"

    Args:
        df(dataframe): dataframe wiith datetime index and columns inclusive of variable ("ticker") to be evaluated  
        ticker (string): variable name corresponding to the dataframe column to be evaluated  
        returns (string): variable name of the market daily returns column corresponding to the ticker column  

    Returns: 
        The input dataframe with an additional variable "ticker_R" corresponding to daily returns

    |

    """

    tickerReturns=ticker + "_R"
    r=df[ticker] / df[ticker].shift(1) - 1
    df.loc[:,tickerReturns]=r
    df.loc[df.index[0],tickerReturns]=0
    return  df



def fmstrategytrade(df,ticker,returns,trade):
    """
    Strategy trade function evaluates a trade strategy and computes
    the value based on executing the trade or cash signal.

    Args:
        df(dataframe): dataframe wiith datetime index and columns inclusive of variable ("ticker") to be evaluated  
        ticker (string): variable name corresponding to the dataframe column to be evaluated against the tradecash signal  
        returns (string): variable name of the market returns column corresponding to the ticker column  
        trade (string): column with trade or notrade signal (1 := trade, 0 or otherwise := cash, do not trade)   

    Returns: 
        The input dataframe with an additional variable "ticker_SP" representing the value corresponding to execution of the tradecash signal  

    |

    """

    # Strategy Price Column name
    strategyvalue = ticker+"_strategyvalue"

    # New column Strategy Price
    df.loc[:, strategyvalue] = [0] * df.index.size

    # initialize holdPrice
    cashvalue = df.loc[df.index[0], ticker]

    # loop through df and apply trading strategy
    for i in df.index:
        if ((df.loc[i, trade] == 1 ) or ( np.isnan(df.loc[i, trade ]) ) ):
            # compute new strategy price
            SV=cashvalue * (1 + df.loc[i, returns])
            df.loc[i, strategyvalue] = SV
            # store new holdPrice
            cashvalue= SV
        else:
            df.loc[i, strategyvalue] = cashvalue
    # Return DF
    return df
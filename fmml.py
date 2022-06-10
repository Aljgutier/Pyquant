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
.. module:: fmml.py
   :Python version 3.7 or greater
   :synopsis: Fin Market Machine Learning helpers and wrappers

.. moduleauthor:: Alberto Gutierrez <aljgutier@yahoo.com>
"""



def mlalign(dfX, dfY, NshiftFeatures, target_variable ='y' , verbose=True):

    # get column 0 for reference "dummy variable", then drop later
    dfXaligned=pd.DataFrame(data=dfX[dfX.columns[0]],index=dfX.index)
    dfYaligned=pd.DataFrame(data=dfY[dfY.columns[0]],index=dfY.index)

    for Nshift, Features in NshiftFeatures:
        print(Nshift, Features)
        for feature in Features:
            aligned_feature = feature + '_n' + str(Nshift-1)
            dfXaligned[aligned_feature]=dfX[feature].shift(Nshift-1)

    # first column is a dummy so drop it
    dfXaligned.drop(dfX.columns[0],axis=1,inplace=True)

    # first NshiftMax rows will have nulls due to the shifting ... drop
    NshiftMax = max([k[0] for k in NshiftFeatures])
    
    dfXaligned=dfXaligned.loc[dfXaligned.index[NshiftMax]:]
    dfYaligned=dfYaligned.loc[dfYaligned.index[NshiftMax]:]

    # shift the target variable ... tomorrow to today ... today will predict tomorrow
    new_target_variable = target_variable+'_1'
    dfYaligned[new_target_variable] = dfY[target_variable].shift(-1)
    dfYaligned.drop(target_variable,axis=1,inplace=True)

    # print some diagnostic info
    if verbose == True:
        print('NshiftMax =',NshiftMax)
        print('X Features = ', list(dfXaligned.columns))
        display(dfXaligned.tail(3))
        display(dfYaligned.tail(3))

    return dfXaligned, dfYaligned



def fmclftraintest(dfX,dfY,y, predict_s, predict_e,modeltrain_ndays=1, last_training_date='', 
                   model='DT',posvalue=0,negvalue=1,clf_model = '', v=1):

    """

    Financial market classifier training and test. This function makes use of the SciKit learn library
    Decision Tree Classifer, Random Forest Classifier, XGB classifiers, K Nearest Neighbor, Support Vector Machine, 
    and Logistic Regression models. 


    Args:
        dfX(dataframe): dataframe of independent variable columns, 
        dfY(dataframe): Y dependent variable columns,
        y (string): name of the dependent variable
        predict_s: date of the first prediction. Each prediction is for the next day, for example prediction_s = dt.datetime(2020,11,2) (Monday). the prediction on 11/2, p_1, is a prediction corresponding to market performance on the next day, 11/3.
        predict_e: last date to make a market prediction. If labeled data exists up to, for example, dt.datetime(2020,11,4), then the last date possible to make a rediction is on 11/4 corresponding predicted market performance on 11/5.
        modeltrain_ndays(integer): train the model every modeltrain_ndays days. 
        model(string): indicates the type of ML model to use: "DT", "RF", "SVM', "XGB"
        v(integer): Verbosity level. if = 0 some initial diagnostic information is printed, if = 1 print helpful information indicating progress yearly, if = 2, print monthly.
        last_training_date: do not train after this date. For example, there may no longer be valid mkt truth variable past a certain date. 
        posvalue(integer): the integer value of a positive detection. Default 0, correpsonding to market down.
        negvalue(integer): the integer value of a negative detection. Default 1, corresponding to market up.


    Returns: 
        Training results dataframe dfTR from test_st to test_et and the classifier clf. 
        dfTR columns include the following   
        
            dfTR training results dataframe (see below)

            dfXYTR: all the columns from the dfXY input dataframe (see below)

            t: true 1 up or 0 down market indicator  

            p: predicted market indicator  

            t_1: true market indicator for the next day (t shifted back by one day)  

            p_1: predicted market indicator for the next day (p shifted back by one day)


    How does it work?   
        The fmcltraintest() function returns the training results dataframe dfTR, which contains the prediction p_1 (prediction one day forward). 
        Also returned is the dfXYTR data frame, which is the dfTR dataframe (prediction results) merged back into a composite dataframe containing 
        ML Features, training labels, and prediction results. The dfXYTR dataframe is useful for analyzing and studying the prediction results 
        along with the ML Feature set.

        The model is trained to forecast one day forward. The training and prediction procedure 
        works as follows. For example, suppose we want a prediction for Wednesday, January 8, 2020. Data preperation is input in dFX and dfY 
        and are aligned (outside this function) as follows.
        One set of ML Features per trading day, up to two market days (Monday, January 6) before the prediction. The market results ("labels") 
        are paired with ML Feature rows. The model is trained to predict one day forward, so the market result (label) from January 7 is paired 
        with the ML feature row on January 6. The model is trained with supervised learning to predict one day forward, up to January 7. After 
        the model is trained, the ML features on January 7 (after market close) are input to the model to create buy-sell prediction (classifier output). 
        The model output predicts, p_1, the *mkt* variable for the close of trading on Wednesday, January 8.

        For convenience, p_1 is shifted forward by 1 day and becomes the varialbe p
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
        #ne=100
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
    id_e = dfX.index.searchsorted(predict_e)   # X (i-th row) corresponding to last prediction


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
    print('train after every k =',modeltrain_ndays,'days')
    print('predict start date =', predict_s.strftime('%Y-%m-%d'))
    print('predict end date = ',predict_e.strftime('%Y-%m-%d'))
    print('model =',model)
    xysamplesize=dfX.iloc[0:id_s].index.size
    print('first training sample =',dfX.index[0].strftime('%Y-%m-%d'))
    print('train samples available =',xysamplesize)


    ###################################################################
    ## Setup the training dataframes                                 ##
    ## Shift to align the ML features/predictors for max correlation ##
    ###################################################################


    # Seperate into Positive and Negative samples DataFrames


    #################################################################
    ## initialize variables for the prediction and training loop   ##
    #################################################################

    year = dfX.index[id_s].year
    month = dfX.index[id_e].month
    trainsamples=xysamplesize
    kcount = 0   # train when kcount is = 0
    prev_i= dfX.index[id_s-1] # model training date concurrent with previous i (index)
    print(f'last training sample, first model = {prev_i.strftime("%Y-%m-%d")}, including data from first training sample {dfX.index[0].strftime("%Y-%m-%d")}')


    dfXY = dfX.join(dfY, lsuffix ='left', rsuffix = 'right')
    dfXYp = dfXY[dfXY[y]==posvalue]
    dfXYn = dfXY[dfXY[y]==negvalue]

    #############################
    ## Train and Predict Loop ###
    #############################'
    
    gt_last_train_date_flag=True
    
    do_not_train_flag=False
    if clf_model != '':
        do_not_train_flag == True
        clf = clf_model

    for i in dfTR.index:  # i corresponds to  x  index to predict t_n

        if v > 1:
            print('i=',i.strftime('%Y-%m-%d'),'last_i =',last_i.strftime('%Y-%m-%d'))

        dfTR.loc[i, 'xtrain_s'] = dfX.index[0]

        if last_training_date != '':
            if prev_i > last_training_date:
                gt_last_train_date_flag=False
        
        if kcount == 0 and gt_last_train_date_flag==True and do_not_train_flag == False:
            xysamplesize=dfX.loc[dfX.index[0]:prev_i].index.size
            trainsamples=xysamplesize
            psamplesize= dfXYp.loc[dfXYp.index[0]:prev_i].index.size
            nsamplesize= dfXYn.loc[dfXYn.index[0]:prev_i].index.size
            #samples=xysamplesize
            samples = xysamplesize
            psamples = psamplesize
            nsamples = nsamplesize



            dfXTrain=dfX.loc[dfX.index[0]:prev_i]    
            dfYTrain=dfY.loc[dfY.index[0]:prev_i]
            
            #print(prev_i)
            #display(dfYTrain.tail(1))
            


            ########################
            #### Fit the Model #####
            ########################

            
            clf.fit(dfXTrain.values, dfYTrain.values.ravel())
            model_date=prev_i


            #print('... train',prev_i.strftime('%Y-%m-%d'), 'kcount =', kcount )
            if v > 1:
                print('... train',prev_i.strftime('%Y-%m-%d'), 'kcount =', kcount )


        ####################
        ###### Predict #####
        ####################


        p_1 = clf.predict(dfX.loc[i].values.reshape(1,-1))  # get a new row of data 

        if v > 2:
            print('... predict, i =', i.strftime('%Y-%m-%d') ,'p=',p_1[0])
            print()
        
        
        #print(f'i={i} , p_1 ={p}, model_date = {model_date}')
        dfTR.loc[i, 'p_1'] = p_1  # prediction
        dfTR.loc[i,'model_date']=model_date  # model training date 


        #########################
        # Loop Housekeeping    ##
        #########################

        #  is it time to retrain?
        kcount += 1

        if kcount == modeltrain_ndays:
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
                print(i.strftime('%Y-%m-%d'),'train samples =',xysamplesize)
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

    #display(dfYTrain.tail())

    ##############################################
    ## Put all dfTR variables back into dfXY    ##
    ## its convenient to have all in one place  ##
    ##############################################


    dfXYTR=dfXY
    cols=['p_1','model_date']
    for c in cols:
        dfXYTR.loc[dfTR.index[0]:dfTR.index[len(dfTR.index)-1],c]=dfTR.loc[dfTR.index[0]:dfTR.index[len(dfTR.index)-1],c]


    dfXYTR['p']=dfXYTR['p_1'].shift(1)



    if v==1:

        ### Display the final Predictions
        display(dfTR[['p_1','y_1','model_date']].tail())

       ### Print final Stats after Loop #### 
        print(i.strftime('%Y-%m-%d'),'train samples =',xysamplesize)
        print('  samples =',trainsamples, 'pos samples =', psamples,'neg samples =', nsamples)
        print('  pos samplesize =', psamplesize,'neg samplesize =', nsamplesize)
        (accuracy, precision, recall, fscore, dfcma, dfcmr, tp, fp, tn, fn)=fmclfperformance(dfTR.loc[dfXY.index[0]:i],y,'p_1',v=0)
        print('  accuracy = % 1.3f' %(accuracy))
        print('  precision (tp /(tp + fp)) = %1.3f' %(precision))
        print('  recall tp /(tp + fn) = %1.3f' %(recall))
        print('  fscore = 2*precision*recall / (precision + recall) = %1.3f' %(fscore))
        print('  tp =', tp,'fp =', fp,'tn =', tn, 'fn =', fn)





    return dfXYTR, dfTR, clf


####################################################################################

def binarysmooth(df, y, NW = 3, y_s='', thr=0.5):
    y_s = y + '_s' if y_s == '' else y_s
    df[y_s] = df[y].rolling(NW, min_periods=1).mean()
    df.loc[df[y_s] >  thr , y_s ] = 1
    df.loc[df[y_s] <=  thr , y_s ] = 0
    return df

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
    neg = df[df[tcol] == negvalue].index.size          # true ... up market 
    p_pos = df[df[predcol] == posvalue ].index.size     # predicted positive
    p_neg = df[df[predcol] == negvalue ].index.size    # predicted positive
    samplesize = df.index.size                      # number of samples
    nerrors = errors.index.size                     # number of errors
    ncorrect = correct.index.size                   # number of correct
    er = (nerrors)/(samplesize)                     # error rate

    #print('total =', samplesize, '\n  pos (up) = ', pos,'\n  neg (dwn)',neg)
    #print('errors =',errors,'correct = ',correct)


    tp = correct[correct[tcol] == posvalue].index.size     # true positives
    tn = correct[correct[tcol] == negvalue].index.size     # true negatives
    fn = errors[errors[predcol] == negvalue].index.size    # false negatives
    fp = errors[errors[predcol] == posvalue].index.size     # false positives


    

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
    dfcma=pd.DataFrame({'Predicted Positive':[tp,fp],'Predicted Negative':[fn,tn],'Totals':[tp+fn, fp+tn]}, index=['actual Positive','actual Negative'])
    # Confusion Matrix Rates
    dfcmr=pd.DataFrame({'Predicted Positive':[tpr,fpr],'Predicted Negative':[fnr,tnr],'Totals': [pos, neg]}, index=['actual Positive','actual Negative'])

    # if verbosity, v == 1 then print summary of results
    if v==1:
        display(dfcma)
        print('posvalue = ', posvalue, 'negvalue =',negvalue)
        print('accuracy = %1.3f' %accuracy)
        print('errors = %d' %errors.index.size)
        print('total samples = %d' %df.index.size)
        print('precision (tp /(tp + fp))= %1.3f' %precision)
        print('recall tp /(tp + fn) = %1.3f' %recall)
        print('fscore = 2*precision*recall / (precision + recall) =  %1.3f' %fscore)
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

    #Ra = ((Rc + 1) ** (1 / n)) - 1 # anualized return for price_variable
    #Ra_strat = ((Rc_strat + 1) ** (1 / n)) - 1 # anualized return for strategy


    d={ 'nyear':n, 'Rc': Rc,'Rc_strat': Rc_strat}
    dfreturns=pd.DataFrame(d,index=[end_date])  # annualized returns


    return dftsummary,  dft


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
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


#reads csv file into data frame parsed by date

#materials
mat_df = pd.read_csv('XLB.csv', parse_dates=True, index_col=0)

#energy
nrg_df = pd.read_csv('XLE.csv', parse_dates=True, index_col=0)

#financials
fin_df = pd.read_csv('XLF.csv', parse_dates=True, index_col=0)

#industrial
ind_df = pd.read_csv('XLI.csv', parse_dates=True, index_col=0)

#technology
tech_df = pd.read_csv('XLK.csv', parse_dates=True, index_col=0)

#Consumer Staples
cstp_df = pd.read_csv('XLP.csv', parse_dates=True, index_col=0)

#utilities
utl_df = pd.read_csv('XLU.csv', parse_dates=True, index_col=0)

#Health Care
hc_df = pd.read_csv('XLV.csv', parse_dates=True, index_col=0)

#consumer discretionary
cdsc_df = pd.read_csv('XLY.csv', parse_dates=True, index_col=0)

#S&P
sp_df = pd.read_csv('S&P.csv', parse_dates=True, index_col=0)


#dictionary of sectors and their dataframes
sectors = {'Materials' : mat_df, 'Energy' : nrg_df, 'Financial' : fin_df,
           'Industrial' : ind_df, 'Technology' : tech_df,
           'Consumer Staples' : cstp_df, 'Health Care' : hc_df, 'Consumer Discretionary' : cdsc_df,
           'S&P' : sp_df}


#Compiles adjusted close values for all sectors into one data frame
#adjusted close is used to compensate for things like stock splits
def compile_data():
    main_df = pd.DataFrame()

    #loops through sectors stripping unused values
    for ticker in sectors.keys():
        sectors[ticker].rename(columns = {'Adj Close': ticker}, inplace=True)
        sectors[ticker].drop(['Open','High','Low', 'Close', 'Volume'],1,inplace=True)
        main_df = main_df.join(sectors[ticker], how='outer')
    return main_df


main_df = pd.DataFrame()
main_df=compile_data()


def process_data(sector):
    #number of weeks to examine over
    dl_weeks = 5
    df = main_df
    sectors = df.columns.values
    df.fillna(0, inplace=True)

    #evalutes relative change in data over the given time span in 5 day (1 trading week) intervals
    for i in range(1, dl_weeks+1):
        df['{}_{}weeks'.format(sector,i)] = (df[sector].shift(-i*5)-df[sector])/df[sector]

    df.fillna(0, inplace=True)
    return sectors, df


#Creates dataframe with percent change values for all sectors
def percent_change():
    pc_df = pd.DataFrame()
    for ticker in sectors.keys():
        pc_df = pc_df.join(main_df[ticker].pct_change(), how='outer')
        pc_df.fillna(0, inplace=True)
    return pc_df

#classifies data into buy(1), sell(-1), and hold(0)
def evaluate(*args):
    rows = [i for i in args]

    
    ##### Minimum %Change to Classify ######
    minPC = .03
    ########################################

    
    for j in rows:
        if j < -minPC:
            return -1
        if j < minPC:
            return 1
    return 0

def features(sector):
    sectors, df = process_data(sector)

    df['{}_target'.format(sector)]=list(map(evaluate,df['{}_1weeks'.format(sector)], df['{}_2weeks'.format(sector)],df['{}_3weeks'.format(sector)],df['{}_4weeks'.format(sector)],df['{}_5weeks'.format(sector)]))#,df['{}_6weeks'.format(sector)],df['{}_7weeks'.format(sector)]))
    bhs = df['{}_target'.format(sector)].values.tolist()
    choices = [str(i) for i in bhs]
    print('Distribution of Data Selections', Counter(choices))

    #############DATA NORMALIZATION###################
    #generates array of percent change values
    df_vals = df[[sector for sector in sectors]].pct_change()

    #fix holes in data if present
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    
    X = df_vals.values
    y = df['{}_target'.format(sector)].values

    return X,y,df


#Area for applying machine learning techniques
def machine(sector):
    print('##########################'+sector+'##########################')
    X,y,df = features(sector)

    #30% train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3)



    #KNN##########################################################
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)

    print('KNN Results')
    performanceTest = clf.score(X_test, y_test)
    print('Performance on test Data: ' + '%.4f' % performanceTest)
    
    predictionsTest = clf.predict(X_test)
    print('Distribution of predictions in test data: ', Counter(predictionsTest))

    performanceTrain = clf.score(X_train, y_train)
    print('Performance on train Data: ' + '%.4f' % performanceTrain)

    predictionsTrain = clf.predict(X_train)
    print('Distribution of predictions in train data: ', Counter(predictionsTrain))
    print('\n')

    print('CLASSIFICATION REPORT')
    print(classification_report(y_test, predictionsTest))


    #SVM###########################################################
    svm = tree.DecisionTreeClassifier()
    svm.fit(X_train,y_train)

    print('Decision Tree Results')
    svmperformanceTest = svm.score(X_test, y_test)
    print('Performance on test Data: ' + '%.4f' % svmperformanceTest)
    
    svmpredictionsTest = svm.predict(X_test)
    print('Distribution of predictions in test data: ', Counter(svmpredictionsTest))

    svmperformanceTrain = svm.score(X_train, y_train)
    print('Performance on train Data: ' + '%.4f' % svmperformanceTrain)

    svmpredictionsTrain = svm.predict(X_train)
    print('Distribution of predictions in train data: ', Counter(svmpredictionsTrain))
    print('\n')

    print('CLASSIFICATION REPORT')
    print(classification_report(y_test, svmpredictionsTest))

    #NaiveBayes####################################################

    gnb = GaussianNB()
    gnb.fit(X_train,y_train)

    print('Naive Bayes Results')
    gnbperformanceTest = gnb.score(X_test, y_test)
    print('Performance on test Data: ' + '%.4f' % gnbperformanceTest)
    
    gnbpredictionsTest = gnb.predict(X_test)
    print('Distribution of predictions in test data: ', Counter(gnbpredictionsTest))

    gnbperformanceTrain = gnb.score(X_train, y_train)
    print('Performance on train Data: ' + '%.4f' % gnbperformanceTrain)

    gnbpredictionsTrain = gnb.predict(X_train)
    print('Distribution of predictions in train data: ', Counter(gnbpredictionsTrain))
    print('\n')

    print('CLASSIFICATION REPORT')
    print(classification_report(y_test, gnbpredictionsTest))
    




#The passed sector is what is modeled
for sector in sectors:
    machine(sector)










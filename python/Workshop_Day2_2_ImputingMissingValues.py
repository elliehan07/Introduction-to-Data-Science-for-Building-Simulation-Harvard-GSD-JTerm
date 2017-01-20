
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# # predict missing values

# In[57]:

# import library
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[59]:

# statistical function
def MSE_MeanSquaredErrorLoss(fullData,targetData ): # common use 
    sum = RSS_ResidualSumofSquares(fullData,targetData)
    return sum / len(fullData)

def MSE_MeanSquaredErrorLossForCategorical(fullData,targetData ): # Categorical
    sum = 0.0;
    for i in range(len(fullData)):
        if(fullData[i] != targetData[i]): sum +=1
    return sum / len(fullData)

def GetMean(dataArray):
    return np.mean(dataArray)
#     theSum=0.0
#     for d in dataArray: theSum += d
#     return theSum/len(dataArray)
def GetMedian(dataArray):
    return np.median(dataArray)

def RSS_ResidualSumofSquares(dataFull, missingFill): # residual standard error, RSE ??? -- OLS (Ordinary Least Squares) Residual Sum of Squares(RSS)
    sumSoFar=0.0
    for i in range(len(missingFill)):
        sumSoFar += (dataFull[i]-missingFill[i])**2
    return sumSoFar

def TSS_TotalSumOfSquare(fullData,targetData):
    # meanVal = targetData.mean()# for numpy
    meanVal = GetMean(targetData)# manual
    sumSoFar= 0.0
    for i in range(len(targetData)):
        # sumSoFar += (fullData.y[i]-meanVal)**2
        sumSoFar += (fullData[i]-meanVal)**2
    return sumSoFar

def R_Squared_CoefficientOfDetermination(fullData, targetData): # this is for regression 
    RSS = RSS_ResidualSumofSquares(fullData, targetData)
    TSS = TSS_TotalSumOfSquare(fullData, targetData)
    return 1-(RSS/TSS)
def R_Squared_CoefficientOfDeterminationBySKLearn(fullData, targetData):
    return sk.metrics.r2_score(fullData, targetData)


# In[60]:

# utility function
def DeepCopy(d):
    return d.copy(deep=True)
def GetPandasFromCsv(path, fileName):
    path = os.path.join(path , fileName)
    return pd.read_csv(path)


# ### 1. Load data

# In[62]:

# fn1 =  'dailyChilledWaterWithFeatures'
fn2 =  'dailyElectricityWithFeatures'
# fn3 =  'dailySteamWithFeatures'
theDir = 'data/'

dTrain = GetPandasFromCsv(theDir, fn2 + "_train.csv")
dTest = GetPandasFromCsv(theDir, fn2 + "_test.csv")

index =  dTrain.columns.values
dTrain.rename(columns={index[0]:"time"},inplace=True)
dTest.rename(columns={index[0]:"time"},inplace=True)

print dTrain.head(2)
print dTest.head(2)


# ### 2. data exploration

# In[63]:

# print null values
def GetCountForNullFromPandas(d):
    return d.isnull().sum()

colTest = dTest['electricity-kWh']
colTrain = dTrain['electricity-kWh']
 
x = dTest.index.values

print x
print colTest.shape
print GetCountForNullFromPandas(colTest)
print "----------------"
print colTrain.shape
print GetCountForNullFromPandas(colTrain)


# In[64]:

colTest.describe()


# In[65]:

def VisScatterPlot(d):
    x = range(len(d))
    y = d
    plt.figure(figsize=(20,10))
    plt.scatter(x, y, color="red", alpha=0.5 )
    plt.title("data exploration")
    plt.xlabel('time')
    plt.ylabel('electricity-kWh')
    plt.tight_layout()
    plt.show()


# In[66]:

VisScatterPlot(colTest)


# In[67]:

# copy for each algorithm
colTrainByMean = DeepCopy(colTrain)
colTrainByMedian = DeepCopy(colTrain)
colTrainByKNN = DeepCopy(colTrain)
colTrainByReg = DeepCopy(colTrain)


# ### overview:  filling by mean 

# In[69]:

# clearn data by removing null data
print colTrain[:10]

dfnull = colTrain[colTrain.isnull()]
dfnull_index = dfnull.index.values

dfnotnull = colTrain[colTrain.notnull()]
dfnotnull_index = dfnotnull.index.values

print dfnotnull_index
print dfnull[:10]
print dfnotnull[:10]


# In[70]:

# prepare for the test value 
temp = dfnull.index.values
my_test =colTest[temp]
print my_test[:10]


# In[71]:

mean = GetMean(dfnotnull.tolist())
print mean


# In[73]:

mean = dfnotnull.mean()


# In[74]:

colTrainByMean[colTrainByMean.isnull()] = mean
print colTrainByMean[:10]


# ### Loss Function or Cost Function

# In[77]:

print "R Square Error:", R_Squared_CoefficientOfDetermination(colTrainByMean,colTest)


# In[78]:

print "Mean Squared Error :", MSE_MeanSquaredErrorLoss(colTrainByMean,colTest )


# ### filling by mean

# In[81]:

def fillByMean(dataFrame):
    dfnull = dataFrame[dataFrame.isnull()]
    dfnull_index = dfnull.index.values
    dfnotnull = dataFrame[dataFrame.notnull()]
    dfnotnull_index = dfnotnull.index.values
    mean = GetMean(dfnotnull.tolist())
    print "the mean of the data :", mean
    dataFrame[dataFrame.isnull()] = mean
    return dataFrame


# In[82]:

colTrainByMean = DeepCopy(colTrain)
colTrainByMean = fillByMean(colTrainByMean)
RSquareMean = R_Squared_CoefficientOfDetermination(colTrainByMean,colTest)
print "R Square :", RSquareMean


# ### filling by median

# In[83]:

def fillByMedian(dataFrame):
    dfnull = dataFrame[dataFrame.isnull()]
    dfnull_index = dfnull.index.values
    dfnotnull = dataFrame[dataFrame.notnull()]
    dfnotnull_index = dfnotnull.index.values
    median = GetMedian(dfnotnull.tolist())
    print "the median of the data :", median
    dataFrame[dataFrame.isnull()] = median
    return dataFrame


# In[84]:

colTrainByMedian = DeepCopy(colTrain)
colTrainByMedian = fillByMedian(colTrainByMedian)
RSquareMedian = R_Squared_CoefficientOfDetermination(colTrainByMedian,colTest)
print "R Square :", RSquareMedian


# ### KNN (k-nearest neighbors algorithm) 
# https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

# In[85]:

import numpy as np
import pandas as pd
import random
import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.linear_model import LinearRegression as Lin_Reg
from statsmodels.regression.linear_model import OLS
import time
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[86]:

def fillByKNN(dataFrame, k):
    dfnull = dataFrame[dataFrame.isnull()]
    dfnull_index = dfnull.index
    dfnull_index = dfnull_index.values.reshape((dfnull_index.shape[0], 1))
    dfnotnull = dataFrame[dataFrame.notnull()]
    dfnotnull_index = dfnotnull.index
    #preparing data in array form
    dfnotnull_index = dfnotnull_index.values.reshape((dfnotnull_index.shape[0], 1))
    #set data for KNN
    x_train = dfnotnull_index
    x_test = dfnull_index
    y_train = dfnotnull.tolist()
    y_test = dfnull.tolist()
    #fit model, predict and evaluate
    neighbours = KNN(n_neighbors=k)
    neighbours.fit(x_train, y_train)
    y_pred = neighbours.predict(x_test)
    r = neighbours.score(x_test, my_test)
    return y_pred, r


# In[88]:

colTrainByKNN = DeepCopy(colTrain)
k = 3
colTrainByKNN, r = fillByKNN(colTrainByKNN, k)

print len(colTrainByKNN)
print 'R^2 value of KNN fit, for k=', k, ', ', r


# In[90]:

colTrainByKNN = DeepCopy(colTrain)
rSquare = []
theMax = -100000
theMaxK = 0
for k in range(1,20):
    temp, r = fillByKNN(colTrainByKNN, k)
    print 'R^2 value of KNN fit, for k=', k , ', ', r
    rSquare.append(r)
    if theMax < r:
        theMax = r
        theMaxK = k
print "---------------------------"
print "the max K :", theMaxK , ", R:", theMax


# In[91]:

def VisSquarePlot(xData, title = 'title', YLable="YLable" , path=""):
    plt.figure(figsize=(20,10))
    yTime = range(len(xData))
    plt.plot(yTime,xData)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel(YLable)
    plt.legend(loc='best')
#     plt.axhline(13, color='r', label=r'True $\beta_1$')
#     plt.axhline(20, color='r', label=r'True $\beta_1$')
#     plt.axhspan(13, 20, facecolor='0.5', alpha=0.5,label=r'True $\beta_1$')
    if path != "":
        plt.savefig(path)
    plt.tight_layout()
    plt.show()
    return plt


# In[92]:

VisSquarePlot(rSquare)


# ### Linear regression
# https://en.wikipedia.org/wiki/Linear_regression

# In[101]:

def fillByLinReg(dataFrame):
    dfnull = dataFrame[dataFrame.isnull()]
    dfnull_index = dfnull.index
    dfnull_index = dfnull_index.values.reshape((dfnull_index.shape[0], 1))
    dfnotnull = dataFrame[dataFrame.notnull()]
    dfnotnull_index = dfnotnull.index
    #preparing data in array form
    dfnotnull_index = dfnotnull_index.values.reshape((dfnotnull_index.shape[0], 1))
    #set data for KNN
    x_train = dfnotnull_index
    x_test = dfnull_index
    y_train = dfnotnull.tolist()
    y_test = dfnull.tolist()
    #####
    regression = Lin_Reg()
    
    regression.fit(x_train, y_train)
    
    predicted_y = regression.predict(x_test)
    ####
    r = regression.score(x_test, my_test)
    
    plt.figure(figsize=(20,10))
    plt.scatter(x_train, y_train, color='red')
    plt.plot(x_test, regression.predict(x_test), color='blue',linewidth=1)
    plt.tight_layout()
    plt.show()
    return predicted_y, r


# In[102]:

colTrainByReg = DeepCopy(colTrain)
s, rReg = fillByLinReg(colTrainByReg)
print 'R^2 value =',rReg


# ### Polynomial regression
# https://en.wikipedia.org/wiki/Linear_regression

# In[103]:

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
def fillByPolynomial(dataFrame):
    dfnull = dataFrame[dataFrame.isnull()]
    dfnull_index = dfnull.index
    dfnull_index = dfnull_index.values.reshape((dfnull_index.shape[0], 1))
    dfnotnull = dataFrame[dataFrame.notnull()]
    dfnotnull_index = dfnotnull.index
    #preparing data in array form
    dfnotnull_index = dfnotnull_index.values.reshape((dfnotnull_index.shape[0], 1))
    #set data for KNN
    x_train = dfnotnull_index
    x_test = dfnull_index
    y_train = dfnotnull.tolist()
    y_test = dfnull.tolist()

    x_plot = np.linspace(0, len(x_train))
    y = y_train
    X = x_train
    lw = 2
    
    colors = ['teal', 'yellowgreen', 'gold', 'red', 'blue']
    plt.figure(figsize=(20,10))
    plt.scatter(X, y, color='navy', s=30, marker='o', label="training points")
    maxR = 0
    
    for count, degree in enumerate([1,2,4,8,12]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(X)
        plt.plot(X, y_plot, color=colors[count], linewidth=lw, label="degree %d" % degree)
        r = model.score(x_test, my_test)
        if maxR < r:
            maxR = r
        print "degree:", degree, ", R:", r
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    return maxR


# In[104]:

colTrainByPoly = DeepCopy(colTrain)
rPoly = fillByPolynomial(colTrainByPoly)


# ### Support Vector Regression (SVR) using linear and non-linear kernels
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py

# In[107]:

from sklearn.svm import SVR
def SupportVectorRegression(dataFrame):
    dfnull = dataFrame[dataFrame.isnull()]
    dfnull_index = dfnull.index
    dfnull_index = dfnull_index.values.reshape((dfnull_index.shape[0], 1))
    dfnotnull = dataFrame[dataFrame.notnull()]
    dfnotnull_index = dfnotnull.index
    #preparing data in array form
    dfnotnull_index = dfnotnull_index.values.reshape((dfnotnull_index.shape[0], 1))
    #set data for KNN
    x_train = dfnotnull_index
    x_test = dfnull_index
    y_train = dfnotnull.tolist()
    y_test = dfnull.tolist()

    y = y_train
    X = x_train

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.9)
#     svr_lin = SVR(kernel='linear', C=1e3)
#     svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
#     y_lin = svr_lin.fit(X, y).predict(X)
#     y_poly = svr_poly.fit(X, y).predict(X)

    rSqure = 0
    svr_rbfR = svr_rbf.score(x_test, my_test)
#     if rSqure < svr_rbfR:
#         rSqure = svr_rbfR
    print "SupportVectorRegression R (Kernel:rbf):", svr_rbfR
#     svr_linR = svr_lin.score(x_test, my_test)
#     if rSqure < svr_linR:
#         rSqure = svr_linR
#     print "SupportVectorRegression R (Kernel:linear):", svr_linR
#     svr_polyR = svr_poly.score(x_test, my_test)
#     print "SupportVectorRegression R (Kernel:poly):", svr_polyR
    
    lw = 2
    plt.figure(figsize=(20,10))
    plt.scatter(X, y, color='darkorange', label='data')
    plt.hold('on')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
#     plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#     plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return svr_rbfR


# In[109]:

colTrainBySVR = DeepCopy(colTrain)
rSVR = SupportVectorRegression(colTrainBySVR)


# ### 8. result 

# In[111]:

# create dic for save the R Square
dailyElectricityWithFeatures = {}
dailyElectricityWithFeatures["mean"] = RSquareMean
dailyElectricityWithFeatures["median"] = RSquareMedian
dailyElectricityWithFeatures["knn"] = theMax
dailyElectricityWithFeatures["rReg"] = rReg
dailyElectricityWithFeatures['rPoly'] = rPoly
dailyElectricityWithFeatures['rSVR'] = rSVR
print dailyElectricityWithFeatures


# # dailyChilledWaterWithFeatures.csv

# ### 1. data exploration 

# In[112]:

fn1 =  'dailyChilledWaterWithFeatures'
theDir = 'data/'

dTrain = GetPandasFromCsv(theDir, fn1 + "_train.csv")
dTest = GetPandasFromCsv(theDir, fn1 + "_test.csv")

index =  dTrain.columns.values
dTrain.rename(columns={index[0]:"time"},inplace=True)
dTest.rename(columns={index[0]:"time"},inplace=True)

indexNum = 1
index = dTrain.columns.values
print index
print index[indexNum]
colTest = dTest[index[indexNum]]
colTrain = dTrain[index[indexNum]]

dfnull = colTrain[colTrain.isnull()]
temp = dfnull.index.values
my_test =colTest[temp]


# In[113]:

VisScatterPlot(colTest)


# ### 2. by mean

# In[114]:

colTrainByMean = DeepCopy(colTrain)
colTrainByMean = fillByMean(colTrainByMean)
RSquareMean = R_Squared_CoefficientOfDetermination(colTrainByMean,colTest)
print "R Square :", RSquareMean


# ### 3. by median

# In[115]:

colTrainByMedian = DeepCopy(colTrain)
colTrainByMedian = fillByMedian(colTrainByMedian)
RSquareMedian = R_Squared_CoefficientOfDetermination(colTrainByMedian,colTest)
print "R Square :", RSquareMedian


# ### 4. by KNN

# In[116]:

colTrainByKNN = DeepCopy(colTrain)
rSquare = []
theMax = -100000
theMaxK = 0
for k in range(1,20):
    temp, r = fillByKNN(colTrainByKNN, k)
    print 'R^2 value of KNN fit, for k=', k , ', ', r
    rSquare.append(r)
    if theMax < r:
        theMax = r
        theMaxK = k
print "---------------------------"
print "the max K :", theMaxK , ", R:",theMax


# In[117]:

VisSquarePlot(rSquare)


# ### 5. Linear regression

# In[118]:

colTrainByReg = DeepCopy(colTrain)
s, rReg = fillByLinReg(colTrainByReg)
print 'R^2 value =', rReg


# ### 6. Polynomial regression

# In[119]:

colTrainByPoly = DeepCopy(colTrain)
rPoly = fillByPolynomial(colTrainByPoly)


# ### Support Vector Regression (SVR) using linear and non-linear kernels

# In[121]:

colTrainBySVR = DeepCopy(colTrain)
rSVR = SupportVectorRegression(colTrainBySVR)


# ### 8. result 

# In[122]:

dailyChilledWaterWithFeatures = {}
dailyChilledWaterWithFeatures["mean"] = RSquareMean
dailyChilledWaterWithFeatures["median"] = RSquareMedian
dailyChilledWaterWithFeatures["knn"] = theMax
dailyChilledWaterWithFeatures["rReg"] = rReg
dailyChilledWaterWithFeatures['rPoly'] = rPoly
dailyChilledWaterWithFeatures['rSVR'] = rSVR
print dailyChilledWaterWithFeatures


# # dailySteamWithFeatures.csv

# ### 1. data exploration 

# In[123]:

fn3 =  'dailySteamWithFeatures'
theDir = 'data/'

dTrain = GetPandasFromCsv(theDir, fn3 + "_train.csv")
dTest = GetPandasFromCsv(theDir, fn3 + "_test.csv")

index =  dTrain.columns.values
dTrain.rename(columns={index[0]:"time"},inplace=True)
dTest.rename(columns={index[0]:"time"},inplace=True)

indexNum = 1
index = dTrain.columns.values
print index
print index[indexNum]
colTest = dTest[index[indexNum]]
colTrain = dTrain[index[indexNum]]

dfnull = colTrain[colTrain.isnull()]
temp = dfnull.index.values
my_test =colTest[temp]


# In[124]:

VisScatterPlot(colTest)


# ### 2. by mean

# In[125]:

colTrainByMean = DeepCopy(colTrain)
colTrainByMean = fillByMean(colTrainByMean)
RSquareMean = R_Squared_CoefficientOfDetermination(colTrainByMean,colTest)
print "R Square :", RSquareMean


# ### 3. by median

# In[127]:

colTrainByMedian = DeepCopy(colTrain)
colTrainByMedian = fillByMedian(colTrainByMedian)
RSquareMedian = R_Squared_CoefficientOfDetermination(colTrainByMedian,colTest)
print "R Square :", RSquareMedian


# ### 4. by KNN

# In[129]:

colTrainByKNN = DeepCopy(colTrain)
rSquare = []
theMax = -100000
theMaxK = 0
for k in range(1,20):
    temp, r = fillByKNN(colTrainByKNN, k)
    print 'R^2 value of KNN fit, for k=', k , ', ', r
    rSquare.append(r)
    if theMax < r:
        theMax = r
        theMaxK = k
print "---------------------------"
print "the max K :", theMaxK , ", R:",theMax


# In[130]:

VisSquarePlot(rSquare)


# ### 5. Linear regression

# In[131]:

colTrainByReg = DeepCopy(colTrain)
s, rReg = fillByLinReg(colTrainByReg)
print 'R^2 value =', rReg


# ### 6. Polynomial regression

# In[132]:

colTrainByPoly = DeepCopy(colTrain)
rPoly = fillByPolynomial(colTrainByPoly)


# ### 7. Support Vector Regression (SVR) using linear and non-linear kernels

# In[133]:

colTrainBySVR = DeepCopy(colTrain)
rSVR = SupportVectorRegression(colTrainBySVR)
print rSVR


# ### 8. result 

# In[134]:

dailySteamWithFeatures = {}
dailySteamWithFeatures["mean"] = RSquareMean
dailySteamWithFeatures["median"] = RSquareMedian
dailySteamWithFeatures["knn"] = theMax
dailySteamWithFeatures["rReg"] = rReg
dailySteamWithFeatures['rPoly'] = rPoly
dailySteamWithFeatures['rSVR'] = rSVR
print dailySteamWithFeatures


# # Conclusion

# In[135]:

print dailyElectricityWithFeatures
print dailyChilledWaterWithFeatures
print dailySteamWithFeatures
resultList = [dailyElectricityWithFeatures,dailyChilledWaterWithFeatures,dailySteamWithFeatures]


# In[136]:

index = ['dailyElectricity', 'dailyChilledWater', 'dailySteam']
resultDF = pd.DataFrame(resultList, index=index)
resultDF.T


# In[ ]:




# In[ ]:




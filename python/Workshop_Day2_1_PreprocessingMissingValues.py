
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# # missing values

# In[1]:

# import library
import numpy as np
import pandas as pd
import os


# In[2]:

# import data
def GetPandasFromExcel(path, fileName):
    path = os.path.join(path , fileName)
    return pd.read_excel(path) # pip install xlrd needed


# In[3]:

fn1 =  'dailyChilledWaterWithFeatures.xlsx'
fn2 =  'dailyElectricityWithFeatures.xlsx'
fn3 =  'dailySteamWithFeatures.xlsx'
theDir = 'data/'
d1 = GetPandasFromExcel(theDir, fn1)
d2 = GetPandasFromExcel(theDir, fn2)
d3 = GetPandasFromExcel(theDir, fn3)
print d1.head(2)
print d2.head(2)
print d3.head(2)


# In[4]:

# get index string
index1 = d1.columns.values[0]
index2 = d2.columns.values[0]
index3 = d3.columns.values[0]
print index1
print index2
print index3


# In[5]:

# get the column
c1 = d1[index1] #['chilledWater-']
c2 = d2[index2] #['electricity-kWh']
c3 = d3[index3] #['steam-LBS']
print c1
print c2
print c3


# In[29]:

def InsertNoneByRandom(data, percent): # 0.7
    # percent = 1.0 - percent
    newDF = data.copy(deep=True)
    dataLength = len(data)
    rndIndex  = np.random.choice(dataLength,int(dataLength * percent) )
    newDF.iloc[rndIndex] = np.nan# 'null'
    print "The total length of old data:", dataLength
    print "The total length of new data:", newDF.count() 
    print "the length of None in the new data:", newDF.isnull().sum(),"(",percent,"%)"
    return newDF


# In[30]:


c1Train = InsertNoneByRandom(c1, 0.3)
print "-----------------------------"
c2Train = InsertNoneByRandom(c2, 0.3)
print "-----------------------------"
c3Train = InsertNoneByRandom(c3, 0.3)


# In[31]:

print c1Train.head(10)
print c2Train.head(10)
print c3Train.head(10)


# In[32]:

# sve dat for test
fn1 =  'dailyChilledWaterWithFeatures.xlsx'
fn2 =  'dailyElectricityWithFeatures.xlsx'
fn3 =  'dailySteamWithFeatures.xlsx'

d1.to_csv("data/" + fn1[:-5] + "_test." + "csv", header=True)
d2.to_csv("data/" + fn2[:-5] + "_test." + "csv", header=True)
d3.to_csv("data/" + fn3[:-5] + "_test." + "csv", header=True)


# In[33]:

# save data for train
d1[index1] = c1Train#['chilledWater-']
d2[index2] = c2Train#['electricity-kWh']
d3[index3] = c3Train#['steam-LBS']

d1.to_csv("data/" + fn1[:-5] + "_train." + "csv", header=True)
d2.to_csv("data/" + fn2[:-5] + "_train." + "csv", header=True)
d3.to_csv("data/" + fn3[:-5] + "_train." + "csv", header=True)


# In[ ]:




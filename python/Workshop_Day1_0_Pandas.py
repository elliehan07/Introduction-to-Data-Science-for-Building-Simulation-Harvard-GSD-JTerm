
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# # Anaconda  
# ***
# Python is a manifold scripting language for data management and analysis, however managing a Python project environment can be nuanced and tricky. Anaconda is a platform built to complement Python by providing or producing customizable and easily accessible environments in which you can run Python scripts with diverse libraries. 
# 
# For references, the Anaconda homepage is found at the following address.
# 
# https://www.continuum.io/why-anaconda
# 
# For the course, Anaconda 4.2.0(Python 2.7 version[64 bit]) is needed. 
# 
# https://www.continuum.io/downloads
# ***
# 
# # Jupyter Notebook
# ***
# The Jupyter Notebook is a web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, machine learning and much more.
# 
# http://jupyter.org/
# 
# This is our IDE(Integrated Development Environment) in this course. There are two ways to execute it, one is to use anaconda and other is to type "jupyter notebook" in the command window for Window or in terminal for Mac.
# 
# 
# ***
# 
# # Pandas
# 
# ***
# Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. It gives a tons of useful functions for data manipulation and analysis in easy and quick ways.
# 
# http://pandas.pydata.org/ 
# 
# Anaconda has the pandas library as a default, so that we try to take an advantage of the fantastic algolithm and the 

# 
#  ## Basic Pandas workshop

# In[1]:

# import essential library
import pandas as pd
import numpy as np
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# ## Overview

# In[2]:


import math
print math.sqrt(4)


# In[3]:

start = datetime.datetime(2010, 1,1)
end = datetime.datetime(2017, 1, 5)

df = web.DataReader('XOM', 'yahoo', start, end)
print df.head()
print "---------------------------------------"
print df.describe()
df["Adj Close"].plot()
plt.show()


# ## 1 Create Data Frame
# * to create pandas data frame from dictionary in python
# * to create pandas data frame from csv

# In[4]:

# pandas data frame from dictionary in python
data = { "name" : ['a','b','c','d','e','f'],
         "id" : [1,3,8,10,12,60],
         "value" : [23,21,18,29,43,21]}

print type(data)
print data

df = pd.DataFrame(data)
print type(df)
print df
df = df.set_index("id")
print df
df.plot()
plt.show()
df.describe()

print df['value']
print df.value


# In[5]:

# your codes here

a =['a','b','c','d','e','f']
b = [1,3,8,10,12,60]




# In[6]:

# pandas data frames from list of list
listOfList = [[1,2,3],
             [4,5,6],
             [7,8,9]]

dfFromList = pd.DataFrame(listOfList)
print dfFromList


lstA = [1,2,3]
lstB = [4,5,6]
lstC = [7,8,9]
listOfList2 = []
listOfList2.append(lstA)
listOfList2.append(lstB)
listOfList2.append(lstC)

dfFromList = pd.DataFrame(listOfList2)
dfFromList.columns = ["ID", "valA", 'valB']
print dfFromList


# In[7]:

# your codes here






# In[8]:

# pandas data frame from csv
def GetPandasFromFileCSV(path):
    return pd.read_csv(path, delimiter=',')
def GetPandasFromFileCSVWithoutCol(path):
    return pd.read_csv(path, delimiter=',', index_col=0)

path = "data/Baseline_sunday_designbuilder.csv"
df = GetPandasFromFileCSV(path)
print df.head(5)
print df.tail(3)


# In[9]:

# your codes here






# ### 2. index and columns, and converting data
# * convert data frame to list 
# * convert data frame to numpy
# * convert numpy or list to data frame
# 
# * manipulation columns of data frame
# * manipulation row of data frame

# In[10]:

print "the index :", df.index
print "------------------------------"
print "the index :", df.index.values
print "------------------------------"
print 'the columns', df.columns.values
print "------------------------------"
print df['Air Temperature']
print df[['Date/Time','Radiant Temperature']]


# In[11]:

# your codes here


time = df['Date/Time'].tolist()
temp = df['Air Temperature']

print time


# In[12]:

# import datetime
# from dateutil.parser import parse
# import matplotlib
# import matplotlib.pyplot as plt
# from __future__ import print_function
# import datetime
# import matplotlib.pyplot as plt
# from matplotlib.dates import MONDAY
# from matplotlib.finance import quotes_historical_yahoo_ochl
# from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter

# # plt.plot(time,temp)
# # plt.show()


# According to the functions like plot or data analysis, we need to shift the data structure, for example, basically Pandas data frames is a two dimensional matrix, so, sometimes, one dimensional data stricture such as 1D List or 1D Array is needed to match the input data of certain functions.

# In[13]:

# to List
airList = df['Air Temperature'].tolist()
print type(airList)
print len(airList)
print "------------------"
print airList


# In[14]:

# to Numpy array
npArray = np.array(df['Air Temperature'])
print type(npArray)
print npArray


# In[15]:

# to Pandas Data Frame
temp = df[['Date/Time', 'Sensible Cooling', 'Zone Sensible Cooling']]
newNp = np.array(temp)
print newNp
# print newNp[5:10]
newDF = pd.DataFrame(temp)
print newDF


# In[16]:

# rename columns 
print newDF.head(3)
newDF.columns = ["date", "SCZ", 'SC']
print newDF.head(3)
newDF.rename(columns={"date" : 'day_time' }) # renmae selectively
print newDF.head(3)


# In[17]:

# get data from selected columns
dataCol = newDF['SC'].values
print dataCol

dataCol = newDF['SC']
print dataCol


# In[18]:

# slice data between start and end index 
dataCol = newDF['SC']
# print dataCol
print dataCol[5:20]


# In[19]:

# data change by index
dataCol[9] = 9999999999 
dataCol.iloc[6] = 8888888888 # for 2D dataCol.iloc[:, 9]
print dataCol[5:20]


# In[20]:

data = { "name" : ['a','b','c','d','e','f'],
         "id" : [1,3,8,10,12,60],
         "value" : [23,21,18,29,43,21]}
df = pd.DataFrame(data)
print df.describe()
print "---------------------------------"
print df
print "the sum of value column :", df['value'].sum()
print "the standard deviation of value column :", df['value'].std()
print "the mean of value column :", df['value'].mean()
print "the min of value column :", df['value'].min()
print "the max of value column :", df['value'].max()


# In[21]:

print "the sum of value column :", df[['id', 'value']].sum()
print "the standard deviation of value column :", df[['id', 'value']].std()
print "the mean of value column :", df[['id', 'value']].mean()
print "the min of value column :", df[['id', 'value']].min()
print "the max of value column :", df[['id', 'value']].max()


# In[22]:

# sort 
sortedData = df.sort(['value'], ascending=[True])
print sortedData


# In[23]:

# transpose
print sortedData.T
print "-------------------"
print sortedData.T.sort([1], ascending=False)
print "------------------"
print ord('b') # b is considered as 98 (ASCII value)
print chr(98)


# In[26]:

# conditional statement
print df
print "---------------------"
print "---------------------"
print df['value'] > 30
print "---------------------"
print df[df['value'] > 20]
print "---------------------"
print df[(df["value"] > 10) & (df['value'] < 20) ]
print "---------------------"
print df[ (df["value"] > 10) & (df['id'] > 10) ]


# ### 3. save data
# * save data as csv
# * save data as HTML

# In[27]:

# save data frame as csv
newDF.to_csv("data/new_data.csv", header=True)
newDF.to_html("data/new_data.html", header=True)


# ### 4. manipulating data frames

# In[26]:

theIndex = [2017, 2016, 2015, 2014]
dict1 = {'MA':[80,85,73,99],
       'rate':[5,8,3,5],
       'value':[33,23,42,28]}
dict2 = {'NY':[67,88,76,89],
       'rate':[7,9,6,4],
       'value':[35,53,45,25]}
dict3 = {'CA':[20,43,53,56],
       'rate':[7,4,9,9],
       'value':[13,33,43,38]}
df1 = pd.DataFrame(dict1, index=theIndex)
df2 = pd.DataFrame(dict2, index=theIndex)
df3 = pd.DataFrame(dict3, index=theIndex)
print df1
print df2
print df3


# In[27]:

concat = pd.concat([df1, df2, df3])
print concat.shape
print concat


# In[28]:

df4 = df1.append(df2)
print df4


# In[ ]:




# In[ ]:




# ### Advanced

# In[28]:

#pip install html5lib needed
bosTempatue = pd.read_html("http://www.usclimatedata.com/climate/boston/massachusetts/united-states/usma0046")


# In[29]:

print bosTempatue


# In[30]:

print type(bosTempatue)

theIndex = bosTempatue[0].T[:1]
df1 = bosTempatue[0].T[1:]
df2 =  bosTempatue[1].T[1:]

theIndex = np.array(theIndex)[0]
theIndex[0] = "Month"
print theIndex
result = pd.DataFrame(df1, index=theIndex)
result = pd.concat([df1, df2], ignore_index=True)
result.columns = np.array(theIndex)

result.T
print result


# In[31]:

resultSorted = result.sort(['Av. precipitation in :'], ascending=[False])
print resultSorted


# In[32]:

result.to_csv("data/BostonWeatherFromWeb.csv", header=True)


# # reference

# Pandas documentation:
# 
# http://pandas.pydata.org/pandas-docs/stable/
# 
# Binary Installers: 
# 
# http://pypi.python.org/pypi/pandas
# 
# Source Repository: 
# 
# http://github.com/pydata/pandas
# 
# Issues & Ideas: 
# 
# https://github.com/pydata/pandas/issues
# 
# Q&A Support: 
# 
# http://stackoverflow.com/questions/tagged/pandas
# 
# Developer Mailing List: 
# 
# http://groups.google.com/group/pydata
# 
# Pandas IO Tools:
# 
# http://pandas.pydata.org/pandas-docs/stable/io.html


# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os


# In[2]:

def getPandasFromFile(path, fileName, theSkipRow):
    path = os.path.join(path , fileName)
    return pd.read_csv(path, skiprows= theSkipRow , header=None)

def SavePandasToCSV(d, path):
    d.to_csv(path)
    return "done!!"


# In[3]:

def getPandasFromFile_my(path, fileName):
    path = os.path.join(path , fileName)
    return path 


# In[4]:

dd = getPandasFromFile_my("",'USA_MA_Boston-Logan.Intl.AP.725090_TMY3.epw')
print dd


# In[5]:

d = getPandasFromFile("",'USA_MA_Boston-Logan.Intl.AP.725090_TMY3.epw', 8)
print d.head(3)
print d.shape


# In[6]:

d.columns.values


# In[7]:

df = d.iloc[:5,:10]
print df


# In[8]:

df = d.iloc[:,0:32]
print df.shape


# In[9]:

theDf =df.rename(index=str, columns={0:"year", 1: "Month", 2: "Day", 3:"Hour",4:"Minute",6:"DB_temp",
                                     7:"Dew_Point",
                                     8:"RH",9:"P", 10:"Horiz_Rad",11:"Normal_Rad",12:"Sky_Rad",
                                     13:"G_Horiz_Rad",14:"Dir_Normal_Rad",15:"Diff_Horiz_Rad",
                                     16:"G_Horiz_Illu",17:"Dir_Normal_Illu",18:"Diff_Horiz_Illu",
                                     19:"Zenith_Illu",20:"Wind_Direction",21:"Wind_Speed",
                                     22:"Total_Sky_Cov",23:"Opaque_Sky_Cov",24:"Visibility",
                                     25:"field_Ceiling_H",26:"Whtr_Observ",27:"Whtr_Codes",
                                     28:"Pred_Water",29:"AeroesOptical_D",30:"Snow_Depth",
                                     31:"Days_Since_Snow"
                                    })


# In[10]:

theDf.columns.values


# In[11]:

NewDf = theDf[['DB_temp',"Dew_Point",
               'RH','P','Normal_Rad', 'Sky_Rad','Wind_Direction', 'Wind_Speed'
              ]]
NewDf.head()


# In[12]:

NewDf.describe()


# In[13]:

SavePandasToCSV(NewDf, "dd.csv")


# In[14]:

import datetime
from dateutil.parser import parse
import matplotlib
import matplotlib.pyplot as plt
from __future__ import print_function
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import MONDAY
from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter


# In[15]:

x = NewDf.index
temp = NewDf["DB_temp"]

plt.figure(figsize=(20,10))
myTime = pd.date_range('1/1/2016', periods=8760, freq='H')


# plt.ylim((20,100) )

plt.plot(myTime,temp)
plt.title('NV potential days')
plt.xlabel('Year')
plt.ylabel('Temperature (F)')
plt.legend(loc='best')

plt.axhline(13, color='r', label=r'True $\beta_1$')
plt.axhline(20, color='r', label=r'True $\beta_1$')
plt.axhspan(13, 20, facecolor='0.5', alpha=0.5,label=r'True $\beta_1$')

plt.show()


# In[16]:

x = NewDf.index
RH = NewDf["RH"]

plt.figure(figsize=(20,10))
# # plot
plt.hist(RH, bins =36, color ="red", alpha = 0.5)
plt.show()


# In[ ]:




# In[ ]:




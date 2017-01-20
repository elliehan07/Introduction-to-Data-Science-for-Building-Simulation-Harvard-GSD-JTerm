
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# In[ ]:

import pandas as pd
import datetime
from datetime import timedelta
import time
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# # Single File

# In[ ]:

Result = pd.read_csv('data/EnergyplusSimulationData.csv')


# In[ ]:

# print Result.head()
# print Result.shape


# In[ ]:

# Getting Data Information 
Result.columns.values[:20]


# In[ ]:

Result['Date/Time'].tail()


# # Index : order of time 

# In[ ]:

#Created by Clayton Miller (miller.clayton@arch.ethz.ch)
#Function to convert timestamps

def eplustimestamp(simdata):
    timestampdict={}
    for i,row in simdata.T.iteritems():
        timestamp = str(2013) + row['Date/Time']
        try:
            timestampdict[i] = datetime.datetime.strptime(timestamp,'%Y %m/%d  %H:%M:%S')
        except ValueError:
            tempts = timestamp.replace(' 24', ' 23')
            timestampdict[i] = datetime.datetime.strptime(tempts,'%Y %m/%d  %H:%M:%S')
            timestampdict[i] += timedelta(hours=1)
    timestampseries = pd.Series(timestampdict)
    return timestampseries


# In[ ]:

Result.index = eplustimestamp(Result)


# In[ ]:

Result['Date/Time'].tail()


# # Column selection 

# In[ ]:

ColumnsList = pd.Series(Result.columns)
# print ColumnsList.head()
# print ColumnsList.head(10)
# print ColumnsList.head(50)


# In[ ]:

# ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)")[:50]


# In[ ]:

# ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))]


# In[ ]:

ZoneTempPointList = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))])
# print ZoneTempPointList


# In[ ]:

# (ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("U1"))


# In[ ]:

ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("U1"))]


# In[ ]:

BasementZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("U1"))])
GroundFloorZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("00"))])
Floor1ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("01"))])
Floor2ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("02"))])
Floor3ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("03"))])
Floor4ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("04"))])


# In[ ]:

# BasementZoneTemp


# In[ ]:

# GroundFloorZoneTemp


# In[ ]:

# Floor1ZoneTemp


# In[ ]:

ZoneTemp = Result[ZoneTempPointList]#.drop(['EMS:All Zones Total Heating Energy {J}(Hourly)'],axis=1)


# # Visualization

# In[ ]:

# ZoneTemp.plot(figsize=(50,25))


# In[ ]:

# ZoneTemp[BasementZoneTemp].plot(figsize=(20,10))

# plt.title("Temperature per Zone")
# plt.xlabel('Time')
# plt.ylabel("Temperature")

# plt.tight_layout()
# plt.show()


# In[ ]:

ZoneTemp[GroundFloorZoneTemp].plot(figsize=(20,10))

plt.title("Temperature per Zone")
plt.xlabel('Time')
plt.ylabel("Temperature")

# plt.axhline(18, color='r', label=r'True $\beta_1$')
# plt.axhline(24, color='r', label=r'True $\beta_1$')
# plt.axhspan(18, 24, facecolor='y', alpha=0.2,label=r'True $\beta_1$')

plt.tight_layout()
plt.show()


# # Zooming

# In[ ]:

ZoneTemp[Floor1ZoneTemp].plot(figsize=(20,10))

plt.axhspan(18, 24, facecolor='y', alpha=0.2,label=r'True $\beta_1$')

plt.tight_layout()
plt.show()


# In[ ]:

# ZoneTemp[Floor2ZoneTemp].truncate(before='2013-02-01',after='2013-02-14')


# In[ ]:

# ZoneTemp[Floor2ZoneTemp].truncate(before='2013-02-01',after='2013-02-14').plot(figsize=(20,10))

# plt.tight_layout()
# plt.show()


# # Multiple Files

# In[ ]:

from os import walk
import numpy as np

mypath = 'data'
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
    
print f  
f = f[1:]
print f


# In[ ]:

dfs = []
for i in f:
    fn = mypath + '/' + i
    dfs.append(pd.read_csv(fn))


# In[ ]:

# len(dfs)


# In[ ]:

# dfs[0]


# In[ ]:

dfsNames=dfs[0].columns.values[1:]


# In[ ]:

def VisBarPlotByAX(ax, xData, width=0.5, offset=0, color = "purple", title = 'title', YLable="YLable" , path="", axLine1 = 0, axLine2 = 0,label='Heating' ):
    if(offset !=0):
        yTime = [i+offset for i in range(0,len(xData))]     
    else:
        yTime = [i for i in range(0,len(xData))]

    ax.bar(yTime,xData,width=width, alpha=0.2, color=color,label=label)
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(YLable)
    ax.legend(loc='best')
    if(axLine1 != 0 and axLine2 != 0 ):
        ax.set_axhline(axLine1, color='r', label=r'True $\beta_1$')
        ax.set_axhline(axLine2, color='r', label=r'True $\beta_1$')
        ax.set_axhspan(axLine1, axLine2, facecolor='0.5', alpha=0.5,label=r'True $\beta_1$')


# In[ ]:

bar = dfs[0]
bar1 = bar['Heating:DistrictHeating [J](Monthly)']
bar2 = bar['Cooling:DistrictCooling [J](Monthly) ']

ind=np.arange(12)
width =0.35

path = "barChartPlot"
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1)

VisBarPlotByAX(ax, bar1, width = 0.3, color="red", YLable="Electricity",label='Heating')
VisBarPlotByAX(ax, bar2, width=0.3,offset=0.3, color="blue", YLable="Electricity",label='Cooling')


plt.savefig(path)
plt.tight_layout()
plt.show()


# In[ ]:

def VisStackBarPlotByAX(ax, xData, width=0.5, bottom="", color = "purple", title = 'title', YLable="YLable" , path="", axLine1 = 0, axLine2 = 0 ,label='Heating'):

    yTime = [i for i in range(0,len(xData))]

    try:
        if(bottom == ""):
            ax.bar(yTime,xData,width=width, alpha=0.2, color=color,label=label)
    except:
        ax.bar(yTime,xData, bottom=bottom, width=width, alpha=0.2, color=color,label=label)
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(YLable)
    ax.legend(loc='best')
    if(axLine1 != 0 and axLine2 != 0 ):
        ax.set_axhline(axLine1, color='r', label=r'True $\beta_1$')
        ax.set_axhline(axLine2, color='r', label=r'True $\beta_1$')
        ax.set_axhspan(axLine1, axLine2, facecolor='0.5', alpha=0.5,label=r'True $\beta_1$')


# In[ ]:

# bar = dfs[0]
# bar1 = bar['Heating:DistrictHeating [J](Monthly)']
# bar2 = bar['Cooling:DistrictCooling [J](Monthly) ']

# ind=np.arange(12)
# width =0.35

# path = "barChartPlot"
# fig = plt.figure(figsize=(15, 8))
# ax = fig.add_subplot(1, 1, 1)
# theWidth = 0.5

# VisStackBarPlotByAX(ax, bar1, width = theWidth, color="red", YLable="Electricity",label="heating")
# VisStackBarPlotByAX(ax, bar2, bottom=bar1, width=theWidth, color="blue", YLable="Electricity",label="cooling")

# plt.savefig(path)
# plt.tight_layout()
# plt.show()


# # Transpose DataFrame

# In[ ]:

temp = dfs[0]
my_temp = temp.T
my_temp


# In[ ]:

my_temp = my_temp.drop("Date/Time",0)
my_temp


# In[ ]:

d1 = my_temp[0]
print d1
index = d1.index

ytime = [1,2,3,4,5,6]
labels = ["Lighting","Equipment","Fans","pumps","Heating","Cooling"]
explode = (0.1, 0.1, 0.2, 0.2,0.1,0.1) 
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral',"coral","tan"]


plt.bar(ytime,list(d1),alpha=0.8, color=colors,)
plt.xticks(ytime, labels)
plt.plot()


# In[ ]:

d1 = my_temp[0]
print d1.index

labels = ["Lighting","Equipment","Fans","pumps","Heating","Cooling"]
explode = (0.1, 0.1, 0.2, 0.2,0.1,0.1) 
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral',"coral","tan"]

plt.pie(d1, explode=explode,labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


# In[ ]:

# fig = plt.figure(figsize=(15, 8))
# fig, axs = plt.subplots(len(dfs)/2, 2, figsize=(20, 40))
# count =0;
# for i in range(0,len(dfs)/2):
#     for j in range(2):
#         temp = dfs[count]
#         my_temp = temp.T
#         # print my_temp.drop("Date/Time",0)
#         my_temp=my_temp.drop("Date/Time",0)
#         d1 = my_temp[0]
        
# #         labels = d1.index
#         explode = (0, 0.1, 0, 0,0,0) 
#         colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral',"coral","tan"]
#         axs[i][j].pie(d1, explode=explode,labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

#         axs[i][j].set_aspect('equal')
#         count+=1
# plt.tight_layout()
# plt.show()


# In[ ]:




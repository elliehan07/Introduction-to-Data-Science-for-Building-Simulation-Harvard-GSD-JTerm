
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# In[1]:

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import mode
from sklearn import linear_model
import matplotlib
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor as KNN
get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import os, random
import sklearn
print sklearn.__version__


# # Getting Data #

# In[2]:

def GetPandasFromFileCSV(path):
    return pd.read_csv(path, delimiter=',')

def GetPandasFromFile(path, theSkipRow):
    return pd.read_csv(path, skiprows= theSkipRow , header=None)


# In[3]:

df =GetPandasFromFileCSV("data/_RentPriceTruliaMergeFinal.csv")
print df 
# print df.shape
# print df.head(3)
# print df.columns.values


# In[4]:

df.head()


# In[5]:

print df.shape
print df.columns.values


# In[6]:

df.columns.values


# In[7]:

data = df.convert_objects(convert_numeric=True)

to_float = []
to_encode = []
for col in data.columns:
    if data[col].dtype =='object':
        to_encode.append(col);
    if data[col].dtype =='int64':
        to_float.append(col);
#     print col,data[col].dtype
        
# print to_float
# print "----------------------"
# print to_encode

for feature_name in to_float:
    data[feature_name] = data[feature_name].astype(float)

def encode_categorical(array):
    if not array.dtype == np.dtype('float64'):
        return preprocessing.LabelEncoder().fit_transform(array) 
    else:
        return array
    
# Categorical columns for use in one-hot encoder
categorical = (data.dtypes.values != np.dtype('float64'))

# Encode all labels
data = data.apply(encode_categorical)


# In[8]:

for col in df.columns:
    print col,len(df[df[col].isnull()])


# # Deleting Null

# In[9]:

def RemoveRowWithNAN(data):
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    return data.reset_index()

def RemoveColumnsWithNull(data, num):
    complete_cols = [column for column in data.columns if len(data[column][data[column].isnull()]) < num]
    return data[complete_cols]

def ReomveRowwithNANWithNum(data):
    data = data.dropna(thresh=None)
    return data

def GetNumpyColumnFromIndex(theDF):
    theD = pd.DataFrame(theDF.values);
    return theD.as_matrix()

def CheckPandasNAN(data):
    theResult = pd.isnull(data)
    count = 0;
    for i in theResult:
        if(i == True): count+=1
    return "the number of NAN is :" , count

print "before processing NAN : ", df.shape

data.dropna(axis=0,subset=['RoomType','Price','Bathrooms'],inplace=True)

## deal with the NAN data !!!!!!!!!!!!!!!!!!!!!!!!!!!!
df_new = RemoveRowWithNAN(data)

# df_new = df_new.convert_objects(convert_numeric=True)
df_new = ReomveRowwithNANWithNum(df_new)

print "after processing NAN :", df_new.shape


# # Filling Null

# In[10]:

k=4
knntest = data[data['SQFT'].isnull()]
knntrain = data[data['SQFT'].isnull()==False]

xknn_train = knntrain[['RoomType','Bathrooms','Longitude','Latitude','Zip']].values
yknn_train = knntrain['SQFT'].values

xknn_test = knntest[['RoomType','Bathrooms','Longitude','Latitude','Zip']].values
neighbours = KNN(n_neighbors=k)
neighbours.fit(xknn_train, yknn_train)
yknn_test = neighbours.predict(xknn_test)

my_df = data.set_value( data['SQFT'].isnull(),'SQFT',yknn_test)


# In[11]:

my_df.shape


# In[12]:

for col in my_df.columns:
    print col,len(my_df[my_df[col].isnull()])


# In[13]:

my_df.head()


# # Selecting columns

# In[14]:

my_df.columns.values


# In[16]:

# df_drop = my_df.drop('Address',1)
df_drop = my_df.drop('Address',1)


# In[22]:

df_drop = df_drop.drop('Zip',1)
df_drop = my_df.drop('Address',1)


# In[30]:

my = 0
my1 = 0.0

print type (float(my))
print type (my1)


# In[24]:

df_drop = df_drop.drop('Zip',1)


# In[25]:

df_drop.columns.values


# In[26]:

df_iloc = df_drop.iloc[:10000,7:]


# In[32]:

print df_iloc.columns.values
print df_iloc.shape
print df_iloc.head()


# In[33]:

df_iloc['pixelWater'] = df_iloc['pixelRiver'] + df_iloc['pixelSea']


# In[34]:

df_iloc.head()


# In[35]:

df_iloc.rename(columns={'pixelWater':'datawater'}, inplace=True)


# In[36]:

df_iloc.head()


# In[37]:

my_Names = ['Latitude','Longitude','crime','Bathrooms','SQFT','Price','energySiteEUI']

datacon = df_iloc[my_Names]
datacon.head()


# In[39]:

datacon['energySiteEUI'].describe()


# # Visualizing data

# In[ ]:

# energy = datacon['energySiteEUI']
# temp = energy.values
# plt.hist(temp,bins=50)
# plt.show()


# In[40]:

energy = datacon['energySiteEUI'].values

plt.hist(energy,bins =50)
plt.show()


# In[ ]:

## Delete rows with zeros 
# dataNoZero = datacon[(datacon['energySiteEUI']==0) == False]
# dataNoZero.shape


# In[41]:

dataNoZero = datacon[(datacon['energySiteEUI']==0) == False]


# In[42]:

dataNoZero.shape


# In[44]:

energy = dataNoZero['energySiteEUI']
temp = energy.values
plt.hist(temp,bins=50)
plt.show()


# In[45]:

## Delete rows with Outliers 

dataNoOutliers = dataNoZero[(dataNoZero['energySiteEUI']>1000) == False]

energy = dataNoOutliers['energySiteEUI']
temp = energy.values
plt.hist(temp,bins=50)
plt.show()


# In[46]:

dataNoOutliers['energySiteEUI'].describe()


# In[47]:

y = energy.values
plt.hist(y,bins = 50,color="#FFBBBB")
plt.axvline(68.800000,c="green")
plt.axvline(130.900000,c="red")
plt.axvline(250.000000,c="green")
plt.xlabel("energyEUI")
plt.ylabel("Frequency")
plt.xlim(0, 1000)
plt.legend()
plt.show()


# In[48]:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors

fig = plt.figure(figsize=(35,10))

#Project onto axes: 1, 2, 3
ax1 = fig.add_subplot(1, 3, 1,  projection='3d')

newData = pd.DataFrame()
newData['Longitude'] = dataNoOutliers['Longitude']
newData['Latitude'] = dataNoOutliers['Latitude']
newData['y'] = y

newData1 = newData[newData['y']<68.8]
newData2 = newData[(newData['y']>68.8 )&(newData['y']<250)]
newData3 = newData[250<newData['y']]


ax1.scatter(newData1['Longitude'], newData1['Latitude'],newData1['y'], label='Low' , facecolors = "gray",edgecolors = "blue",alpha = 0.5, s=18)
ax1.scatter(newData2['Longitude'], newData2['Latitude'],newData2['y'], label='Mid' , facecolors = "#FFBBBB",edgecolors = "green",alpha = 0.5, s=18)
ax1.scatter(newData3['Longitude'], newData3['Latitude'],newData3['y'], label='High' , facecolors = "gray",edgecolors = "red",alpha = 0.5, s=18)


ax1.set_xlabel('\n'+'\n' + 'Longitude')
ax1.set_ylabel('\n'+'\n' +'Latitude')
ax1.set_zlabel('\n'+'\n' +'Energy')
ax1.set_title('Boston Energy use By Longitude & Latitude')
ax1.legend(loc='lower left')

plt.tight_layout()
plt.show()


# In[49]:

def SavePandasToCSV(d, path):
    d.to_csv(path)
    return "done!!"


# In[50]:

SavePandasToCSV(dataNoOutliers, "energy.csv")


# In[ ]:




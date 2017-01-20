
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# ## 1 PSET
# the rules of multiplication 

# In[1]:

### your code here




# end code


# In[2]:

# possible answer
for i in range(1,10):
    for j in range(1,10):
        print "%s X %s =" %(i,j) , i * j


# ## 2 PSET
# max, min, median, mean of data

# In[3]:

data = [2,5,7,2,1,10,12,45,62,23,53,12,3,6,7]


# In[4]:

### your code here




# end code


# In[5]:

import math
# a possible answer
def getMin(data):
    theMin = data[0]
    for i in data:
        if theMin > i:
            theMin = i
    return theMin
def getMax(data):
    theMin = data[0]
    for i in data:
        if theMin < i:
            theMin = i
    return theMin
def getMean(data):
    theSum = 0.0;
    for i in data:
        theSum+=i
    return theSum / len(data)
def getMedian(data):
    data.sort()
    theIndex = len(data) / 2
    if theIndex % 2 == 0:
        return data[theIndex]
    else:
        return (data[theIndex-1] + data[theIndex]) / 2.0
        
print "the min:", getMin(data)        
print "the max:", getMax(data)  
print "the mean:", getMean(data)  
print "the median:", getMedian(data)


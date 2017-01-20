
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# # python basic

# ## Basic Syntax

# ## 1. variable and type

# In[1]:

myVariable = 10

print myVariable


# In[56]:

# your code here


# In[2]:

myString = "hello world"
myInt = 7
myFloat = 1.23
myBool = True

print myString
print myInt
print myFloat
print myBool


# In[57]:

# your code here




# In[3]:

print type(myString)
print type(myInt)
print type(myBool)


# In[58]:

# your code here




# ## 2. number

# In[4]:

theNum = 3
theNum = 2
print theNum


# In[59]:

# your code here




# In[5]:

numA = 3 
numB = 2.0
theSum = numA + numB
print theSum


# In[60]:

# your code here




# In[6]:

myNum = 1
myNum += myNum
myNum += myNum
print myNum


# In[61]:

# your code here




# In[7]:

total = ((2 ** 2 ) * 0.5 ) + 3.14159 
print total


# In[62]:

# your code here




# ## 3. comment

# In[8]:

# your school
schoolName = "GSD" 
year = 2017 # the year
'''
this is ...
the result below
'''
print schoolName , year


# In[63]:

# your code here




# ## 4. string
# 
# documentation: https://docs.python.org/2/library/string.html

# In[9]:

theText = "hello world"
print theText


# In[64]:

# your code here




# In[10]:

print theText[0]
print theText[3]
print theText[-1]
print theText[len(theText)/2]
print len(theText)


# In[65]:

# your code here




# In[11]:

theString = "Hello GSD"
print theString.lower()
print theString.upper()


# In[66]:

# your code here




# In[12]:

# concatenate for string
print theText + ", "+ theString
print "theText %s, theString %s" %(theText, theString)


# In[67]:

# your code here




# ## 5. Casting

# In[13]:

myStringNum= "3"
myFloatNum = 3.0

print ("myStringNum is ", myStringNum, "theType is", type(myStringNum),"\n", 
       "myFloatNum is" , myFloatNum, "myFloatNum", type(myFloatNum))

print 10 * float(myStringNum)
print "befor:", type(myFloatNum), "after casting" , type(str(myFloatNum))


# In[ ]:

# your code here




# In[14]:

theBool ="True"
print type(theBool)
print type(bool(theBool))


# In[ ]:

# your code here




# ## 6. Conditional Statement

# In[15]:

booA = True
booB = False
print booA == booB
print booA != booB


# In[ ]:

# your code here




# In[16]:

a = 10
b = "10"
print a == b
print a == int(b)


# In[ ]:

# your code here




# In[17]:

boolA = 3 > 12
print boolA
boolB = 4 < 6 * 4
print boolB


# In[ ]:

# your code here




# In[18]:

a = 5
b = 5.0
print type(a) == type(b)
print type(float(a)) == type(b)


# In[ ]:

# your code here




# In[19]:

print 3 == 2 or 5 == 5
print 3 == 2 and 5 == 5


# In[ ]:

# your code here




# In[20]:

print (3 == 2 or 5 == 5) and  (3 == 2 and 5 == 5)


# In[ ]:

# your code here




# In[21]:

a = 3
b = 5

if a > b:
    print "a is greater than b"
elif(a == b):
    print "a is equal to b"
else:
    print "a is smaller than b"


# In[ ]:

# your code here




# ## 7. loop

# In[22]:

for i in range(3):
    print "hello world"


# In[ ]:

# your code here




# In[23]:

for i in range(3,6):
    print i


# In[ ]:

# your code here




# In[24]:

theIter = 10
for i in range(theIter):
    if i % 2 == 0: # modulo operation
        print i
    else :
        print "odd number"


# In[ ]:

# your code here




# In[25]:

theIter = 5
while theIter > 0:
    print theIter
    theIter -= 1


# In[ ]:

# your code here




# In[26]:

for i in range(2):
    for j in range (3):
        print "i =" , i ,",", "j =", j


# In[ ]:

# your code here




# ## Data Structure

# ## 8. List
# https://docs.python.org/2/tutorial/datastructures.html

# In[27]:

theListA = [1,2,3,4,5]
theListB = ["hello world"]
print theListA
print theListA[3:]

print theListB

theListB.append(3)
theListB.append(6)
print theListB


# In[ ]:

# your code here




# In[28]:

theString = "hello GSD"
print theString[3:]
print "the length of string is",len(theString), "."
print "---------------------------"
for i in theString:
    print i


# In[ ]:

# your code here




# In[29]:

basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
basket = set(basket) # https://docs.python.org/2/library/sets.html
for f in sorted(basket):
    print f


# In[ ]:

# your code here




# In[30]:

myList = [1,2,5,8,9]
for i in myList:
    print i


# In[ ]:

# your code here




# In[31]:

print len(myList)
print myList[-1]


# In[ ]:

# your code here




# In[32]:

theNewList = []
for i in range(10):
    theNewList.append(i*10)
print theNewList


# In[ ]:

# your code here




# In[33]:

theNewList.index(50)


# In[ ]:

# your code here




# In-built function of List, python
# https://docs.python.org/2/tutorial/datastructures.html

# In[34]:

myList = [1,2,3,"gsd", -5, True,'mdes',10]
print myList
myList.insert(0, 0)
print myList
myList.remove('gsd')
print myList


# In[ ]:

# your code here




# In[35]:

myList.pop()
print myList


# In[ ]:

# your code here




# In[36]:

myList.pop(2) # index
print myList


# In[ ]:

# your code here




# In[37]:

myList.sort()
print myList


# In[ ]:

# your code here




# In[38]:

myList.reverse()
print myList


# In[ ]:

# your code here




# ## 9. Dictionary
# https://docs.python.org/2/tutorial/datastructures.html

# In[39]:

person = {'height': 170, "name": 'Tony', "weight": 65  }
print person
print person['name']
print person['weight']


# In[ ]:

# your code here




# In[40]:

person['adress'] = 'Cambridge, MA'
print person


# In[ ]:

# your code here




# In[41]:

person['height'] += 10
print person


# In[ ]:

# your code here




# In[42]:

person["skill"] = ['rhino', 'grasshopper', 'photoshop']
print person


# In[ ]:

# your code here




# In[43]:

for i in person:
    content = person[i]
    if(type(content) == list):
        for j, skill in enumerate(content):
            print "the skill",j , ":", skill
    else:
        print content


# In[ ]:

# your code here




# In[44]:

theIndex = ["a", "b", "c"]
theNewDic = {} # Empty dictionary
for j in theIndex:
    lst = []
    for i in range(3):
        lst.append(i);
    theNewDic[str(j)] = lst
print theNewDic


# In[ ]:

# your code here




# ## 10. Tuple
# https://docs.python.org/2/tutorial/datastructures.html

# In[45]:

theTuple = (1,2, 3)
print theTuple
print theTuple[0]


# In[ ]:

# your code here




# In[46]:

tA = (1,2,3)
print tA * 2

tB = ('a','b','c')
tC = tA + tB
print tC


# In[ ]:

# your code here




# In[47]:

theIndex = ["a", "b", "c"]
theList = [1,2,3]
theTuple = zip(theIndex, theList)
print theTuple
for i in theTuple:
    print i[0], i[1]


# In[ ]:

# your code here




# In[48]:

theTuple = (1,2,3)
print theTuple
print type(theTuple)

newList = list(theTuple)
print newList
print type(newList)

newTuple = tuple(newList)
print newTuple
print type(newTuple)


# In[ ]:

# your code here




# ### 11. Function

# In[49]:

def theSum(a,b):
    print a + b
theSum(10,3)


# In[ ]:

# your code here




# In[50]:

def theSum(a,b):
    return a + b
mySum = theSum(10,3)
print mySum


# In[ ]:

# your code here




# In[51]:

def find_A_character_from_word(word):
    theWord = word.upper()
    for i, char in enumerate(theWord):
        if(char == "A"):
            print "yes"
            return i
    print "no"
    return -1
theIndex = find_A_character_from_word("apple")
print theIndex
print "--------------------------------"

theIndex = find_A_character_from_word("lemon")
print theIndex
print "--------------------------------"

theIndex = find_A_character_from_word("strawberry")
print theIndex


# In[ ]:

# your code here




# In[53]:

import random # https://docs.python.org/2/library/random.html

def getRandomValue(theMin, theMax):
    return random.randint(theMin, theMax)

def getNumofValue(theIter, theMin, theMax):
    outList = []
    for i in range(theIter):
        outList.append(getRandomValue(theMin, theMax))
    return outList


theIter = int(raw_input("number of iteration?"))
theMin = int(raw_input("min"))
theMax = int(raw_input("max"))
theList = getNumofValue(theIter, theMin, theMax)
print theList
print sum(theList)


# In[ ]:

# your code here




# In[54]:

from datetime import datetime # https://docs.python.org/2/library/datetime.html
now = datetime.now()

print '0%s/%s/%s' % (now.month, now.day, now.year)

from datetime import datetime
now = datetime.now()

print '0%s:%s:%s' % (now.hour, now.minute, now.second)

from datetime import datetime
now = datetime.now()

print '0%s/%s/%s 0%s:%s:%s' % (now.month, now.day, now.year, now.hour, now.minute, now.second)


# In[ ]:

# your code here




# In[55]:

import math
print dir(math)

def degreeFromRadians(degrees):
    return (degrees * (math.pi / 180.0))        

def radiansFromDegree(radians):
    return (radians * (180.0 / math.pi))

theValue = 90 
thesult = degreeFromRadians(90)
print thesult
print radiansFromDegree(thesult)


# In[ ]:

# your code here




# ## Additional Reading and Resources
# 
# #### python documentation(2.7x) 
# https://docs.python.org/2/
# 
# #### codecademy
# https://www.codecademy.com/learn/python

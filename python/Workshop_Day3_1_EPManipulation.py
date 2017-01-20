
# coding: utf-8

# ### J-Term 2017, Harvard GSD :
# ### Introduction to Data Science for Building Simulation
# ***
# Instructor: Jung Min Han, elliehan07@gmail.com <br>
# Teaching Assistant: NJ Namju Lee, nj.namju@gmail.com <br>
# Date/Time: Jan 9-12/ 1:00 - 3:00 p.m. <br>
# Location: 20 Sumner/Room 1-D<br>
# ***

# In[129]:

from pandas import *
import pandas as pd
import numpy as np


# Pandas Series Object: 1 dimensional data container
# ======
# 
# This object is a data container for vectors -- incorporating an index and string search functions

# In[4]:

s = Series(np.random.randn(5))
s


# In[6]:

labels = ['a', 'b', 'c', 'd', 'e']

s = Series(np.random.randn(5), index = labels)
s


# In[8]:

'z' in s


# In[9]:

s['b']


# In[10]:

s


# In[11]:

mapping = s.to_dict()
mapping


# DataFrame: 2D collection of Series
# ==================================

# In[12]:

df = DataFrame({'a': np.random.randn(6),
                'b': ['foo', 'bar'] * 3,
                'c': np.random.randn(6)})
df.shape


# In[13]:

df.index


# In[14]:

df


# In[15]:

#Transpose
df.T


# In[16]:

df = DataFrame({'a': np.random.randn(6),
                'b': ['foo', 'bar'] * 3,
                'c': np.random.randn(6)},
               index = date_range('1/1/2000', periods=6))
df


# In[19]:

df = DataFrame({'a': np.random.randn(6),
                'b': ['foo', 'bar'] * 3,
                'c': np.random.randn(6)},
               columns=['a', 'b', 'c', 'd','result'])
df.T


# # List Comprehension
# 

# In[22]:

d = [0,1,2,3,4,5]

for i in range(len(d)):
    d[i]+=2

print(d)


# In[24]:

d = [0,1,2,3,4,5]


d = [i + 2 for i in d]

print d


# In[31]:

d = [0,1,2,3,4,5]


d2 = [i < 3 for i in d]

print d2


# In[33]:

d = [0,1,2,3,4,5]

d = [i < 3 for i in d]

d = [i for i in d if i % 2 == 0]

print d


# In[37]:

words = "Hello,world,and,GSD"
words = words.split(",")
print words

stuff = [[d.upper(), d.lower(), len(d)] for d in words]

print stuff


# # IDF Editor

# Credit : Authors: Santosh Philip, Leora Tanjuatco
# ###Eppy is a scripting language for E+ idf files, and E+ output files. Eppy is written in the programming language Python. 
# 
# As a result it takes full advantage of the rich data structure and idioms that are avaliable in python. You can programmatically navigate, search, and modify E+ idf files using eppy. The power of using a scripting language allows you to do the following:
# 
# - Make a large number of changes in an idf file with a few lines of eppy code.
# - Use conditions and filters when making changes to an idf file
# - Make changes to multiple idf files.
# - Read data from the output files of a E+ simulation run.
# - Based to the results of a E+ simulation run, generate the input file for the next simulation run.
# 
# ###So what does this matter? Here are some of the things you can do with eppy:
# 
# - Change construction for all north facing walls.
# - Change the glass type for all windows larger than 2 square meters.
# - Change the number of people in all the interior zones.
# - Change the lighting power in all south facing zones. 
# - Change the efficiency and fan power of all rooftop units.
# - Find the energy use of all the models in a folder (or of models that were run after a certain date) 
# - If a model is using more energy than expected, keep increasing the R-value of the roof until you get to the expected energy use.

# In[38]:

# pip install eppy
from eppy import modeleditor 
from eppy.modeleditor import IDF


# In[39]:

IDF.setiddname("Energy+V7_2_0.idd")
idf = IDF("smallidf.idf")


# In[40]:

idf.printidf()


# In[42]:

idf.idfobjects['BUILDING']  # put the name of the object you'd like to look at in brackets


# In[47]:

bd =idf.idfobjects['BUILDING'][0]


# In[48]:

bd.Name


# In[49]:

bd.Terrain


# In[50]:

bd.Solar_Distribution


# In[51]:

building = idf.idfobjects['BUILDING'][0]


# In[52]:

print building.Name
print building.North_Axis
print building.Terrain
print building.Loads_Convergence_Tolerance_Value
print building.Temperature_Convergence_Tolerance_Value
print building.Solar_Distribution
print building.Maximum_Number_of_Warmup_Days
print building.Minimum_Number_of_Warmup_Days


# In[53]:

idf_con = IDF("constructions.idf")


# In[54]:

idf_con.idfobjects


# In[64]:

materials = idf_con.idfobjects["MATERIAL"]
ms = materials


# In[61]:

len(materials)


# In[65]:

Metal  = materials[0]
insulation  = materials[1]


# In[66]:

Metal


# In[67]:

materials[-1]


# In[68]:

materials[-1].Name = 'My_material'
materials[-1].Roughness = 'MediumSmooth'
materials[-1].Thickness = 0.03
materials[-1].Conductivity = 0.16
materials[-1].Density = 600
materials[-1].Specific_Heat = 1500


# In[69]:

materials[-1]


# ## Looping through E+ objects

# In[70]:

for material in materials:
    print material.Name 


# In[71]:

[material.Name for material in materials] 


# In[72]:

[material.Roughness for material in materials]


# In[73]:

[material.Thickness for material in materials]


# In[74]:

[material.Thickness for material in materials if material.Thickness > 0.1]


# In[77]:

[[material.Name,material.Thickness] for material in materials if material.Thickness > 0.1]


# In[78]:

thick_materials = [material for material in materials if material.Thickness > 0.1]


# In[79]:

thick_materials


# In[80]:

# change the names of the thick materials
for material in thick_materials:
    material.Name = "THICK " + material.Name   


# In[85]:

thick_materialsidf1.save()


# In[86]:

idf_con.save("thick_Materials")


# ## Geometry functions in eppy

# In[87]:

idf_geo = IDF("5ZoneSupRetPlenRAB.idf")
surfaces = idf_geo.idfobjects['BUILDINGSURFACE:DETAILED']


# In[90]:

# Let us look at the first surface
surface = surfaces[0]

print "surface azimuth =",  surface.azimuth, "degrees"
print "surface tilt =", surface.tilt, "degrees"
print "surface area =", surface.area, "m2"


# In[91]:

# all the surface names
s_names = [surface.Name for surface in surfaces]
print s_names
# print s_names[:5] # print five of them


# In[92]:

# surface names and azimuths
s_names_azm = [(sf.Name, sf.azimuth) for sf in surfaces]

print s_names_azm[:5] # print five of them


# In[93]:

# surface names and tilt
s_names_tilt = [(sf.Name, sf.tilt) for sf in surfaces]

for name, tilt in s_names_tilt: 
    print name, tilt


# In[97]:

# surface names and areas
s_names_area = [(sf.Name, sf.area) for sf in surfaces]

for name, area in s_names_area[:5]: # just five of them
    print name, area, "m2"


# ##Selecting Walls

# In[99]:

# just vertical walls
vertical_walls = [sf for sf in surfaces if sf.tilt == 90.0]
print [sf.Name for sf in vertical_walls]


# In[100]:

# north facing walls
north_walls = [sf for sf in vertical_walls if sf.azimuth == 0.0]
print [sf.Name for sf in north_walls]


# In[101]:

# north facing exterior walls
exterior_nwall = [sf for sf in north_walls if sf.Outside_Boundary_Condition == "Outdoors"]
print [sf.Name for sf in exterior_nwall]


# In[102]:

# print out some more details of the north wall
north_wall_info = [(sf.Name, sf.azimuth, sf.Construction_Name) for sf in exterior_nwall]
for name, azimuth, construction in north_wall_info:
    print name, azimuth, construction   


# In[104]:

# change the construction in the exterior north walls
for wall in exterior_nwall:
    wall.Construction_Name = "NORTHERN-WALL" # make sure such a construction exists in the model 


# In[105]:

# see the change
north_wall_info = [(sf.Name, sf.azimuth, sf.Construction_Name) for sf in exterior_nwall]
for name, azimuth, construction in north_wall_info:
    print name, azimuth, construction  


# In[109]:

# see this in all surfaces
for sf in surfaces:
    print sf.Name, sf.azimuth, sf.Construction_Name


# # Parametric Example

# In[110]:

IDF.setiddname("Energy+V7_2_0.idd")
idf1 = IDF("Baseline.idf")


# In[111]:

idf1.printidf()


# In[112]:

idf1.idfobjects


# In[113]:

glazing = idf1.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM'][0]


# In[114]:

glazing


# In[115]:

print glazing.UFactor
print glazing.Solar_Heat_Gain_Coefficient
print glazing.Visible_Transmittance


# In[117]:

glazing.UFactor = "=$U_Factor"
glazing.Solar_Heat_Gain_Coefficient = "=$SHGC"
glazing.Visible_Transmittance = "=$Tvis"


# In[118]:

glazing


# In[121]:

parametric1 = idf1.idfobjects['PARAMETRIC:SETVALUEFORRUN']
print parametric1


# In[122]:

IDF.setiddname("Energy+V7_2_0.idd")
idf2 = IDF("EP_Parametric.idf")


# In[123]:

parametric2 = idf2.idfobjects['PARAMETRIC:SETVALUEFORRUN']


# In[124]:

parametric2


# In[125]:

idf1.idfobjects['PARAMETRIC:SETVALUEFORRUN'] = parametric2


# In[126]:

idf1.idfobjects['PARAMETRIC:SETVALUEFORRUN']


# In[127]:

idf1.save()


# In[128]:

idf1.saveas('Baseline_Parametric.idf')


# In[ ]:




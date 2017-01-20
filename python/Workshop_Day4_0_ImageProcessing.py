
# coding: utf-8

# In[8]:

import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[13]:

# define the list of boundaries
##[min(B,G,R),max(B,G,R)]
boundaries = [
    ([0, 0, 150], [165, 165, 255]),
    ([0, 0, 30], [230, 190, 150]),
    ([155, 0, 0], [255, 255, 255])
]


# In[16]:

# import the necessary packages
import numpy as np
import argparse
import cv2

# path = 'Z:/Dropbox/1_Jungmin&NJ/[DataScience_JTerm]/[release]/Day4/image2.jpg'
path = "C:/Users/EllieHan/Dropbox/1_Jungmin&NJ/[DataScience_JTerm]/[release]/Day4/image2.jpg"
image = cv2.imread(path)
# print image

# define the list of boundaries
# loop over the boundaries
count =0
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

#     # show the images
    result = np.hstack([image, output])
    total = result.shape[0] * result.shape[1]
    print total
#     dd = result.flat()

#     result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = result[:,result.shape[1]/2:]
#     plt.imshow(result)
    cv2.imshow("image", result)
    cv2.imwrite("image"+str(count)+".png", result)
#     cv2.imwrite()
    cv2.waitKey(0)
    count+=1


# In[ ]:




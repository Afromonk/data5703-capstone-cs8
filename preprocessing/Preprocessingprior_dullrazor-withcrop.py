#!/usr/bin/env python
# coding: utf-8

# In[19]:


#LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from skimage.measure import label
from keras.applications.resnet50 import preprocess_input
import random


# In[20]:


#MY DRIVE PATH
image_path = "3classes/HAM10000_train_by_class"
image_output_path = "3classes/HAM10000_train_dull_G-7-7-5-5_M-3-3"
dullrazor = True

#This just triggers the reduction of the images down to imgnum
doingtest = True
#change to desired number of images for training; this was simply to do downsampling as I was testing on 2k per class
imgnum = 2000


# In[21]:


#LOAD IMAGES
image_path_mel = os.path.join(image_path,"mel/")
image_path_bcc = os.path.join(image_path,'bcc/')
image_path_others = os.path.join(image_path,'others/')
image_output_mel = os.path.join(image_output_path,'mel/')
image_output_bcc = os.path.join(image_output_path,'bcc/')
image_output_others = os.path.join(image_output_path,'others/')

def loadImages(path):
    # Put files into lists and return them as one list with all images in the folder
    image_files = sorted([os.path.join(path, file)
                          for file in os.listdir(path)
                          if file.endswith('.jpg')])
    return image_files


# In[22]:


import pathlib
#check number of files in those directories here
melcount = 0
for path in pathlib.Path(image_path_mel).iterdir():
    if path.is_file():
        melcount += 1
print(melcount)
otherscount = 0
for path in pathlib.Path(image_path_others).iterdir():
    if path.is_file():
        otherscount += 1
print(otherscount)
bcccount = 0
for path in pathlib.Path(image_path_bcc).iterdir():
    if path.is_file():
        bcccount += 1
print(bcccount)


# In[23]:


#only do once, done to remove files in others down to imgnum; that said, doesn't do anything if its already down to imgnum
#sole purpose of this was to reduce training set for 2000 for when i was testing
"""
files = os.listdir('3classes/HAM10000_train_by_class/others')
otherscount = 0
for path in pathlib.Path('3classes/HAM10000_train_by_class/others').iterdir():
    if path.is_file():
        otherscount += 1
for file in random.sample(files,otherscount-imgnum):
    os.remove(os.path.join('3classes/HAM10000_train_by_class/others',file))
"""


# In[24]:


def read_image_cv2(Dataset):  
  img = [cv2.imread(i, cv2.IMREAD_UNCHANGED,) for i in Dataset[:len(Dataset)]]
  return img
data = loadImages(image_path_mel)
image = cv2.imread(data[2], cv2.IMREAD_UNCHANGED)


# In[26]:


#HYPER-PARAMETERS FOR DULL RAZOR

#resize
resizeon = True #triggers resize
resizeW = 600 #height of resized image
resizeH = 450 #width of resized image
ResizeParam = [resizeW, resizeH]

#Resize alternative: Retain aspect ratio by cropping down to the central 'square'
resizecropon = True #triggers on, overrides resizeon
resizealtres = 450 #height and width of resulting square image

#dull razor
razorblur = "G" #M for median, G for Gaussian, have found Gaussian works better for this at cost of removal 
mediankernel_razorblur = 7 #blur number
filterstructure = 7 #will form structure for kernel
lowerbound = 5 #15 original
inpaintmat = 5

#blur
blur = True #turns on blur
normalblur = "M" #M for Median, G for Gaussian
mediankernel_blur = 3 #for Median, higher numbers = more blur, think it has to be odd
blurnum = 3 #for Gauss, higher numbers = more blur, think it has to be odd


# In[27]:


#RESIZE IMAGE
def resize(Image,ResizeParam):
    #resize
    dim = (ResizeParam[0], ResizeParam[1])
    res_img = []
    if resizeon == True:
        for i in range(len(Image)):
            res = cv2.resize(Image[i], dim, interpolation=cv2.INTER_LINEAR)
            res_img.append(res)
    else:
        res_img = Image
    return res_img

#Resize but has to be down to square, assuming mole would be in the middle 'square'
def retainaspectresize(images,resolution):
    res_images = []
    for image in images:
        h,w,d = image.shape
        if h > w:
            croppedimg = image[0:w,int(round(h/2-w/2)):int(round(h/2+w/2))]
            resimg = cv2.resize(croppedimg, (resolution,resolution), interpolation=cv2.INTER_LINEAR)
        elif w > h:
            croppedimg = image[int(round(w/2-h/2)):int(round(w/2+h/2)),0:h]
            resimg = cv2.resize(croppedimg, (resolution,resolution), interpolation=cv2.INTER_LINEAR)
        else:
            resimg = cv2.resize(image,(resolution,resolution),interpolation=cv2.INTER_LINEAR)
        res_images.append(resimg)
    return res_images

#REMOVING HAIR USING DULL RAZOR
def dull_razor(ResizedImages):
    #dull razor
    hair_removed_images = []
    if dullrazor == True:
        for i in range(len(ResizedImages)):
            if razorblur == "M":
                tempimg = cv2.medianBlur(ResizedImages[i],mediankernel_razorblur)
            elif razorblur == "G":
                tempimg = cv2.GaussianBlur(ResizedImages[i], (mediankernel_razorblur, mediankernel_razorblur),0)
            else:
                tempimg = ResizedImages[i]
            gyimage = cv2.cvtColor(tempimg, cv2.COLOR_RGB2GRAY)
            filtersize = (filterstructure,filterstructure)
            kernelrazor = cv2.getStructuringElement(cv2.MORPH_RECT, filtersize)
            gyimage = cv2.morphologyEx(gyimage, cv2.MORPH_BLACKHAT, kernelrazor)

            retrazor, maskrazor = cv2.threshold(gyimage, lowerbound, 255, cv2.THRESH_BINARY)
            x = cv2.inpaint(ResizedImages[i], maskrazor, inpaintmat, cv2.INPAINT_TELEA)
            hair_removed_images.append(x)
        return hair_removed_images
    else:
        return ResizedImages

#BLUR AFTER REMOVING HAIR
def blur(HairRemovedImages):
    #blur
    if blur == True:
        if normalblur == "M":
            for i in range(len(HairRemovedImages)):
                HairRemovedImages[i] = cv2.medianBlur(HairRemovedImages[i], mediankernel_blur)
        elif normalblur == "G":
            for i in range(len(HairRemovedImages)): 
                HairRemovedImages[i] = cv2.GaussianBlur(HairRemovedImages[i], (mediankernel_blur, mediankernel_blur), 0)
    return HairRemovedImages


# In[28]:


#SOFT ATTENTION MAPPING COMPLETE PROCESS
#####################
#PREPROCESS IMAGES BEFOFE MAPPING
def softention_preprocess(SoftentionImages):
    expanded_list = []
    # use the pre processing function of ResNet50 
    for i in SoftentionImages:
        first = preprocess_input(i)
        second = np.expand_dims(first, 0)
        expanded_list.append(second)
    return expanded_list


# In[29]:


#PLOT ALL THREE IMAGES - RESIZED IMAGE, IMAGE WITH HAIR REMOVED AND IMAGE WITH SOFT ATTENTION MAPPING
def plot_heatmaps(Range, SoftentionExpandedImages, pretrained_model, ResizeParam, SoftentionImages, ResizedImages, HairRemovedImages, k):  
    #given a range of indices generate the heat maps 
    level_map_list = []
    for i in Range:
        for j in SoftentionExpandedImages:
            heatmap = softention_mapping(j, i, pretrained_model, ResizeParam, SoftentionImages)
            level_map_list.append(heatmap)
    plt.figure(figsize=(8, 8))
    display_three(ResizedImages[k],HairRemovedImages[k],level_map_list[k], "original","dullrazor+blur", "soft attention")


# In[30]:


def delete_random_elems(input_list,n):
    to_delete = set(random.sample(range(len(input_list)),n))
    return[x for i,x in enumerate(input_list) if not i in to_delete]


# In[31]:


#clear out output directory
for root, dirs, files in os.walk(image_output_others):
    for file in files:
        os.remove(os.path.join(root, file))
for root, dirs, files in os.walk(image_output_mel):
    for file in files:
        os.remove(os.path.join(root, file))
for root, dirs, files in os.walk(image_output_bcc):
    for file in files:
        os.remove(os.path.join(root, file))


# In[32]:


data = loadImages(image_path_others)
if doingtest == False:
    data = delete_random_elems(data,len(data)-imgnum)
img = read_image_cv2(data)
if resizecropon == True:
    retainaspectresize(img,resizealtres)
else:
    res_img = resize(img, ResizeParam)
hair_removed_image = dull_razor(res_img)
SoftentionImage = hair_removed_image_furtherbluring = blur(hair_removed_image)
imagecount = 0
for i in SoftentionImage:
    cv2.imwrite(os.path.join(image_output_others,str(imagecount)+".jpg"),i)
    imagecount=imagecount+1


# In[33]:


#I know I should reiterate but chances are this code will never see the light of day. Will redo if it happens to be worth it.


# In[34]:


data = loadImages(image_path_mel)
if doingtest == False:
    data = delete_random_elems(data,len(data)-imgnum)
img = read_image_cv2(data)
if resizecropon == True:
    retainaspectresize(img,resizealtres)
else:
    res_img = resize(img, ResizeParam)
hair_removed_image = dull_razor(res_img)
SoftentionImage = hair_removed_image_furtherbluring = blur(hair_removed_image)
imagecount = 0
for i in SoftentionImage:
    cv2.imwrite(os.path.join(image_output_mel,str(imagecount)+".jpg"),i)
    imagecount=imagecount+1


# In[35]:


data = loadImages(image_path_bcc)
if doingtest == False:
    data = delete_random_elems(data,len(data)-imgnum)
img = read_image_cv2(data)
if resizecropon == True:
    retainaspectresize(img,resizealtres)
else:
    res_img = resize(img, ResizeParam)
hair_removed_image = dull_razor(res_img)
SoftentionImage = hair_removed_image_furtherbluring = blur(hair_removed_image)
imagecount = 0
for i in SoftentionImage:
    cv2.imwrite(os.path.join(image_output_bcc,str(imagecount)+".jpg"),i)
    imagecount=imagecount+1


# In[36]:


import pathlib
#check number of files in those directories here
melcount = 0
for path in pathlib.Path(image_output_mel).iterdir():
    if path.is_file():
        melcount += 1
print(melcount)
otherscount = 0
for path in pathlib.Path(image_output_others).iterdir():
    if path.is_file():
        otherscount += 1
print(otherscount)
bcccount = 0
for path in pathlib.Path(image_output_bcc).iterdir():
    if path.is_file():
        bcccount += 1
print(bcccount)


# In[ ]:





# In[ ]:





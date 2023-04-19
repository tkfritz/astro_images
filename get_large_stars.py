#get zoo galaxies in same way as for field one for other fields
#standard libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.cm as cm
import sys
import os
#wcs is incompabible with newest numpy thus below not used 
#from astropy import wcs
#to access astronomical images in fits format
from astropy.io import fits
from functions_wcs import *

#name of field to be changed for each run  
field=str(4)

df=pd.read_csv('Stripe82_'+field+'b.csv',sep=',')
print(df.columns)

print(df['class'].value_counts())

print(df['subclass'].value_counts())


df2=df[(df['probPSF_r']==1 ) & (df['probPSF_i']==1 ) & (df['probPSF_g']==1 )  &(df['class']=='STAR') & (df['psfMag_r']>14.8) & (df['psfMag_r']<18.0)]
print(df2.shape)

path='/home/tobias/ml-testing/astr-images/'
list_input_files=[f for f in os.listdir(path) 
    if f.endswith('_rdeep.fits.gz') and os.path.isfile(os.path.join(path, f))]
list_input_files.sort()
print(list_input_files)

centers=np.zeros((2,len(list_input_files)))
wcs_par=np.zeros((6,len(list_input_files)))
for i in range(len(list_input_files)):
    print(i)
    #get images
    hbin=fits.open(path+list_input_files[i],memmap=True)
    #get parameters wanted 
    res=image_area(hbin)
    par=image_par(hbin)
    hbin.close()
    centers[0,i]=(res[0,0]+res[0,1])/2  #center is avarage of extension in both dimensions
    centers[1,i]=(res[1,0]+res[1,1])/2
    #parameters to find objects on images
    wcs_par[:,i]=par
print(centers)  




# I add dummy columns which are later filter with the image and the pixels positions on it for all onbjects
df2['image']=-1
df2['pixel_x']=-1.0
df2['pixel_y']=-1.0
print(df2.columns)
df2['off_image']=False

for i in range(df2.shape[0]):
    #distances to all images ceneters
    r=np.sqrt((df2['ra'].iloc[i]-centers[0])**2+(df2['dec'].iloc[i]-centers[1])**2)
    #id of the image
    df2['image'].iloc[i]=np.argmin(r)
    #get pixel coordinates of image 
    coor=image_xy(df2['ra'].iloc[i],df2['dec'].iloc[i],par=wcs_par[:,df2['image'].iloc[i]],image=False)
    df2['pixel_x'].iloc[i]=coor[0]
    df2['pixel_y'].iloc[i]=coor[1]
    
    
df2=df2.sort_values(by='image')
#reset index since the previous row is not wanted 
df2=df2.reset_index()
print(df2.head)    

df2,cut_out=get_cutouts(df2,21,list_input_files)

df3=df2[df2.off_image==False]
#new image array, to which also a 4 dimension of zero size is added 
cut_out2=np.zeros((cut_out.shape[0],cut_out.shape[1],1,df3.shape[0]))
counter=0
for i in range(df2.shape[0]):
    if df2.off_image.iloc[i]==False:
        #adding the cut outs not of image
        cut_out2[:,:,0,counter]=cut_out[:,:,i]
        counter+=1
print(f"counter is {counter}, number of rows is {df3.shape[0]}")    

#only saved when there is something otherwise likely a problem        
if counter>0: 
    np.save("stripe82_"+field+"_stars_im.npy",cut_out2)    
    df3.to_csv('stripe82_'+field+'_stars_table.csv') 
else:
    print("nothing saved, there is nothing collected")    

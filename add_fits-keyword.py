from astropy.io import fits
import os


path='/home/tobias/ml-testing/astr-images/'
list_input_files=[f for f in os.listdir(path) 
    if f.endswith('_rdeep.fits') and os.path.isfile(os.path.join(path, f))]
list_input_files.sort()
print(list_input_files)


for i in range(len(list_input_files)):
    print(i)
    #get images
    hdul = fits.open(path+list_input_files[i])
    hbin = hdul[0].header
    #check for crval2
    x='CRVAL2' in hbin  # Check for existence
    print(x)
    if x==False:
        hbin['CRVAL2'] =0  # Add a crval2 keyword 
        hdul.writeto(list_input_files[i],overwrite=True) #overwriting
    hdul.close()

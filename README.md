# Astronomical images. 

In this notebook I classify astronomical images in supervised learning. 

I start with separating stars from galaxies min the notebooks, the results are mainly presented in star_galaxy3b.ipynb

For the next problem of classfication galaxies by two types a training set is needed for which the original images are too large to be saved locally. In get_zoo_galaxies.py I save the subimages around the targets for many fields together with explanatoraty data frames 

The data is then prepared in galaxy_type_prep.ipynb and fit in galaxy_type_fit1.ipynb, galaxy_type_fit2.ipynb, run_fits.py, run_fits_noi.py and run_fits_rotmir.py
The main results are presented in galaxy_type_present1.ipynb

Own functions and classes from functions_ml.py are used for the galaxy type process. 

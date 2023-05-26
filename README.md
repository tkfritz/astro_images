# Astronomical images

In this notebook I classify astronomical images with supervised learning methods, in particular logistic regression, xgboost and perceptron neural networks and convolutional neural networks. 

I start with separating stars from galaxies in the notebooks, the results are mainly presented in star_galaxy3b.ipynb

For the next problem of classification galaxies by two types a training set is needed for which the original images are too large to be saved locally. In get_zoo_galaxies.py I save the subimages around the targets for many fields together with explanatory data frames. It uses functions from functions_wcs.py. add_fits-keyword.py helps to correct files in the process. 
Different collections of images and associated tables are created with get_large_stars.py, get_small_objects.py and get_zoo_galaxies_all.py

The data is then prepared in galaxy_type_prep.ipynb and fit in galaxy_type_fit1.ipynb, galaxy_type_fit2.ipynb, run_fits.py, run_fits_noi.py and run_fits_rotmir.py
The main results can be applied with apply_models.py on data with the same meta properties.
The main results are presented in galaxy_type_present1.ipynb
Own functions and classes from functions_ml.py are used for the galaxy type process. 

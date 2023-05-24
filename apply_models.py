#standard libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.cm as cm
import sys
import os
import time
import random as random
#torch functions
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
#sklearn helper functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,f1_score, log_loss
#xgboost for comparison
from xgboost import XGBClassifier
#logistic regression for comparison 
from sklearn.linear_model import LogisticRegression
from functions_ml import *
import pickle

#getting the list of images
myPath='/home/tobias/ml-testing/astr-images'
list_images=[f for f in os.listdir(myPath) 
    if f.endswith('_ell_spiral_im.npy') ]
list_images.sort()
#getting the list of tables 
list_tables=[f for f in os.listdir(myPath) 
    if f.endswith('_ell_spiral_table.csv')]
list_tables.sort()

#apply the best models and save results 

#done
#probs=predict_probs(list_images,list_tables,'xgboost_model_spiral_ell_rot_mirr_l2reg3.json',modelname='xgboost',image_output=False,df_output=False,split=0.6,train_choice=False,seed=1)
#np.savetxt("best_v0_xgb_rottest_prob.txt",probs)

#done
#probs=predict_probs(list_images,list_tables,'xgboost_model_spiral_ell_rot_mirr_l2reg3.json',modelname='xgboost',image_output=False,df_output=False,split=0.6,train_choice=True,seed=1)
#np.savetxt("best_v0_xgb_rottrain_prob.txt",probs)



probs=predict_probs(list_images,list_tables,'mlp_4layers_reg0.003_rotmir_200epochs_v0.pkl',modelname='perceptron',image_output=False,df_output=False,split=0.6,train_choice=False,seed=1)
np.savetxt("best_v0_per_rottest_prob.txt",probs)

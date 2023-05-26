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
from astropy.io import fits
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

#combines numpy arrays of 4d shape, same shape first 3, last variable
#combines numpy arrays of 4d shape, same shape first 3, last variable
def comb_nump_4d(input_list):
    l=0
    for i in range(len(input_list)):
        a=np.load(input_list[i])
        l+=a.shape[3]
    combined=np.zeros((a.shape[0],a.shape[1],a.shape[2],l))
    l=0
    for i in range(len(input_list)):
        a=np.load(input_list[i])
        combined[:,:,:,l:l+a.shape[3]]=a
        l+=a.shape[3]  
    return combined

def loop_xgboost(feature_train,target_train,feature_test,target_test,regs):
    stats=np.zeros((5,len(regs)))
    for i in range(len(regs)):
        print(f"doing l2 regularization {regs[i]}")
        xc1=XGBClassifier(max_depth=6,reg_lambda=regs[i]).fit(feature_train,target_train)
        train_pred=xc1.predict(feature_train)
        test_pred=xc1.predict(feature_test)
        train_pred_prob=xc1.predict_proba(feature_train)
        test_pred_prob=xc1.predict_proba(feature_test)
        stats[0,i]=regs[i]
        stats[1,i]=f1_score(target_train,train_pred)
        stats[2,i]=f1_score(target_test,test_pred)       
        stats[3,i]=log_loss(target_train,train_pred_prob)
        stats[4,i]=log_loss(target_test,test_pred_prob)
    return stats

def loop_logistic(feature_train,target_train,feature_test,target_test,regs):
    stats=np.zeros((5,len(regs)))
    for i in range(len(regs)):
        print(f"doing l2 regularization {regs[i]}") #does not always converge but are cases which are certainly not useful ones
        xc1=LogisticRegression(max_iter=10000,penalty='l2',C=regs[i]).fit(feature_train,target_train)
        train_pred=xc1.predict(feature_train)
        test_pred=xc1.predict(feature_test)
        train_pred_prob=xc1.predict_proba(feature_train)
        test_pred_prob=xc1.predict_proba(feature_test)
        stats[0,i]=regs[i]
        stats[1,i]=f1_score(target_train,train_pred)
        stats[2,i]=f1_score(target_test,test_pred)       
        stats[3,i]=log_loss(target_train,train_pred_prob)
        stats[4,i]=log_loss(target_test,test_pred_prob)
    return stats

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class ClassificationDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class CNNBinary4(torch.nn.Module):
    def __init__(self,keep_prob):
        super(CNNBinary4, self).__init__()
        # L1 ImgIn shape=(?, 43, 43, 1)
        # Conv -> (?, 41, 41, 16)
        # Pool -> (?, 20, 20, 16)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 20, 20, 16)
        # Conv      ->(?, 18, 18, 32)
        # Pool      ->(?, 9, 9, 32)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 9, 9, 32)
        # Conv      ->(?, 7, 7, 64)
        # Pool      ->(?, 3, 3, 64)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))        
        # L3 FC 3x3x64 inputs -> 128 outputs
        self.fc1 = torch.nn.Linear(3 * 3 * 64, 128, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight) # initialize parameters
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L4 Final FC 128 inputs -> 1 output
        self.fc2 = torch.nn.Linear(128, 1, bias=True) #
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out) #dont forget to add/omit layer here
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = torch.sigmoid(self.fc2(out))      #sigmoid because there only two classes  
        return out

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out) #dont forget to add/omit layer here
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = torch.sigmoid(self.fc2(out))      #sigmoid because there only two classes  
        return out


#parameters: model used m(need to be fined before), train_data, test_data, epochs, batch_size, 
#learning_rate, file for  collecting statistic, 
#optional regularization and whether  output is printed during running 
def torch_fit(model,train_loader,test_loader,epochs,batch_size,learning_rate,loss_stats,l2reg=0,silent=False):
    learning_rate = learning_rate
    criterion = torch.nn.BCELoss()    # Softmax is internally computed.
    #if no regularization
    if l2reg==0:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    #l2 regularization is added in optimizer as weight_decay
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate,weight_decay=l2reg)  
    if silent==False:    
        print("Begin training.")
    for e in tqdm(range(1, epochs+1)):
    
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
        
            y_train_pred = model(X_train_batch)
        
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        
            train_loss.backward()
            optimizer.step()
        
            train_epoch_loss += train_loss.item()
        
        
        # VALIDATION    
        with torch.no_grad():
        
            test_epoch_loss = 0
        
            model.eval()
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            
                y_test_pred = model(X_test_batch)
                        
                test_loss = criterion(y_test_pred, y_test_batch.unsqueeze(1))
            
                test_epoch_loss += test_loss.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['test'].append(test_epoch_loss/len(test_loader))                              
        if silent==False:
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Test Loss: {test_epoch_loss/len(test_loader):.5f}')     
            
def pred_torch(model,data):
    y_pred_list_c = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in data:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list_c.append(y_test_pred.cpu().numpy())
    y_pred_list_c = [a.squeeze().tolist() for a in y_pred_list_c]
    return y_pred_list_c   

#prediction, model (class), train data, etst data, train data for prediction, train targets, test targets, epochs , batch size, alpha of fit, 
#regularizations to try, number of feature (not neded for convolutional)
def run_loop_torch2(model,train,test,train_for_pred,train_target,test_target,epochs,batch,alpha,regs,num_features=0):
    #collects statistics f1 score and  log loss
    stats=np.zeros((5,len(regs)))
    for i in range(len(regs)):
        print(f"running reg of {regs[i]}")
        keep_prob=1
        if num_features==0:
            model3 =model()
        else:
            #num_features partlz needed
            model3 =model(num_features)            
        model3.to(device)
        loss_stats_test3 = {
        'train': [], 'test': []
        }
        #first with large regularization 
        print(f"initial run of high regularization")
        torch_fit(model3,train,test,20,batch,alpha,loss_stats_test3,l2reg=max(regs))
        print(f"run with given regularization")
        torch_fit(model3,train,test,epochs,batch,alpha,loss_stats_test3,l2reg=regs[i])
        test_pred=pred_torch(model3,test)
        train_pred=pred_torch(model3,train_for_pred)
        stats[0,i]=regs[i]
        stats[1,i]=f1_score(train_target,np.round(train_pred))
        stats[2,i]=f1_score(test_target,np.round(test_pred))
        stats[3,i]=log_loss(train_target,(train_pred))
        stats[4,i]=log_loss(test_target,(test_pred))   
        print(f"stats of l2reg of  {regs[i]} are {np.round(stats[1:5,i],5)}")
    print(f"full stats are {np.round(stats[:,:].T,5)}")
    return stats

#mlp networks with 4 layers 
#two output options 
class BinaryClassification4(nn.Module):
    def __init__(self, num_features):
        super(BinaryClassification4, self).__init__()
        self.fc1 = nn.Linear(num_features, 300)
        self.fc2 = nn.Linear(300, 100)  
        self.fc3 = nn.Linear(100, 30)        
        self.fc4 = nn.Linear(30, 10)   
        self.fc5 = nn.Linear(10, 1)          
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))        
        x = torch.sigmoid(self.fc5(x))
        return (x)
    
#parameters, model class,train data, test data, train data for prediction, test_tagert, train target, epochs of main fit
#batch isze, alpha iteration, l2/regularziations tried, optinial, initial iterations, optional starting regularziation, num_features if not convolutional
def run_loop_torch2b(model,train,test,train_for_pred,train_target,test_target,epochs,batch,alpha,regs,epoch_init=15,max_reg=0,num_features=0):
    #collects statistics f1 score and  log loss
    stats=np.zeros((8,len(regs)))
    for i in range(len(regs)):
        print(f"running reg of {regs[i]}")
        keep_prob=1
        if num_features==0:
            model3 =model()
        else:
            #num_features partly needed
            model3 =model(num_features)            
        model3.to(device)
        loss_stats_test3 = {
        'train': [], 'test': []
        }
        #first with large regularization 
        print(f"initial run of high regularization")
        if max_reg==0:
            torch_fit(model3,train,test,epoch_init,batch,alpha,loss_stats_test3,l2reg=max(regs))
        else:
            torch_fit(model3,train,test,epoch_init,batch,alpha,loss_stats_test3,l2reg=max_reg)
        print(f"run with given regularization")
        torch_fit(model3,train,test,epochs,batch,alpha,loss_stats_test3,l2reg=regs[i])
        test_pred=pred_torch(model3,test)
        train_pred=pred_torch(model3,train_for_pred)
        stats[0,i]=regs[i]
        stats[1,i]=f1_score(train_target,np.round(train_pred))
        stats[2,i]=f1_score(test_target,np.round(test_pred))
        stats[3,i]=log_loss(train_target,(train_pred))
        stats[4,i]=log_loss(test_target,(test_pred))   
        stats[5,i]=epochs
        if max_reg==0:
            stats[6,i]=max(regs[i])
        else:
            stats[6,i]=max_reg
        #was not there before
        stats[7,i]=epoch_init
        print(f"stats of l2reg of  {regs[i]} are {np.round(stats[1:5,i],5)}")
    print(f"full stats are {np.round(stats[:,:].T,5)}")            
    return stats      

#expand file of all version and apply changes that all look the same and have also two dimensions
#input filename 
def expand_2_output(filename):
    a=np.loadtxt(filename)
    spl=filename.split('_')
    spl2=spl[5].split('.')
    its=int(spl2[0])
    if a.shape[0]==8 and a.ndim==2:
        #change if zero 
        if a[7,0]==0:
            a[7,:]=15
        return a
    elif a.shape[0]==8 and a.ndim==1:
        b=np.zeros((8,1))
        if a[7]==0:
            a[7]=15
        b[0:8,0]=a
        return b
    elif  a.shape[0]==5 and a.ndim==2: 
        b=np.zeros((8,a.shape[1]))
        b[0:5,:]=a
        b[5,:]=its
        b[6,:]=np.max(a[0,:])
        b[7,:]=20 #was always 20 then
        return b
    elif  a.shape[0]==5 and a.ndim==1: 
        b=np.zeros((8,1))
        b[0:5,0]=a
        b[5,0]=its
        b[6,0]=a[0]
        b[7,0]=20 #was always 20 then
        return b
    
#input is list of np arrays,  and then what is minimised,  options are log_loss_test, log_loss_train, (minimised) f1_test and f1_train
#epochs (maximised)
def combine_fit_results(input_list,minimise="log_loss_test"):
    list_reg=[]
    for i in range(len(input_list)):
        if i==0:
            for j in range(input_list[i].shape[1]):
                list_reg.append(input_list[i][0,j])   
        if i>0:
            for j in range(input_list[i].shape[1]):
                par=input_list[i][0,j] in list_reg
                if par==False:
                    list_reg.append(input_list[i][0,j])   
    list_reg.sort()                     
    res=np.zeros((8,len(list_reg)))
    res[0,:]=list_reg
    for i in range(len(list_reg)):
        #getting identical events how many are there 
        c=0
        for j in range(len(input_list)):
            for k in range(input_list[j].shape[1]):
                if input_list[j][0,k]==list_reg[i]:
                    c+=1                    
        #make array and fill it then 
        array=np.zeros((8,c))    
        c=0
        for j in range(len(input_list)):
            for k in range(input_list[j].shape[1]):
                if input_list[j][0,k]==list_reg[i]:
                    array[:input_list[j].shape[0],c]=input_list[j][:input_list[j].shape[0],k]
                    c+=1                    
        #choose with different options
        if minimise=='log_loss_test':
            res[1:8,i]=array[1:8,np.argmin(array[4,:])]
        if minimise=='log_loss_train':
            res[1:8,i]=array[1:8,np.argmin(array[3,:])]    
        if minimise=='f1_train':
            res[1:8,i]=array[1:8,np.argmax(array[1,:])]    
        if minimise=='f1_test':
            res[1:8,i]=array[1:8,np.argmax(array[2,:])]   
        if minimise=='epochs':
            res[1:8,i]=array[1:8,np.argmax(array[5,:])]      
    return res    

def plot_image(image,target,x):
    plt.axis('off')
    plt.title(f"spiral is {bool(target[x])}")
    plt.imshow(abs(image[x,0,:,:])**0.5,cmap=cm.gray, interpolation='nearest')
    
#image_set, true class, better prediction, list of images wanted, single or several prediction, 
#other prediction (can be omitted),  display scale default 0.5, optiional minium difference in value 
def plot_bad_images(image,target,prediction1, x,single=False,prediction2=0,scale=0.5,lim=0,silent=True):
    #print(x_test_pred.shape,len(c_test_pred))
    list_bad=[]
    for i in range(len(prediction1)):
        if single==False:
            if round(prediction1[i])==target[i] and round(prediction1[i])==round(prediction2[i]) and abs(prediction2[i]-target[i])>lim:
                list_bad.append(i)
        else:
            if round(prediction1[i])!=target[i] and abs(prediction1[i]-target[i])>lim:
                list_bad.append(i)
    #print(list_bad)        
    plt.axis('off')
    dimage=np.zeros((43,len(x)*43+len(x)-1))
    string=''
    for i in range(len(x)):
        dimage[:,0+i*44:43+i*44]=(abs(image[list_bad[x[i]],0,:,:])**scale)/(np.max(abs(image[list_bad[x[i]],0,:,:])**scale))    
        if silent==False:
            print(list_bad[x[i]])     
        if bool(target[list_bad[x[i]]])==False and i!=len(x)-1:
            string+='elliptical          '
        elif bool(target[list_bad[x[i]]])==False and i==len(x)-1:
            string+='elliptical'            
        elif  bool(target[list_bad[x[i]]])==True and i!=len(x)-1:
            string+='spiral              '
        else:
            string+='spiral'            
    plt.title(string)     
    plt.imshow(dimage,cmap=cm.gray, interpolation='nearest')    
    

def plot_images(image,target,prediction1, x,single=False,prediction2=0,scale=0.5,lim=0,silent=True, different=True):
    #print(x_test_pred.shape,len(c_test_pred))
    list_bad=[]
    for i in range(len(prediction1)):
        if single==False and different==True:
            if round(prediction1[i])==target[i] and round(prediction1[i])==round(prediction2[i]) and abs(prediction2[i]-target[i])>lim:
                list_bad.append(i)
        elif single==True and different==True:
            if round(prediction1[i])!=target[i] and abs(prediction1[i]-target[i])>lim:
                list_bad.append(i)
        elif single==False and different==False:
            if round(prediction1[i])!=target[i] and round(prediction1[i])!=round(prediction2[i]):
                list_bad.append(i)  
        elif single==True and different==False and abs(prediction1[i]-target[i])<1-lim:
            if round(prediction1[i])==target[i]:
                list_bad.append(i)                       
    plt.axis('off')
    dimage=np.zeros((43,len(x)*43+len(x)-1))
    string=''
    for i in range(len(x)):
        dimage[:,0+i*44:43+i*44]=(abs(image[list_bad[x[i]],0,:,:])**scale)/(np.max(abs(image[list_bad[x[i]],0,:,:])**scale))    
        if silent==False:
            print(list_bad[x[i]])     
        if bool(target[list_bad[x[i]]])==False and i!=len(x)-1:
            string+='elliptical          '
        elif bool(target[list_bad[x[i]]])==False and i==len(x)-1:
            string+='elliptical'            
        elif  bool(target[list_bad[x[i]]])==True and i!=len(x)-1:
            string+='spiral              '
        else:
            string+='spiral' 
    if different==True:  
        plt.title(string,color='red') 
    else:  
        plt.title(string,color='green')         
    plt.imshow(dimage,cmap=cm.gray, interpolation='nearest')   
    
def get_rot_mirror_square(dat):
    if dat.shape[0]!=dat.shape[1]:
        print("Data is not a square")
    else:
        res=np.zeros((8,dat.shape[0],dat.shape[1]))
        res[0,:,:]=dat
        res[4,:,:]=np.flip(dat,0)
        for i in range(3):
            res[1+i,:,:]=np.rot90(dat,k=i+1,axes=(0,1))
            res[5+i,:,:]=np.rot90(res[4,:,:],k=i+1,axes=(0,1))
        return res     
 
def shuffle_ar_list_df(df,series,array):
    if isinstance(series, pd.core.series.Series):    
        series=series.to_list()
    x2=list(range(array.shape[0]))
    random.seed(189)
    random.shuffle(x2)

    newarray=np.zeros((array.shape[0],array.shape[1],array.shape[2],array.shape[3]))
    new_list=series.copy()
    for j in range(len(x2)):
        new_list[x2[j]]=series[j]
        newarray[x2[j],:,:,:]=array[j,:,:,:]
    ndf=df.copy()    
    ndf['new_index']=x2
    ndf=ndf.set_index('new_index')
    ndf=ndf.sort_index()
    return ndf, new_list, newarray

#input image cube, target for them , data frame, default shuffle (likely right for train but not for test)
#problem when not shuffled, but why? 
def  get_rot_mirror_all(image, target,df,shuffle=True):
    image_all=np.zeros((image.shape[0]*8,1,image.shape[2],image.shape[3]))
    new_targ=[]
    #rename column to be used new 
    df.rename(columns={0:'mirror_rot'}, inplace=True)
    ncol=df.columns
    print(ncol)
    #fill data frame with one data set 
    ndf= pd.DataFrame(data=df, columns=ncol)
    print(ndf.shape)
    list_df=[ndf,ndf,ndf,ndf,ndf,ndf,ndf,ndf]
    #multiply by 8 
    ndf=pd.concat(list_df,ignore_index=True)
    #convert target to numpy 
    target=np.array(target)
    print(ndf.shape,ndf.columns,target.shape)
    c=0
    for i in range(image.shape[0]):
        print(i)
        image_all[0+i*8:8+8*i,0,:,:]=get_rot_mirror_square(image[i,0,:,:]) 
        for j in range(8):
            new_targ.append(np.array(target[i]))
            #is slow the filling but still acceptable 
            ndf.iloc[c,0:df.shape[1]-1]=df.iloc[i,0:df.shape[1]-1]
            ndf.iloc[c,df.shape[1]-1]=j
            c+=1
    #adding image values to data frame
    x=0
    for i in range(image_all.shape[2]):
        for j in range(image_all.shape[3]):
            ndf[x]=image_all[:,0,i,j]
            x+=1
    #shuffle all in the same way  
    if shuffle==True:
        ndf,new_targ,image_all=shuffle_ar_list_df(ndf,new_targ,image_all)        
    return image_all,new_targ, ndf 

#full probailities file, how many are combined, mode mean or median
#assume 1d np array input
def comb_entries(probs,mult,avg=True):
    if probs.ndim==1:   
        nprobs=np.zeros((int(probs.shape[0]/mult)))
        for i in range(nprobs.shape[0]):
            if avg==True:
                nprobs[i]=np.mean(probs[0+i*mult:mult*(1+i)])
            if avg==False:
                nprobs[i]=np.median(probs[0+i*mult:mult*(1+i)])    
    if probs.ndim==2:   
        nprobs=np.zeros((int(probs.shape[0]/mult),2))
        for i in range(nprobs.shape[0]):
            if avg==True:
                nprobs[i,0]=np.mean(probs[0+i*mult:mult*(1+i),0])
                nprobs[i,1]=1-nprobs[i,0]
            if avg==False:
                nprobs[i,0]=np.median(probs[0+i*mult:mult*(1+i),0])  
                nprobs[i,1]=1-nprobs[i,0]       
    return nprobs


def loop_xgboost_pred(models,feature_train,target_train,feature_test,target_test,regs,orig=0,mult=8):
    stats=np.zeros((5,len(regs)))
    for i in range(len(regs)):
        print(f"doing l2 regularization {regs[i]}")

        xc1=XGBClassifier()
        xc1.load_model(models[i])
        if orig==0:
            train_pred=xc1.predict(feature_train)
            test_pred=xc1.predict(feature_test)
            train_pred_prob=xc1.predict_proba(feature_train)
            test_pred_prob=xc1.predict_proba(feature_test) 
        if orig==1 or orig==2:
            train_pred=xc1.predict(feature_train)
            train_pred_prob=xc1.predict_proba(feature_train)
            test_pred_prob2=xc1.predict_proba(feature_test)
            if orig==1:
                test_pred_prob=comb_entries(test_pred_prob2,mult=mult,avg=True)
                test_pred=(np.round(test_pred_prob[:,1]))
            if orig==2:
                test_pred_prob=comb_entries(test_pred_prob2,mult=mult,avg=False)
                test_pred=(np.round(test_pred_prob[:,1]))     
        stats[0,i]=regs[i]
        stats[1,i]=f1_score(target_train,train_pred)
        stats[2,i]=f1_score(target_test,test_pred)       
        stats[3,i]=log_loss(target_train,train_pred_prob)
        stats[4,i]=log_loss(target_test,test_pred_prob)
    return stats    

#mode0 single, 1 mean, 2 median
def predict_torch2(model_list,model,test,train_for_pred,train_target,test_target,regs,num_features=0,keep_prob=1,mode=0,mult=8):
    #collects statistics f1 score and  log loss
    stats=np.zeros((5,len(regs)))
    for i in range(len(regs)):
        print(f"{regs[i]}")
        if num_features==0:
            model3 =model()
        else:
            #num_features partlz needed
            model3 =model(num_features)  
        print(keep_prob)    
        model3.load_state_dict(torch.load(model_list[i]))
        model3.eval()
        train_pred=pred_torch(model3,train_for_pred)
        if mode==0:
            test_pred=pred_torch(model3,test)
        if mode==1:    
            test_pred2=np.array(pred_torch(model3,test))
            test_pred=comb_entries(test_pred2,mult=mult,avg=True)
        if mode==2:    
            test_pred2=np.array(pred_torch(model3,test))
            test_pred=comb_entries(test_pred2,mult=mult,avg=False)            
        stats[0,i]=regs[i]
        stats[1,i]=f1_score(train_target,np.round(train_pred))
        stats[2,i]=f1_score(test_target,np.round(test_pred))
        stats[3,i]=log_loss(train_target,(train_pred))
        stats[4,i]=log_loss(test_target,(test_pred))   
        print(f"stats of l2reg of  {regs[i]} are {np.round(stats[1:5,i],5)}")
    print(f"full stats are {np.round(stats[:,:].T,5)}")
    return stats

def get_noise(im,option="std"):
    res=np.zeros((im.shape[0],im.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if option=="std":
                res[i,j]=np.std(im[i,j,:,:])
            elif option=="IQR":
                res[i,j]=np.quantile(im[i,j,:,:], 0.75)-np.quantile(im[i,j,:,:], 0.25)
    return res    

def add_noise_image(image,noise_level):
    ran=np.random.normal(0, noise_level, size=(image.shape[0],image.shape[1]))
    ran+=image
    return ran   

def  get_noise_all(image, target,df,nmult=8,noise_level=6*0.761):
    image_all=np.zeros((image.shape[0]*nmult,1,image.shape[2],image.shape[3]))
    new_targ=[]
    #rename column to be used new 
    ncol=df.columns
    print(ncol)
    #fill data frame with one data set 
    ndf= pd.DataFrame(data=df, columns=ncol)
    print(ndf.shape)
    list_df=[]
    for j in range(nmult):
        list_df.append(ndf)
    #multiply by x
    ndf=pd.concat(list_df,ignore_index=True)
    print(ndf.shape,ndf.columns,target.shape)
    c=0
    for i in range(image.shape[0]):
        print(i)
        for j in range(nmult):
            new_targ.append(target[i])
            image_all[c,0,:,:]=add_noise_image(image[i,0,:,:],noise_level)
            #is slow the filling but still acceptable 
            ndf.iloc[c,0:df.shape[1]-1]=df.iloc[i,0:df.shape[1]-1]
            c+=1
    #adding image values to data frame
    x=0
    for i in range(image_all.shape[2]):
        for j in range(image_all.shape[3]):
            ndf[x]=image_all[:,0,i,j]
            x+=1
    #shuffle all in the same way         
    ndf,new_targ,image_all=shuffle_ar_list_df(ndf,new_targ,image_all)        
    return image_all,new_targ, ndf  

#list of models, model-class, test-sets,train-sets, target_train, target-test, regulizations,
#num-features(for mlp) , optional list of information fiels, op list of iterations , optional keep prob to select
def predict_torch2b(model_list,model,test,train_for_pred,train_target,test_target,regs,num_features=0,keep_prob=1,mode=0,mult=8,infolist=[],itlist=[]):
    #collects statistics f1 score and  log loss
    stats=np.zeros((8,len(regs)))
    #stats2=pd.DataFrame([1, 2, 3], index=["a", "b", "c"], columns=["x"]) later use more 
    for i in range(len(regs)):
        print(f"{regs[i]}")
        if len(infolist)!=0:
            if infolist[i]!=None:
                ta=pd.read_csv(infolist[i])
                print(ta.head())
                stats[2,i]=ta.alpha[0]
                stats[3,i]=ta.keep_prob[0]
            else:
                stats[2,i]=0.001
                stats[3,i]=1
        if num_features==0:
            model3 =model()
        else:
            #num_features partlz needed
            model3 =model(num_features)  
        print(keep_prob)    
        model3.load_state_dict(torch.load(model_list[i]))
        model3.eval()
        train_pred=pred_torch(model3,train_for_pred)
        if mode==0:
            test_pred=pred_torch(model3,test)
        if mode==1:    
            test_pred2=np.array(pred_torch(model3,test))
            test_pred=comb_entries(test_pred2,mult=mult,avg=True)
        if mode==2:    
            test_pred2=np.array(pred_torch(model3,test))
            test_pred=comb_entries(test_pred2,mult=mult,avg=False)            
        stats[0,i]=regs[i]
        if len(itlist)>0:
            stats[1,i]=itlist[i]
        stats[4,i]=f1_score(train_target,np.round(train_pred))
        stats[5,i]=f1_score(test_target,np.round(test_pred))
        stats[6,i]=log_loss(train_target,(train_pred))
        stats[7,i]=log_loss(test_target,(test_pred))   
        print(f"stats of l2reg of  {regs[i]} are {np.round(stats[1:5,i],5)}")
    print(f"full stats are")
    print(np.round(stats[:,:].T,5))
    return stats

#input file made with predict_torch2b, keep_prob to select and what is minimized
def combine_fit_results2(input_file,keep_prob,list_model,minimise="log_loss_train"):
    list_reg=[]
    for i in range(0,input_file.shape[1]):
        if len(list_reg)==0 and input_file[3,i]==keep_prob:
            list_reg.append(input_file[0,i])   
        else:
            par=input_file[0,i] in list_reg
            if par==False and input_file[3,i]==keep_prob:
                list_reg.append(input_file[0,i])   
    list_reg.sort()                     
    res=np.zeros((9,len(list_reg)))
    res[0,:]=list_reg
    for i in range(len(list_reg)):
        #getting identical events how many are there 
        c=0
        for j in range(input_file.shape[1]):
            if input_file[0,j]==list_reg[i] and input_file[3,j]==keep_prob:
                c+=1                    
        #make array and fill it then 
        array=np.zeros((9,c))    
        c=0
        for j in range(input_file.shape[1]):
            if input_file[0,j]==list_reg[i] and input_file[3,j]==keep_prob:         
                array[0:8,c]=input_file[:,j]
                array[8,c]=j
                c+=1                    
        #choose with different options
        if minimise=='log_loss_test':
            res[1:8,i]=array[1:8,np.argmin(array[7,:])]
            res[8,i]=array[8,np.argmin(array[7,:])]
        if minimise=='log_loss_train':
            res[1:8,i]=array[1:8,np.argmin(array[6,:])]  
            res[8,i]=array[8,np.argmin(array[6,:])]
        if minimise=='f1_train':
            res[1:8,i]=array[1:8,np.argmax(array[4,:])]  
            res[8,i]=array[8,np.argmin(array[4,:])]            
            
        if minimise=='f1_test':
            res[1:8,i]=array[1:8,np.argmax(array[5,:])]   
            res[8,i]=array[8,np.argmin(array[5,:])]            
        if minimise=='epochs':
            res[1:8,i]=array[1:8,np.argmax(array[1,:])]    
            res[8,i]=array[8,np.argmin(array[1,:])]            
    #select the best models
    list_select=[]
    for i in range(res.shape[1]):
        list_select.append(list_model[int(res[8,i])])
    return res[0:8,:], list_select


#should define colu,mns used backwards for image since not all have the the same size 
#parameters, list of images, list of data frames with classes, model name, iamge output , frame output? 
def predict_probs(images,classes,model,modelname='convolutional',keep_prob=1,num_features=1849,image_output=True,df_output=True, split=1,train_choice=True,seed=1):
    cutouts_new=comb_nump_4d(images).T
    print(cutouts_new.shape)
    list_df2=[]
    for i in range(len(classes)):
        i=pd.read_csv(classes[i])
        list_df2.append(i)  
    print(f"number of tables is {len(list_df2)}") 
    df2=pd.concat(list_df2,ignore_index=True)
    print(f"shape of combined data frame {df2.shape}")
    print(f"shape of image file is {cutouts_new.shape}")
    #below needed? 
    if model!='convolutional':
        x=0
        for i in range(cutouts_new.shape[2]):
            for j in range(cutouts_new.shape[3]):
                df2[x]=cutouts_new[:,0,i,j]
                x+=1
    #if only test or train subset done                 
    if split<1:
        image_train,image_test,df_train,df_test= train_test_split(cutouts_new,df2,train_size=split, shuffle=True, random_state=seed)
        if train_choice==True:
            cutouts_new=image_train 
            df2=df_train
        else:
            cutouts_new=image_test 
            df2=df_test            
    print(df2.shape)
    print(df2.columns)
    # df2.iloc[:,0] is dummy for target 
    image_rot,target_rot,df_rot=get_rot_mirror_all(cutouts_new,df2.iloc[:,0],df2.iloc[:,0:52],shuffle=False)
    if modelname=='xgboost':
        #overwrite not use ones
        #image_rot=0
        target_rot=0
        xgb_reg=XGBClassifier()
        xgb_reg.load_model(model)
        pred_si=xgb_reg.predict_proba(df_rot.iloc[:,-1849:])
        #combine the 8 rotation entries 
        predboth=comb_entries(pred_si,8,avg=True)
        pred=predboth[:,1]
        print(df_rot.columns[52:1901],pred.shape)
    if modelname=='convolutional':
        #df_rot=0
        #setup data for torch 
        train_imrot_dataset = ClassificationDataset(torch.from_numpy(image_rot).float(), torch.from_numpy(np.array(target_rot)).float())
        train_imrot_loader_pred = DataLoader(dataset=train_imrot_dataset, batch_size=1)
        keep_prob=keep_prob
        model_convfin =CNNBinary4(keep_prob)
        model_convfin.load_state_dict(torch.load(model))
        model_convfin.eval()

        pred_si=pred_torch(model_convfin,train_imrot_loader_pred)
        pred=comb_entries(np.array(pred_si),8,avg=True)    
    if modelname=='perceptron':
        #df_rot=0
        #setup data for torch 
        train_rot_dataset = ClassificationDataset(torch.from_numpy(np.array(df_rot.iloc[:,-1849:])).float(), torch.from_numpy(np.array(target_rot)).float())
        #old train_rot_dataset = ClassificationDataset(torch.from_numpy(np.array(df_rot.iloc[:,52:1901])).float(), torch.from_numpy(np.array(target_rot)).float())
        train_rot_loader_pred = DataLoader(dataset=train_rot_dataset, batch_size=1)
        model_perfin =BinaryClassification4(num_features)
        model_perfin.load_state_dict(torch.load(model))
        model_perfin.eval()

        pred_si=pred_torch(model_perfin,train_rot_loader_pred)
        pred=comb_entries(np.array(pred_si),8,avg=True)         
    if image_output==True and df_output==True:    
        return pred, df2, cutouts_new
    elif image_output==False and df_output==True:    
        return pred, df2
    elif image_output==True and df_output==False:    
        return pred, cutouts_new
    else:
        return pred    

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

#getting the list of images
myPath='/home/tobias/ml-testing/astr-images'
list_images=[f for f in os.listdir(myPath) 
    if f.endswith('_ell_spiral_im.npy') ]
list_images.sort()
print(list_images)
#ggeti9ng the list of tables 
list_tables=[f for f in os.listdir(myPath) 
    if f.endswith('_ell_spiral_table.csv')]
list_tables.sort()

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

cutouts=comb_nump_4d(list_images)

list_df=[]
for i in range(len(list_tables)):
    i=pd.read_csv(list_tables[i])
    list_df.append(i)  
    
df=pd.concat(list_df,ignore_index=True)

x=0
for i in range(cutouts.shape[0]):
    for j in range(cutouts.shape[1]):
        df[x]=cutouts[i,j,0,:]
        x+=1

feature_train, feature_test, target_train, target_test,image_train,image_test,df_train,df_test= train_test_split(df.iloc[:,51:1900],df.loc[:,"spiral"],cutouts.T,df,train_size=0.60, shuffle=True, random_state=1)

#adding cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class ClassificationDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
target_train, target_test = np.array(target_train), np.array(target_test)
feature_train, feature_test = np.array(feature_train), np.array(feature_test)
train_im_dataset = ClassificationDataset(torch.from_numpy(image_train).float(), torch.from_numpy(target_train).float())
test_im_dataset = ClassificationDataset(torch.from_numpy(image_test).float(), torch.from_numpy(target_test).float())
train_dataset = ClassificationDataset(torch.from_numpy(feature_train).float(), torch.from_numpy(target_train).float())
test_dataset = ClassificationDataset(torch.from_numpy(feature_test).float(), torch.from_numpy(target_test).float())
    
    
BATCH_SIZE=32
train_im_loader = DataLoader(dataset=train_im_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_im_loader = DataLoader(dataset=test_im_dataset, batch_size=1)
train_im_loader_pred = DataLoader(dataset=train_im_dataset, batch_size=1)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)
train_loader_pred = DataLoader(dataset=train_dataset, batch_size=1)

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
    
#prediction, model (class), train data, etst data, train data for prediction, train targets, test targets, epochs , batch size, alpha of fit, 
#regularizations to try, number of feature (not neded for convolutional)
def run_loop_torch2(model,train,test,train_for_pred,train_target,test_target,epochs,batch,alpha,regs,num_features=0):
    #collects statistics f1 score and  log loss
    stats=np.zeros((5,len(regs)))
    for i in range(len(regs)):
        print(f"running reg of {regs[i]}")
        keep_prob=1
        if num_features==0:
            model3 =model(keep_prob)
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
    
def run_loop_torch2b(model,train,test,train_for_pred,train_target,test_target,epochs,batch,alpha,regs,epoch_init=15,max_reg=0,num_features=0):
    #collects statistics f1 score and  log loss
    stats=np.zeros((8,len(regs)))
    for i in range(len(regs)):
        print(f"running reg of {regs[i]}")
        keep_prob=1
        if num_features==0:
            model3 =model(keep_prob)
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
#now raotation mirror stuff

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

#input image cube, target for them , data frame
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
    print(ndf.shape,ndf.columns,target.shape)
    c=0
    for i in range(image.shape[0]):
        print(i)
        image_all[0+i*8:8+8*i,0,:,:]=get_rot_mirror_square(image_train[i,0,:,:])
        for j in range(8):
            new_targ.append(target[i])
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

start_time=time.time()
rot_image_train,rot_target_train,rot_df_train=get_rot_mirror_all(image_train,target_train,df_train.iloc[:,0:52])
stop_time=time.time()
print(f"snippet needed {np.round(stop_time-start_time,3)} seconds")

#keep_prob=1

def fit_model(model=0,modeltype="perceptron",l2reg=0.01,end_epochs=200,keep_prob=1,start_epochs=15,start_reg=0.3,max_depth=6,alpha=0.001,num_features=1849,rot_image_train=rot_image_train,rot_target_train=rot_target_train,rot_df_train=rot_df_train):
    #options perceptron  convolutional  xgboost
    if modeltype=="perceptron":
        #overwrite to free memory 
        rot_image_train=0
        rot_target_train = np.array(rot_target_train)
        rot_df_train = np.array(rot_df_train.iloc[:,52:1901])
        train_rot_dataset = ClassificationDataset(torch.from_numpy(rot_df_train).float(), torch.from_numpy(rot_target_train).float())
        train_rot_loader = DataLoader(dataset=train_rot_dataset, batch_size=BATCH_SIZE, shuffle=True) #shuffle means that algorithm shuffle was not needed

        start_time=time.time()
        model3 =model(num_features)   #no keep prob yet included     
        model3.to(device)
        loss_stats_test3 = {    'train': [], 'test': []}
        best_reg=l2reg
        df1 = pd.DataFrame([[modeltype, start_reg, start_epochs,l2reg,end_epochs,alpha,keep_prob,num_features]], columns=["modeltype","start_reg", "start_epochs","end_reg","end_epochs","alpha","keep_prob","num_features"])
        print(df1)
        path='/home/tobias/ml-testing/astr-images/'
        strings='mlp_4layers_reg'+str(best_reg)+'_rotmir_'+str(end_epochs)+'epochs'
        list_input_files=[f for f in os.listdir(path) 
        if f.endswith("_info.csv") and f.startswith(strings) and os.path.isfile(os.path.join(path, f))]
        list_input_files.sort()
        print(len(list_input_files))
        string='mlp_4layers_reg'+str(best_reg)+'_rotmir_'+str(end_epochs)+'epochs_v'+str(len(list_input_files))+'_info.csv'        
        
        print(f"run with initial regularization")
        #test is not rotated 
        torch_fit(model3,train_rot_loader,test_loader,start_epochs,BATCH_SIZE,alpha,loss_stats_test3,l2reg=start_reg)
        print(f"run with given regularization")
        torch_fit(model3,train_rot_loader,test_loader,end_epochs,BATCH_SIZE,alpha,loss_stats_test3,l2reg=best_reg)
        PATH='/home/tobias/ml-testing/astr-images/mlp_4layers_reg'+str(best_reg)+'_rotmir_'+str(end_epochs)+'epochs_v'+str(len(list_input_files))+'.pkl'
        torch.save(model3.state_dict(), PATH)
        df1.to_csv(string) 
        stop_time=time.time() 
        print(f"snippet needed {np.round(stop_time-start_time,3)} seconds")
        pass
        #perceptrons needs about  4176.255 seconds
    
    elif modeltype=="xgboost":
        #overwrite to free memory 
        rot_image_train=0
        df1 = pd.DataFrame([[modeltype,l2reg,max_depth]], columns=["modeltype","l2_reg", "max_depth"])
        print(df1)
        path='/home/tobias/ml-testing/astr-images/'
        strings='xgboost_reg'+str(l2reg)+'_rotmir_'
        list_input_files=[f for f in os.listdir(path) 
        if f.endswith("_info.csv") and f.startswith(strings) and os.path.isfile(os.path.join(path, f))]
        list_input_files.sort()
        print(len(list_input_files))
        string='xgboost_reg'+str(l2reg)+'_rotmir_v'+str(len(list_input_files))+'_info.csv'  
        start_time=time.time()
        xgb_reg=XGBClassifier(max_depth=max_depth,reg_lambda=l2reg).fit(rot_df_train.iloc[:,52:1901],rot_target_train) 
        xgb_reg.save_model("xgboost_model_spiral_ell_rot_mirr_"+str(l2reg)+"_v"+str(len(list_input_files))+".json")
        df1.to_csv(string)         
        stop_time=time.time()
        print(f"snippet needed {np.round(stop_time-start_time,3)} seconds")
        pass
        #xgboost snippet needed 532.151 seconds
        
    elif modeltype=="convolutional":
        #overwrite to free memory 
        rot_df_train=0
        best_reg=l2reg
        df1 = pd.DataFrame([[modeltype, start_reg, start_epochs,l2reg,end_epochs,alpha,keep_prob]], columns=["modeltype","start_reg", "start_epochs","end_reg","end_epochs","alpha","keep_prob"])
        print(df1)
        path='/home/tobias/ml-testing/astr-images/'
        strings='conv2d_2layers_reg'+str(best_reg)+'_rotmir_'+str(end_epochs)+'epochs'
        list_input_files=[f for f in os.listdir(path) 
        if f.endswith("_info.csv") and f.startswith(strings) and os.path.isfile(os.path.join(path, f))]
        list_input_files.sort()
        print(len(list_input_files))
        string='conv2d_2layers_reg'+str(best_reg)+'_rotmir_'+str(end_epochs)+'epochs_v'+str(len(list_input_files))+'_info.csv'    
        
        rot_target_train = np.array(rot_target_train)
        train_imrot_dataset = ClassificationDataset(torch.from_numpy(rot_image_train).float(), torch.from_numpy(rot_target_train).float())
        train_imrot_loader = DataLoader(dataset=train_imrot_dataset, batch_size=BATCH_SIZE, shuffle=True)    
        
        start_time=time.time()
        model3 =model(keep_prob)        
        model3.to(device)
        loss_stats_test3 = {    'train': [], 'test': []}
        #first with large regularization 
        print(f"keep prob is {keep_prob} ")
        print(model3)
        print(f"run with initial regularization")
        #test is not rotated 
        torch_fit(model3,train_imrot_loader,test_im_loader,start_epochs,BATCH_SIZE,alpha,loss_stats_test3,l2reg=start_reg)
        print(f"run with given regularization")
        torch_fit(model3,train_imrot_loader,test_im_loader,end_epochs,BATCH_SIZE,alpha,loss_stats_test3,l2reg=best_reg)
        PATH='/home/tobias/ml-testing/astr-images/conv2d_2layers_reg'+str(best_reg)+'_rotmir_'+str(end_epochs)+'epochs_v'+str(len(list_input_files))+'.pkl'
        torch.save(model3.state_dict(), PATH)
        df1.to_csv(string)
        stop_time=time.time()

        print(f"snippet needed {np.round(stop_time-start_time,3)} seconds")
        pass
        #   snippet needed 5733.833 seconds
        #w keep_prob=0.75 snippet needed 8084.415 seconds


#fit_model(model=BinaryClassification4,modeltype="perceptron",l2reg=0.001,end_epochs=200,keep_prob=1,start_epochs=15,alpha=0.0005)
fit_model(model=CNNBinary4,modeltype="convolutional",l2reg=0.0001,end_epochs=240,keep_prob=0.75,start_epochs=15)

#BinaryClaass 0.01 alpha seems worse if no other mistake 
#slow also alpha 0.002 normal maybe in that network 
#or mistake, biut i cannot see it.
#0.0005 seems better overfittimng need automic stopping 
#adapting keep_prob works for CNNBinary4 no or little overfitting with below
# from 80 to 240 at most a ltlle improvement
#fit_model(model=CNNBinary4,modeltype="convolutional",l2reg=0.0001,end_epochs=240,keep_prob=0.75,start_epochs=15)
#shoudl implement some abort method 

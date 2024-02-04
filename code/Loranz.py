import pandas as pd
import numpy as np
import pyvqnet
from torch.utils.data import DataLoader, Dataset
from pyvqnet.tensor import pad_sequence, pad_packed_sequence, pack_pad_sequence
from pyvqnet.tensor import tensor
from pyvqnet.optim.adam import Adam
from pyvqnet.optim import rmsprop
from pyvqnet.nn.loss import CategoricalCrossEntropy,CrossEntropyLoss,MeanSquaredError
from pyvqnet.nn.loss import BinaryCrossEntropy
import models
import time
from pyvqnet import *
from pyvqnet.tensor import QTensor
import matplotlib.pyplot as plt
from pyvqnet.utils.storage import save_parameters,load_parameters
from pyvqnet.utils.storage import load_optim, load_parameters
import io
import datetime
import os
import scipy.special as spl
from pyvqnet.optim import adagrad
from scipy.integrate import odeint
import qpandalite.task.originq_dummy as od
import json
import pickle
# 定义保存模型和参数的函数


# 定义Lorenz方程
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state  # 解包状态向量
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # 返回导数向量

def get_mse(result,label):
    result,label = np.array(result.data), np.array(label.data)
    mse = np.mean(np.power(result - label, 2))
    return mse

class MyData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'text':self.data[idx], 'label':self.label[idx]}
    
def collate_fn(mydata):
    t=[]
    l=[]
    for i in mydata:
        t.append(tensor.to_tensor(i['text']))
        l.append(i['label'])
    res={} 
    res['text'] = pad_sequence(t,batch_first=True)

    l1 = QTensor(np.array(l).astype('float32'))
    res['label'] = l1
    res['seqlen'] = 20
    return res

def train_model(taskname,rho = 28.0,train_set = 800):
    if(type(rho)!=float) or (type(train_set)!=int):
        raise TimeoutError("rho must be float and train_set must be int!")
    if(type(taskname)!=str):
        taskname = str(taskname)
    

    state0 = [1.0, 1.0, 1.0]
    sigma = 10.0
    beta = 8.0 / 3.0
    t = np.arange(0.0, 80.0, 0.02)
    states = odeint(lorenz, state0, t, args=(sigma, rho, beta))



    N = 20
    input20 =[]
    lable1 = []
    for i in range(len(states)-N):
        temp=[]
        for j in range(N):
            temp.append(states[i+j])

        input20.append(temp)
        lable1.append(states[i+N])

    train_dataset = MyData(input20[:train_set],lable1[:train_set])
    train_len = len(train_dataset)
    all_dataset = MyData(input20[:4000],lable1[:4000])
    all_len = len(all_dataset)
    val_dataset = MyData(input20[train_set:train_set+400],lable1[train_set:train_set+400])
    val_len = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, drop_last=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=True,collate_fn=collate_fn)
    all_loader = DataLoader(all_dataset, batch_size=20, shuffle=False, drop_last=True,collate_fn=collate_fn)

    model = models.RegLSTM_hard(input_sz=3,hidden_sz=5,vqc_circuit_lite=models.vqc_circuit_lite_hard)


    main_epoch = 100
    learning_rate = 0.01
    saving_pridict_flag = False



    optimizer = rmsprop.RMSProp(model.parameters(),lr=learning_rate,beta=0.9)

    CCEloss =MeanSquaredError()


    tr_mses = []
    val_mses = []
    tr_losses = []
    val_losses = []
    best_loss = 9999.0
    best_loss_epoch = -1

    now = datetime.datetime.now()



    if not os.path.exists(taskname):
        os.mkdir(taskname)
        modeldirPath = os.path.join(taskname, "models")
        os.mkdir(modeldirPath)           
        optimdirPath = os.path.join(taskname, "optims")
        os.mkdir(optimdirPath)
        premodelPath = os.path.join(taskname, "model_history")
        os.mkdir(premodelPath)
        picsPath = os.path.join(taskname, "pics")
        os.mkdir(picsPath)
        checkpointPath = os.path.join(taskname, "checkpoint")
        os.mkdir(checkpointPath)
    train_info_file_name = "train_model_info.txt"
    

    if_resume = True 
    check_epoch = 0 
    check_step = 0
    manual_reload = False
    flag=True
    com_counter = 0

    modelsavingpath = os.path.join(taskname, "checkpoint","model.model")
    optimsavingpath = os.path.join(taskname, "checkpoint","optim.optim")
    while(flag):
        flag = False 
        if com_counter == 0 and manual_reload == False:
            print("This is the first loop!")
            com_counter += 1
            check_epoch = 0
            check_step = 0
            
        elif manual_reload == True:
            print("Begin manual reload!")
            if_resume = True

            check_epoch = 0
            check_step = 6

            com_counter = 1
            model_para = load_parameters(modelsavingpath)
            model.load_state_dict(model_para)    
            optim_para = load_optim(optimsavingpath)
            optimizer.load_state_dict(optim_para)
            manual_reload = False

        else:
            print("This is the {} loop,there were {} times failure ".format(com_counter+1,com_counter))
            if_resume = False
            com_counter += 1

        
        if if_resume is False:
            print("Begin reload modle and optimizer!")
            model_para = load_parameters(modelsavingpath)
            model.load_state_dict(model_para)    
            optim_para = load_optim(optimsavingpath)
            optimizer.load_state_dict(optim_para)
                
            with open('program_state.pkl', 'rb') as file:
                program_state = pickle.load(file)
                check_epoch = program_state["epoch"]
                check_step = program_state["step"]

        try:
            for epoch in range(check_epoch,main_epoch):
                epoch_mse = 0
                tr_loss = 0
                
            #==============================================================================================
                for step, train_data in enumerate(train_loader):
                    if epoch == check_epoch and step < check_step:
                        print("epoch:{} step:{} have been computed".format(epoch,step))
                        continue
                    print("epoch:{},step:{}".format(epoch,step))
                    saving_pridict_flag = False
                    tstart = time.time()
                    od.dummy_cache_container.cached_results = {}
                    batch_text = train_data['text']
                    optimizer.zero_grad()
                    batch_text = tensor.to_tensor(batch_text)
                    pre_label=model(batch_text)
               
               
                    batch_label = train_data['label']

                    # pre_label1 = QTensor(model(batch_text),dtype=5)
                    loss = CCEloss(batch_label,pre_label)
                    # print("backward begining!")

                    loss.backward()
                    # print("gate grad:")
                    # print(model.rnn.vqc_forget.m_para.grad)
                    # print(model.rnn.U.grad)
                    
                    optimizer._step()
                    bt = time.time()
                    tr_loss += loss.item()
                    # print("loss:{}".format(loss.item()))

                    mse = get_mse(pre_label, batch_label)
                    epoch_mse += mse
                    tfinish = time.time()
                    step_model_saving_Path = os.path.join(taskname, "models","_"+str(epoch)+"_epoch"+str(step)+"_step"+".model")
                    step_optim_saving_Path = os.path.join(taskname, "optims","_"+str(epoch)+"_epoch"+str(step)+"_step"+".optim")
                    pyvqnet.utils.storage.save_parameters(model.state_dict(),step_model_saving_Path)
                    pyvqnet.utils.storage.save_optim(optimizer.state_dict(), step_optim_saving_Path)



                tr_mse = epoch_mse / train_len
                tr_mses.append(tr_mse)
                tr_loss = tr_loss / (step+1)
                tr_losses.append(tr_loss)
                print('epoch:{}  train mse:{}'.format(epoch, tr_mse))
                print('epoch:{}  train loss:{}'.format(epoch, tr_loss))
                
            #============================================================================================

                model.eval()
                val_loss = 0.0
                epoch_corrects = 0
                val_len_loc = 0
                for step, val_data in enumerate(val_loader):
                    batch_text = val_data['text']
                    batch_label = val_data['label']
                    pre_label = model(batch_text)
                    loss = CCEloss(batch_label, pre_label)
                    val_loss += loss.item()
                    acc_mse = get_mse(pre_label, batch_label)
                    epoch_corrects += acc_mse
                val_loss = val_loss / (step+1)

                if(val_loss < best_loss*3):
                    saving_pridict_flag = True
                    best_loss = val_loss*1.1
                    model_saving_Path = os.path.join(taskname, "models",str(val_loss)+"_"+str(epoch)+"epoch.model")
                    optim_saving_Path = os.path.join(taskname, "optims",str(val_loss)+"_"+str(epoch)+"_epoch.optim")
                    pyvqnet.utils.storage.save_parameters(model.state_dict(),model_saving_Path)
                    pyvqnet.utils.storage.save_optim(optimizer.state_dict(), optim_saving_Path)

                val_losses.append(val_loss)
                print('epoch:{}  val loss:{}'.format(epoch, val_loss))
                with io.open(os.path.join(taskname, train_info_file_name), 'w', encoding='utf8') as file:
                    file.write(str(epoch)+",")
                    file.write(str(loss.item())+",")
                    file.write(str(mse)+",")
                    file.write(str(val_loss)+"\n")
            break

        except (Exception,KeyboardInterrupt) as e:
 
         
            pyvqnet.utils.storage.save_optim(optimizer.state_dict(), optimsavingpath)
            pyvqnet.utils.storage.save_parameters(model.state_dict(), modelsavingpath)

            with open('program_state.pkl', 'wb') as file:
                pickle.dump({'epoch':epoch, 'step':step}, file)
                print()
            print(f"Save successful! An error occurred: \n{e}")
    #==============================================================================================
        

    val_loss_Path = os.path.join(taskname, "val_loss.npy")
    train_mse_Path = os.path.join(taskname,"train_mse.npy")
    train_loss_Path = os.path.join(taskname,"train_loss.npy")
    
    np.save(val_loss_Path,val_losses)
    np.save(train_mse_Path,tr_mses)
    np.save(train_loss_Path,tr_losses)


    x0 = np.arange(len(val_losses))
    plt.figure()
    plt.plot(x0, val_losses)
    plt.title('val_set loss')
    pic_saving_name = os.path.join(taskname, "pics","val_loss.png")
    plt.savefig(pic_saving_name)

    x1 = np.arange(len(tr_losses))
    plt.figure()
    plt.plot(x1, tr_losses)
    plt.title('train_set mse')
    pic_saving_name = os.path.join(taskname, "pics","train_mse.png")
    plt.savefig(pic_saving_name)

def val_model(taskname,rho = 28.0,train_set = 600):
        
    if(type(rho)!=float) or (type(train_set)!=int):
        raise TimeoutError("rho must be float and train_set must be int!")
    if(type(taskname)!=str):
        taskname = str(taskname)
    

    state0 = [1.0, 1.0, 1.0]
    sigma = 10.0

    beta = 8.0 / 3.0
    t = np.arange(0.0, 80.0, 0.04)
    states = odeint(lorenz, state0, t, args=(sigma, rho, beta))

    N = 20
    input20 =[]
    lable1 = []
    for i in range(len(states)-N):
        temp=[]
        for j in range(N):
            temp.append(states[i+j])

        input20.append(temp)
        lable1.append(states[i+N])

    train_dataset = MyData(input20[:train_set],lable1[:train_set])
    train_len = len(train_dataset)
    all_dataset = MyData(input20[:4000],lable1[:4000])
    all_len = len(all_dataset)
    val_dataset = MyData(input20[train_set:train_set+400],lable1[train_set:train_set+400])
    val_len = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, drop_last=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=True,collate_fn=collate_fn)
    all_loader = DataLoader(all_dataset, batch_size=20, shuffle=False, drop_last=True,collate_fn=collate_fn)

    model = models.RegLSTM_hard(input_sz=3,hidden_sz=5,vqc_circuit_lite=models.vqc_circuit_lite_hard)



    CCEloss =MeanSquaredError()

    model_para = pyvqnet.utils.storage.load_parameters('rho28best.model')
    
    model.load_state_dict(model_para)  



    if not os.path.exists(taskname):
        os.mkdir(taskname)
        modeldirPath = os.path.join(taskname, "models")
        os.mkdir(modeldirPath)           
        optimdirPath = os.path.join(taskname, "optims")
        os.mkdir(optimdirPath)
        premodelPath = os.path.join(taskname, "model_history")
        os.mkdir(premodelPath)
        picsPath = os.path.join(taskname, "pics")
        os.mkdir(picsPath)
    train_info_file_name = "train_model_info.txt"
    res = []
    manualstep = 30
    for step, train_data in enumerate(train_loader):
        if(step<manualstep):
            continue
        print(step)
        saving_pridict_flag = False
        tstart = time.time()
        od.dummy_cache_container.cached_results = {}
        batch_text = train_data['text']

        batch_text = tensor.to_tensor(batch_text)
        pre_label=model(batch_text)
        batch_label = train_data['label']
        for i in range(20):
            res.append(pre_label.to_numpy()[i])
            print("pre label = {},real label = {}".format(pre_label.to_numpy()[i],batch_label.to_numpy()[i]))

        np.save(str(step)+"step_real_res.npy",np.array(res))
        # pre_label1 = QTensor(model(batch_text),dtype=5)
        loss = CCEloss(batch_label,pre_label)
        # print("backward begining!")
        # print("gate grad:")
        # print(model.rnn.vqc_forget.m_para.grad)
        # print(model.rnn.U.grad)
        bt = time.time()
        print(loss.item())
        # print("loss:{}".format(loss.item()))

        # mse = get_mse(pre_label, batch_label)

        tfinish = time.time()

    np.save("real_res.npy",np.array(res))
        
#train_model("600rho=24.8_112",rho=24.8,train_set=600)
val_model("val28_3para_real_test",rho = 28.0,train_set=1000)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_model", type=str, required=True)
#     parser.add_argument("--rho", type=float, required=True)
#     parser.add_argument("--train_set", type=int, required=True)
#     args = parser.parse_args()

#     train_model(args.train_model, args.rho, args.train_set)

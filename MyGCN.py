#from __future__ import print_function
"""Train  with PyTorch."""
"""
This code is forked and modified from 'https://github.com/kuangliu/pytorch-cifar'. Thanks to its contribution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch_geometric.data import DataListLoader
import statistics
import torchvision
import torchvision.transforms as transforms
from torch_geometric.nn import GCNConv, ChebConv,SplineConv,global_mean_pool,DataParallel
from pyts.image import RecurrencePlot
from torch_geometric.datasets import MNISTSuperpixels,Planetoid
import torch_geometric.transforms as T

import numpy as np
import argparse
#from utils.train import progress_bar
import os,sys
DataPath='/data'
sys.path.append(DataPath)
print_device_useage=False
aresume=True
return_output=False
print_to_logging=True
save_recurrence_plots=False


def ContractionLayerCoefficients(alpha,Numlayers):
    width=[]
    tmpOld=np.random.randint(1433*alpha,1433)
    for k in range(Numlayers):
        tmpNew=np.random.randint(tmpOld*alpha,tmpOld)
        width.append(tmpNew)
        tmpOld=tmpNew
    return width

def ResumeModel(model_to_save):
    # Load checkpoint.
    #print('==> Resuming from checkpoint..')
    checkpoint = torch.load(model_to_save)
    net = checkpoint['net']
    TrainConvergence = checkpoint['TrainConvergence']
    TestConvergence = checkpoint['TestConvergence']
    start_epoch = checkpoint['epoch']+1
    return net,TrainConvergence,TestConvergence,start_epoch

def logging(message):
    global  print_to_logging
    if print_to_logging:
        print(message)
    else:
        pass

def print_nvidia_useage():
    global print_device_useage
    if print_device_useage:
        os.system('echo check gpu;nvidia-smi;echo check done')
    else:
        pass
    
def save_recurrencePlots(net,save_recurrencePlots_file):
    global save_recurrence_plots
    if save_recurrence_plots:
        for name,parameters in net.named_parameters():
            if name=="fc.weight":
                hiddenState=parameters.cpu().detach().numpy()
                rp = RecurrencePlot()
                X_rp = rp.fit_transform(hiddenState)
                plt.figure(figsize=(6, 6))
                plt.imshow(X_rp[0], cmap='binary', origin='lower')
                plt.savefig(save_recurrencePlots_file,dpi=600)
            else:
                continue
    else:
        pass

    
class Net(torch.nn.Module):
    def __init__(self,dataset, width):
        super(Net,self).__init__()
        self.conv1 = GCNConv(dataset.num_features, width[0], cached=True)
        self.conv2 = GCNConv(width[0],width[1], cached=True)
        self.conv3 = GCNConv(width[1],width[2], cached=True)
        self.conv4 = GCNConv(width[2], dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x=self.conv1(x, edge_index)
        x =x*torch.sigmoid(x)
        x=self.conv2(x, edge_index)
        x =x*torch.sigmoid(x)
        x=self.conv3(x, edge_index)
        x =x*torch.sigmoid(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class SPlineNet(torch.nn.Module):
    def __init__(self,datasetroot):
        super(SPlineNet, self).__init__()
        self.conv1 = SplineConv(datasetroot.num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.lin1 = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, datasetroot.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.lin1(x))
        return F.log_softmax(self.lin2(x), dim=1)

    
# Training
def train(trainloader,net,optimizer,criterion):
    net.train()
    train_loss = []
    for data_list in trainloader:
        optimizer.zero_grad()
        output = net(data_list)
        y = torch.cat([data.y for data in data_list]).to(output.device)
        loss = criterion(output, y)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
    return train_loss

def test(testloader,net,criterion):
    net.eval()
    test_loss = []
    for data_list in testloader:
        output = net(data_list)
        y = torch.cat([data.y for data in data_list]).to(output.device)
        loss = criterion(output, y)
        test_loss.append(loss.item())
    return test_loss


def GCN(dataset,params,Epochs,MonteSize,width,lr,savepath):
    model_name="GCN3"
    Batch_size=int(params[0])
        
    for Monte_iter in range(MonteSize):

        # Data
        best_loss = float('inf')  # best test loss
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch         
        TrainConvergence=[]
        TestConvergence=[]

        # model 
        model_to_save='./checkpoint/{}-{}-param_{}_{}-Mon_{}-ckpt.pth'.format(dataset,model_name,params[0],params[1],Monte_iter)
        if dataset=='Cora':
            datasetroot = Planetoid(root='/data/GraphData/'+dataset, name=dataset).shuffle()
            trainloader = DataListLoader(datasetroot, batch_size=Batch_size, shuffle=True)
            testloader = DataListLoader(datasetroot, batch_size=100, shuffle=False)
            if resume and os.path.exists(model_to_save):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=Epochs-1:
                    continue
                
            
            else:
                net=SPlineNet(datasetroot)  
                
                
        if dataset=='MNIST':
            datasetroot = MNISTSuperpixels(root='/data/GraphData/'+dataset, transform=T.Cartesian()).shuffle()
            trainloader = DataListLoader(datasetroot, batch_size=Batch_size, shuffle=True)
            testloader = DataListLoader(datasetroot, batch_size=100, shuffle=False)
            if resume and os.path.exists(model_to_save):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=Epochs-1:
                    continue
                
            
            else:
                net=SPlineNet(datasetroot)          

        elif dataset=='CIFAR10':
            if resume and os.path.exists(model_to_save):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=Epochs-1:
                    continue
            else:
                net=getattr(CIFAR10_resnet,'Resnet20_CIFAR10')(params[1])
        
        
        
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        torch.cuda.is_available()  
        net = DataParallel(net)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
            
            #cudnn.benchmark = True
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        for epoch in range(start_epoch, start_epoch+Epochs):
            if epoch<Epochs:
                logging('Batch size: {},ConCoeff: {},MonteSize:{},epoch:{}'.format(params[0],params[1],Monte_iter,epoch))
                TrainLoss=train(trainloader,net,optimizer,criterion)
                TrainConvergence.append(statistics.mean(TrainLoss))
                TestConvergence.append(statistics.mean(test(testloader,net,criterion)))
            else:
                break
            if TestConvergence[epoch] < best_loss:
                logging('Saving..')
                state = {
                        'net': net.module,
                        'TrainConvergence': TrainConvergence,
                        'TestConvergence': TestConvergence,
                        'epoch': epoch,
                    }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, model_to_save)
                best_loss = TestConvergence[epoch]
                if not os.path.exists('./%s' %model_name):
                    os.makedirs('./%s' %model_name)
                torch.save(net.module.state_dict(), './%s/%s_%s_%s_pretrain.pth' %(model_name, dataset, model_name,Epochs))
            else:
                pass
            ## save recurrence plots
            if epoch%20==0:
                save_recurrencePlots_file="Results/RecurrencePlots/RecurrencePlots_{}_{}_BatchSize{}_ConCoeffi{}_epoch{}.png".format(dataset,
                                                                                                                                     model_name,params[0],params[1],epoch)
                                   
                save_recurrencePlots(net,save_recurrencePlots_file)
          
    
        FileName="{}-{}-param_{}_{}-monte_{}".format(dataset,model_name,params[0],params[1],Monte_iter)
        np.save(savepath+'TrainConvergence-'+FileName,TrainConvergence)
        np.save(savepath+'TestConvergence-'+FileName,TestConvergence)
        torch.cuda.empty_cache()
        print_nvidia_useage()


    if return_output==True:
        return TestConvergence[-1], net.module.fc.weight
    else:
        pass

if __name__=="__main__":   
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset',default='MNIST',type=str, help='dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') 
    parser.add_argument('--ConCoeff', default=0.1, type=float, help='contraction coefficients')
    parser.add_argument('--Epochs', default=3, type=int, help='Epochs')
    parser.add_argument('--MonteSize', default=1, type=int, help=' Monte Carlos size')
    parser.add_argument('--gpus', default="0", type=str, help="gpu devices")
    parser.add_argument('--BatchSize', default=512, type=int, help='Epochs')
    parser.add_argument('--savepath', type=str, default='Results/', help='Path to save results')
    parser.add_argument('--return_output', type=str, default=False, help='Whether output')
    parser.add_argument('--resume', '-r', type=str,default=True, help='resume from checkpoint')
    parser.add_argument('--print_device_useage', type=str, default=False, help='Whether print gpu useage')
    parser.add_argument('--print_to_logging', type=str, default=True, help='Whether print')
    parser.add_argument('--save_recurrencePlots', type=str, default=False, help='Whether print')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print_to_logging=args.print_to_logging
    print_device_useage=args.print_device_useage
    resume=args.resume
    return_output=args.return_output
    width=ContractionLayerCoefficients(args.ConCoeff,3)
    params=[args.BatchSize,args.ConCoeff]
    
    GCN(args.dataset,params,args.Epochs,args.MonteSize,width,args.lr,args.savepath)

    


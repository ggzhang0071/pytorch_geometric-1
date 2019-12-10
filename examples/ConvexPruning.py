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
from torch_geometric.data import DataListLoader,DataLoader
import statistics
import torchvision
import torchvision.transforms as transforms
from torch_geometric.nn import GCNConv, ChebConv,SplineConv,global_mean_pool,GraphConv, TopKPooling,DataParallel
from pyts.image import RecurrencePlot
from torch_geometric.datasets import MNISTSuperpixels,Planetoid,TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
import argparse
import os,sys


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
    NewNetworksize=checkpoint['NewNetworksize']
    start_epoch = checkpoint['epoch']
    return net,NewNetworksize,TrainConvergence,TestConvergence,start_epoch

def FindCutoffPoint(DiagValues,coefficient):
    for i in range(DiagValues.shape[0]-1):
        if DiagValues[i]>DiagValues[i+1]*coefficient:
            CutoffPoint=i+1

    try:
        return CutoffPoint
    except:
        return DiagValues.shape[0] 
    
"""def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('GCNConv') != -1:
        torch.nn.init.orthogonal_(m.weight.data)
        #torch.nn.init.xavier()
        m.bias.data.fill_(0.01)
        
def weight_reset(m):
    if isinstance(m, GCNConv) :
        m.reset_parameters()        
        
def logging(message):
    global  print_to_logging
    if print_to_logging:
        print(message)
    else:
        pass

def print_nvidia_useage():
    global print_device_useage
    if print_device_useage:
        os.system('\n echo check gpu;nvidia-smi;echo check done \r')
    else:
        pass
    
def save_recurrencePlots(net,save_recurrencePlots_file):
    global save_recurrence_plots
    if save_recurrence_plots:
        for name,parameters in net.named_parameters():
            if "fc" in name and parameters.cpu().detach().numpy().ndim==2:
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
    def __init__(self,datasetroot,width):
        self.NumLayers=len(width)
        super(Net,self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(datasetroot.num_features, width[0], cached=True))
        for i in range(self.NumLayers-1):
            layer=GCNConv(width[i],width[i+1], cached=True)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)

        self.layers.append(GCNConv(width[-1], datasetroot.num_classes, cached=True))
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x=layer(x, edge_index)
            x =x*torch.sigmoid(x)
        #x = F.dropout(x, training=self.training)
        x = F.log_softmax(self.layers[-1](x,edge_index),dim=1)
        return x
    
class SPlineNet(torch.nn.Module):
    def __init__(self,datasetroot, width):
        super(SPlineNet,self).__init__()
        self.conv1 = SplineConv(datasetroot.num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.lin1 = torch.nn.Linear(64, width[0])
        self.lin2 = torch.nn.Linear(width[0],width[1])
        self.lin3 = torch.nn.Linear(width[1], datasetroot.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)
        x = x*F.sigmoid(self.lin1(x))
        x = x*F.sigmoid(self.lin2(x))
        
        return F.log_softmax(self.lin3(x), dim=1)
    
    
    
class topk_pool_Net(torch.nn.Module):
    def __init__(self,datasetroot, width):
        self.NumLayers=len(width)
        super(topk_pool_Net, self).__init__()
        self.conv1 = GraphConv(datasetroot.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(256,width[0]))
        for i in range(self.NumLayers-1):
            self.layers.append(torch.nn.Linear(width[i],width[i+1]))   
        self.layers.append(torch.nn.Linear(width[-1], datasetroot.num_classes))
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        for layer in self.layers[:-1]:
            x=layer(x)
            x =x*torch.sigmoid(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.layers[-1](x),dim=1)
        return x

    
# Training

def RetainNetworkSize(net,ConCoeff):
    NewNetworksize=[]
    for layer_name, Weight in net.named_parameters():
        if ("weight" in layer_name) and ("layers" in layer_name):
            print(layer_name)
            [U,D,V]=torch.svd(Weight)
            CutoffPoint=FindCutoffPoint(D,ConCoeff)
            NewNetworksize.append(CutoffPoint)
            """NewWeight= torch.mm(Weight,V[:,:CutoffPoint])
            net.conv2.weight = torch.nn.Parameter( NewWeight)"""
            #print("Original size is {},After SVD is {}".format(Weight.shape[1],CutoffPoint))
    return NewNetworksize


def train(trainloader,net,optimizer,criterion):
    net.train()
    train_loss = []
    for data_list in trainloader:
        optimizer.zero_grad()
        output = net(data_list)
        target= torch.cat([data.y for data in data_list]).to(output.device)
        loss = criterion(output, target)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        #print('\nTrain set: Average tain loss: {:.4f} \n'.format(train_loss[-1]))

        """for layer_name, parameters in net.named_parameters():
             if "weight" in layer_name:
                net.conv.weight.div_(torch.norm(net.conv.weight, dim=2, keepdim=True)"""
        #test(testloader,net,criterion)
        
    return train_loss

       
def test(testloader,net,criterion):
    net.eval()
    test_loss = []
    with torch.no_grad():
        for data_list in testloader:
            output = net(data_list)
            y = torch.cat([data.y for data in data_list]).to(output.device)
            loss = criterion(output, y)
            test_loss.append(loss.item())
    #print('\n Test set: Average loss: {:.4f} \n'.format(test_loss[-1]))

    return test_loss


def GCN(args,dataset,params,num_pre_epochs,num_epochs,MonteSize,width,lr,savepath):
    Batch_size=int(params[0]) 
    for Monte_iter in range(MonteSize):
        # Data
        best_loss = float('inf')  # best test loss
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch         
        TrainConvergence=[]
        TestConvergence=[]

        # model 
        root='/git/data/GraphData/'+dataset

        if dataset=='Cora':
            model_name="PruningGCN"
            datasetroot = Planetoid(root=root, name=dataset).shuffle()
            trainloader = DataListLoader(datasetroot, batch_size=Batch_size, shuffle=True)
            testloader = DataListLoader(datasetroot, batch_size=100, shuffle=False)
            model_to_save='./checkpoint/{}-{}-param_{}_{}_{}_{}-ckpt.pth'.format(dataset,model_name,params[0],params[1],params[2],params[3])
            if Monte_iter==0:
                if resume==True and os.path.exists(model_to_save) :
                    [net,NewNetworksize,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                    if start_epoch>=num_pre_epochs-1:
                        continue
                else:
                    net=Net(datasetroot,width)  
                    #net.apply(weight_reset)

        
        elif dataset=='ENZYMES' or dataset=='MUTAG':
            model_name="topk_pool_Net"
            datasetroot=TUDataset(root,name=dataset)
            trainloader = DataListLoader(datasetroot, batch_size=Batch_size, shuffle=True)
            testloader = DataListLoader(datasetroot, batch_size=100, shuffle=False)
            model_to_save='./checkpoint/{}-{}-param_{}_{}_{}_{}-ckpt.pth'.format(dataset,model_name,params[0],params[1],params[2],params[3])
            if resume==True and os.path.exists(model_to_save):
                [net,NewNetworksize,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=num_epochs-1:
                    continue

            else:
                net=topk_pool_Net(datasetroot,width)   
               
                
        elif dataset=='MNIST':
            datasetroot = MNISTSuperpixels(root=root, transform=T.Cartesian()).shuffle()
            trainloader = DataListLoader(datasetroot, batch_size=Batch_size, shuffle=True)
            testloader = DataListLoader(datasetroot, batch_size=100, shuffle=False)
            model_name='SPlineNet'
            model_to_save='./checkpoint/{}-{}-param_{}_{}_{}_{}-Mon_{}-ckpt.pth'.format(dataset,model_name,params[0],params[1],params[2],params[3],Monte_iter)
            if resume=="True" and os.path.exists(model_to_save):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=num_epochs-1:
                    continue

            else:
                #net=Net(datasetroot,width) 
                net=SPlineNet(datasetroot,width)

        elif dataset=='CIFAR10':
            if resume=="True" and os.path.exists(model_to_save):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=num_epochs-1:
                    continue
            else:
                net=getattr(CIFAR10_resnet,'Resnet20_CIFAR10')(params[1])
        else:
            raise Exception("The dataset is:{}, it isn't existed.".format(dataset))
        
       
        if Monte_iter==0:
            print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
            net = DataParallel(net)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = net.to(device)
                #cudnn.benchmark = True
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            logging('Batch size: {}, Number of layers:{} ConCoeff: {}, CutoffCoffi:{}, MonteSize:{}'.format(params[0], params[1],params[2],params[3],Monte_iter))
            for epoch in range(num_pre_epochs):
                PreTrainLoss=train(trainloader,net,optimizer,criterion)
                print('\nEpoch: {},  Average pre-tain loss: {:.4f} \n'.format(epoch,PreTrainLoss[0]))
            NewNetworksize=RetainNetworkSize(net,params[2])
            #del net

        #NewNetworksize=width
        if dataset=='Cora':
            OptimizedNet=Net(datasetroot,NewNetworksize[0:-1])  
        elif dataset=='ENZYMES' or dataset=='MUTAG':
            NewNetworkSizeAdjust=NewNetworksize[0:-1]
            NewNetworkSizeAdjust[0]=width[0]-1
            OptimizedNet=topk_pool_Net(datasetroot,NewNetworkSizeAdjust) 
            
        #OptimizedNet.apply(init_weights)

        OptimizedNet = DataParallel(OptimizedNet)
        OptimizedNet = OptimizedNet.to(device)
        cudnn.benchmark = True
        criterionNew = nn.CrossEntropyLoss()
        optimizerNew = optim.SGD(OptimizedNet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            
        for epoch in range(start_epoch,num_epochs):
            TrainLoss=train(trainloader,OptimizedNet,optimizerNew,criterionNew)
            print('\n Epoch: {}, Average tain loss: {:.4f} \n'.format(epoch,TrainLoss[0]))
            TrainConvergence.append(statistics.mean(TrainLoss))
            TestConvergence.append(statistics.mean(test(testloader,OptimizedNet,criterion)))

            # save model
            if TestConvergence[epoch] < best_loss:
                logging('Saving..')
                state = {
                                'net': OptimizedNet.module,
                                'TrainConvergence': TrainConvergence,
                                'TestConvergence': TestConvergence,
                                'epoch': num_epochs,
                                'NewNetworksize':NewNetworksize[0:-1],
                       }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, model_to_save)
                best_loss = TestConvergence[epoch]
        
                ## save recurrence plots
            """if epoch%20==0:
                save_recurrencePlots_file="../Results/RecurrencePlots/RecurrencePlots_{}_{}_BatchSize{}    \_ConCoeffi{}_epoch{}.png".format(dataset, model_name,params[0],params[1],epoch)

            save_recurrencePlots(net,save_recurrencePlots_file)"""
          
     
        FileName="{}-{}-param_{}_{}_{}_{}-monte_{}".format(dataset,model_name,params[0],params[1],params[2],params[3],Monte_iter)
        np.save(savepath+'TrainConvergence-'+FileName,TrainConvergence)
        #np.save(savepath+'TestConvergence-'+FileName,TestConvergence)
        #torch.cuda.empty_cache()
        print_nvidia_useage()


    if return_output==True:
        return TestConvergence[-1], net.module.fc.weight
    else:
        pass

if __name__=="__main__":   
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset',default='Cora',type=str, help='dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') 
    parser.add_argument('--ConCoeff', default=0.95, type=float, help='contraction coefficients')
    parser.add_argument('--CutoffCoeff', default=0.1, type=float, help='contraction coefficients')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R',
                        help='cardinality weight (default: 1e-2)')
  
    parser.add_argument('--num_pre_epochs', type=int, default=30, metavar='P',
                        help='number of epochs to pretrain (default: 3)')
    parser.add_argument('--num_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--MonteSize', default=1, type=int, help=' Monte Carlos size')
    parser.add_argument('--gpus', default="0", type=str, help="gpu devices")
    parser.add_argument('--BatchSize', default=512, type=int, help='batch size')
    parser.add_argument('--NumLayers', default=4, type=int, help='Number of layers')

    parser.add_argument('--savepath', type=str, default='Results/', help='Path to save results')
    parser.add_argument('--return_output', type=str, default=False, help='Whether output')
    parser.add_argument('--resume', '-r', type=str,default=True, help='resume from checkpoint')
    parser.add_argument('--print_device_useage', type=str, default=False, help='Whether print gpu useage')
    parser.add_argument('--print_to_logging', type=str, default=True, help='Whether print')
    parser.add_argument('--save_recurrence_plots', type=str, default=False, help='Whether print')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print_to_logging=args.print_to_logging
    print_device_useage=args.print_device_useage
    resume=args.resume
    return_output=args.return_output
    save_recurrence_plots=args.save_recurrence_plots
    width=ContractionLayerCoefficients(args.ConCoeff,args.NumLayers)
    #params=[args.BatchSize,args.NumLayers,args.args.ConCoeff,args.CutoffCoeff]
    params=[args.BatchSize,args.NumLayers,args.ConCoeff,args.CutoffCoeff]
    
    GCN(args,args.dataset,params,args.num_pre_epochs,args.num_epochs,args.MonteSize,width,args.lr,args.savepath)

    

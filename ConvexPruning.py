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
from torch_geometric.nn import GCNConv, ChebConv,global_mean_pool,SplineConv,GraphConv, AGNNConv,TopKPooling,DataParallel
#from spline_conv import SplineConv
from pyts.image import RecurrencePlot
from torch_geometric.datasets import MNISTSuperpixels,Planetoid,TUDataset,PPI
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
import argparse
import os,sys

def ChooseModel(model_name,datasetroot,width):
    if model_name=="GCN":  
        net=GCN(datasetroot,width) 
    elif model_name=="SplineNet":     
        net=SplineNet(datasetroot,width)
    elif model_name=="ChebConvNet":
        net=ChebConvNet(datasetroot,width)  
    elif model_name=="AGNNNet":
        net=AGNNNet(datasetroot,width)  
    
    else:
        raise Exception("model not support, Choose GCN, or SplineNet or  ChebConvNet")
    return net

def ContractionLayerCoefficients(num_features,*args):
    Numlayers,alpha=args
    width=[]
    tmpOld=np.random.randint(num_features*alpha,num_features)
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

      #net.apply(weight_reset)

class GCN(torch.nn.Module):
    def __init__(self,datasetroot,width):
        super(GCN,self).__init__()
        self.NumLayers=len(width)
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(datasetroot.num_features, width[0], cached=True))
        for i in range(self.NumLayers-1):
            layer=GCNConv(width[i],width[i+1], cached=True)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)
        self.layers.append(GCNConv(width[-1], datasetroot.num_classes, cached=True))
        
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        DiagElemnt=[]
        for layer in self.layers[:-1]:
            if len(SVDOrNot)==2:
                NumCutoff=SVDOrNot[0]
                mark=SVDOrNot[1]
                [U,D,V]=torch.svd(x)
                DiagElemnt.append(D[0:NumCutoff].detach().tolist())
                np.save('Results/DiagElement/{}-DiagElement'.format(mark),DiagElemnt)
            x=layer(x, edge_index)
            x =x*torch.sigmoid(x)
            #DiagElemntTorch=torch.cat(DiagElemnt,dim=0)
        #x = F.dropout(x, training=self.training)
        x = F.log_softmax(self.layers[-1](x,edge_index),dim=0)
        return x

class AGNNNet(torch.nn.Module):
    def __init__(self,datasetroot,width):
        super(AGNNNet, self).__init__()
        self.NumLayers=len(width)
        self.layers = nn.ModuleList()
        self.lin1=torch.nn.Linear(datasetroot.num_features, width[0])
        for i in range(self.NumLayers-1):
            self.layers.append(torch.nn.Linear(width[i],width[i+1]))   
        self.layers.append(torch.nn.Linear(width[-1], datasetroot.num_classes))
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.layers = nn.ModuleList()

    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        x = F.dropout(x, training=self.training)
        x=self.lin1(x)
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        DiagElemnt=[]
        for layer in self.layers[:-1]:
            if len(SVDOrNot)==2:
                NumCutoff=SVDOrNot[0]
                mark=SVDOrNot[1]
                [U,D,V]=torch.svd(x)
                DiagElemnt.append(D[0:NumCutoff].detach().tolist())
                np.save('Results/DiagElement/{}-DiagElement'.format(mark),DiagElemnt)
            x=layer(x)
            x =x*torch.sigmoid(x)
            x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)    

class ChebConvNet(torch.nn.Module):
    def __init__(self,datasetroot,width):
        self.NumLayers=len(width)
        super(ChebConvNet,self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(datasetroot.num_features,width[0],K=1))
        for i in range(self.NumLayers-1):
            layer=ChebConv(width[i],width[i+1],K=1)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)
        self.layers.append(ChebConv(width[-1], datasetroot.num_classes,K=1))

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        DiagElemnt=[]
        for layer in self.layers[:-1]:
            if len(SVDOrNot)==2:
                NumCutoff=SVDOrNot[0]
                mark=SVDOrNot[1]
                [U,D,V]=torch.svd(x)
                DiagElemnt.append(D[0:NumCutoff].detach().tolist())
                np.save('Results/DiagElement/{}-DiagElement'.format(mark),DiagElemnt)
            x=layer(x,edge_index)
            x =x*torch.sigmoid(x)
        #x = F.dropout(x, training=self.training)
        x = F.log_softmax(self.layers[-1](x,edge_index),dim=1)
        return x


class SplineNet(torch.nn.Module):
    def __init__(self,datasetroot, width):
        super(SplineNet, self).__init__()
        self.conv1 = SplineConv(datasetroot.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, datasetroot.num_classes, dim=1, kernel_size=2)

    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, torch.rand(edge_index.size()[0])))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, 2)
        return F.log_softmax(x, dim=1)
    
class topk_pool_Net(torch.nn.Module):
    def __init__(self,datasetroot, width):
        super(topk_pool_Net, self).__init__()
        self.NumLayers=len(width)
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
    
def ModelAndSave(dataset,modelName,train_dataset,params): 
    model_to_save='./checkpoint/{}-{}-param_{}_{}_{}_{}-ckpt.pth'.format(dataset,modelName,params[0],params[1],params[2],params[3])
    if resume==True and os.path.exists(model_to_save):
        [net,NewNetworksize,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
        if start_epoch>=num_epochs-1:
            pass

    else:
        width=ContractionLayerCoefficients(train_dataset.num_features,*params[1:3])
        net =ChooseModel(modelName,train_dataset,width)
    return net, model_to_save

    
# Training

def RetainNetworkSize(net,ConCoeff):
    NewNetworksize=[]
    for layer_name, Weight in net.named_parameters():
        if ("weight" in layer_name) and ("layers" in layer_name):
            #print(layer_name)
            if Weight.dim()==3:
                Weight=Weight[0].view(Weight.size()[1],Weight.size()[2])
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
            output= net(data_list)
            y = torch.cat([data.y for data in data_list]).to(output.device)
            loss = criterion(output, y)
            test_loss.append(loss.item())
    #print('\n Test set: Average loss: {:.4f} \n'.format(test_loss[-1]))
    return test_loss


def TrainingNet(dataset,modelName,params,num_pre_epochs,num_epochs,NumCutoff,optimizerName,MonteSize,savepath):
    Batch_size=int(params[0]) 
    for Monte_iter in range(MonteSize):
        # Data
        best_loss = float('inf')  # best test loss
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch         
        TrainConvergence=[]
        TestConvergence=[]
        NewNetworkSizeAdjust=[]

        # model 
        root='/git/data/GraphData/'+dataset

        if dataset=='Cora' or dataset =='Citeseer' or dataset =='Pubmed':
            datasetroot = Planetoid(root=root, name=dataset).shuffle()
            trainloader = DataListLoader(datasetroot, batch_size=Batch_size, shuffle=True)
            testloader = DataListLoader(datasetroot, batch_size=100, shuffle=False)
            [net,model_to_save]=ModelAndSave(dataset,modelName,datasetroot,params)

        elif dataset=='ENZYMES' or dataset=='MUTAG':
            datasetroot=TUDataset(root,name=dataset,use_node_attr=True)
            
        elif dataset =="PPI":
            train_dataset = PPI(root=root, split='train')
            test_dataset = PPI(root=root, split='test')
            trainloader = DataListLoader(train_dataset, batch_size=Batch_size, shuffle=True)
            testloader = DataListLoader(test_dataset, batch_size=100, shuffle=False)
            [net,model_to_save]=ModelAndSave(dataset,modelName,train_dataset,params)


        elif dataset=='MNIST':
            datasetroot = MNISTSuperpixels(root=root, transform=T.Cartesian()).shuffle(True)

        elif dataset=='CIFAR10':
            pass

        if Monte_iter==0:
            print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
            net = DataParallel(net)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = net.to(device)
                #cudnn.benchmark = True
            criterion = nn.CrossEntropyLoss()
            optimizer =optim.Adam(net.parameters(), lr=params[3], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

            logging('dataset:{}, Batch size: {}, Number of layers:{} ConCoeff: {}, LR:{}, MonteSize:{}'.format(dataset, params[0], params[1],params[2],params[3],Monte_iter))
            for epoch in range(num_pre_epochs):
                global SVDOrNot
                if epoch==num_pre_epochs-1:
                    mark="{}-{}-Pretrain".format(dataset,modelName)
                    SVDOrNot=[NumCutoff,mark]
                    PreTrainLoss=train(trainloader,net,optimizer,criterion)
                else:
                    SVDOrNot=[]
                    PreTrainLoss=train(trainloader,net,optimizer,criterion)

                print('\nEpoch: {},  Average pre-tain loss: {:.4f} \n'.format(epoch,PreTrainLoss[0]))
        NewNetworksize=RetainNetworkSize(net,params[2])
            #del net

        if dataset=='Cora' or dataset =='Citeseer' or dataset =='Pubmed': 
            OptimizedNet=ChooseModel(modelName,datasetroot,NewNetworksize[0:-1])
            NewNetworksize.insert(0,datasetroot.num_features)
            NewNetworkSizeAdjust.append(NewNetworksize[0:-1])
            print(NewNetworkSizeAdjust)

            
        elif dataset=='ENZYMES' or dataset=='MUTAG':
            NewNetworkSizeAdjust=NewNetworksize[0:-1]
            NewNetworkSizeAdjust[0]=width[0]-1
            OptimizedNet=topk_pool_Net(datasetroot,NewNetworkSizeAdjust) 
            
        #OptimizedNet.apply(init_weights)

        OptimizedNet = DataParallel(OptimizedNet)
        OptimizedNet = OptimizedNet.to(device)
        cudnn.benchmark = True
        criterionNew = nn.CrossEntropyLoss()
        if optimizerName =="SGD":
            optimizerNew = getattr(optim,optimizerName)(OptimizedNet.parameters(), lr=params[3], momentum=0.9, weight_decay=5e-4)
        elif optimizerName =="Adam":
            optimizerNew = getattr(optim,optimizerName)(OptimizedNet.parameters(), lr=params[3], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
        for epoch in range(start_epoch,num_epochs):
            if epoch==num_epochs-1:
                    mark="{}-{}-Train".format(dataset,modelName)
                    SVDOrNot=[NumCutoff,mark]
                    PreTrainLoss=train(trainloader,net,optimizer,criterion)
            else:
                    SVDOrNot=[]
                    TrainLoss=train(trainloader,OptimizedNet,optimizerNew,criterionNew)
                    
            print('\n Epoch: {}, Average tain loss: {:.4f} \n'.format(epoch,TrainLoss[0]))
            TrainConvergence.append(statistics.mean(TrainLoss))
            TestConvergence.append(statistics.mean(test(testloader,OptimizedNet,criterion)))

            # save model
            if TestConvergence[epoch] < best_loss:
                state = {
                                'net': OptimizedNet.module,
                                'TrainConvergence': TrainConvergence,
                                'TestConvergence': TestConvergence,
                                'epoch': num_epochs,
                                'NewNetworksize':NewNetworkSizeAdjust,
                       }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, model_to_save)
                best_loss = TestConvergence[epoch]
        
                ## save recurrence plots
            """if epoch%20==0:
                save_recurrencePlots_file="../Results/RecurrencePlots/RecurrencePlots_{}_{}_BatchSize{}    \_ConCoeffi{}_epoch{}.png".format(dataset, model_name,params[0],params[1],epoch)

            save_recurrencePlots(net,save_recurrencePlots_file)"""
          
        FileName="{}-{}-param_{}_{}_{}_{}-monte_{}".format(dataset,modelName,params[0],params[1],params[2],params[3],Monte_iter)
        np.save("{}/{}Convergence/TrainConvergence-{}".format(savepath,dataset,FileName),TrainConvergence)
        np.save("{}/{}Convergence/NewNetworkSizeAdjust-{}".format(savepath,dataset,FileName),NewNetworkSizeAdjust)

        
        #np.save(savepath+'TestConvergence-'+FileName,TestConvergence)
        #torch.cuda.empty_cache()
    #print('dataset:{} resized network size is {}, train error is:{}'.format(dataset,NewNetworksize[0:-1],TrainConvergence[-1]))
    print_nvidia_useage()


    if return_output==True:
        return TestConvergence[-1], net.module.fc.weight
    else:
        pass

if __name__=="__main__":   
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset',default='Cora',type=str, help='dataset to train')
    parser.add_argument('--modelName',default='GCN',type=str, help='model to use')
    parser.add_argument('--LR', default=0.1, type=float, help='learning rate') 
    parser.add_argument('--ConCoeff', default=0.99, type=float, help='contraction coefficients')
    parser.add_argument('--CutoffCoeff', default=0.1, type=float, help='contraction coefficients')
    parser.add_argument('--NumCutoff', default=5, type=float, help='contraction coefficients')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R',
                        help='cardinality weight (default: 1e-2)')
    parser.add_argument('--optimizer',default='SGD',type=str, help='optimizer to train')

  
    parser.add_argument('--num_pre_epochs', type=int, default=30, metavar='P',
                        help='number of epochs to pretrain (default: 3)')
    parser.add_argument('--num_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--MonteSize', default=1, type=int, help=' Monte Carlos size')
    parser.add_argument('--gpus', default="0", type=str, help="gpu devices")
    parser.add_argument('--BatchSize', default=512, type=int, help='batch size')
    parser.add_argument('--NumLayers', default=4, type=int, help='Number of layers')
    parser.add_argument('--PruningTimes', default=2, type=int, help='Pruning times')

    parser.add_argument('--savepath', type=str, default='Results/', help='Path to save results')
    parser.add_argument('--return_output', type=str, default=False, help='Whether output')
    parser.add_argument('--resume', '-r', type=str,default=False, help='resume from checkpoint')
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
    #params=[args.BatchSize,args.NumLayers,args.args.ConCoeff,args.CutoffCoeff]
    params=[args.BatchSize,args.NumLayers,args.ConCoeff,args.LR]
    
    TrainingNet(args.dataset,args.modelName,params,args.num_pre_epochs,args.num_epochs,args.NumCutoff,args.optimizer,args.MonteSize,args.savepath)

    

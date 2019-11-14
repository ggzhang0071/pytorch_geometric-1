import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.backends.cudnn as cudnn
import numpy as np
from torch_geometric.nn import GCNConv, ChebConv,DataParallel # noqa
import argparse
import os
os.chdir('../../..')

def ResumeModel(model_to_save):
    checkpoint = torch.load(model_to_save)
    model=checkpoint['model']
    train_acc_convergence = checkpoint['train_acc_convergence']
    best_val_acc_convergence = checkpoint['best_val_acc_convergence']
    test_acc__convergence=checkpoint['test_acc__convergence']
    start_epoch = checkpoint['epoch']+1
    model.load_state_dict()

    return modelOld,train_acc_convergence,best_val_acc_convergence,test_acc__convergence,start_epoch

def ContractionLayerCoefficients(alpha,Numlayers):
    width=[]
    tmpOld=np.random.randint(1433*alpha,1433)
    for k in range(Numlayers):
        tmpNew=np.random.randint(tmpOld*alpha,tmpOld)
        width.append(tmpNew)
        tmpOld=tmpNew
    return width


class Net(torch.nn.Module):
    def __init__(self,dataset, width):
        super(Net,self).__init__()
        self.conv1 = GCNConv(dataset.num_features, width[0], cached=True)
        self.conv2 = GCNConv(width[0],width[1], cached=True)
        self.conv3 = GCNConv(width[1],width[2], cached=True)
        self.conv4 = GCNConv(width[2], dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
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


def train():
    model.train()
    optimizer.zero_grad()
    loss =F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    train_loss = loss.data.item()
    #print(" Train Loss: {}".format(round(train_loss,3)))
    return train_loss


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset',default='Cora',type=str, help='dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') 
    parser.add_argument('--ConCoeff', default=0.8, type=float, help='contraction coefficients')
    parser.add_argument('--Epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--MonteSize', default=1, type=int, help=' Monte Carlos size')
    parser.add_argument('--gpus', default="0", type=str, help="gpu devices")
    parser.add_argument('--BatchSize', default=512, type=int, help='Epochs')
    parser.add_argument('--savepath', type=str, default='GNN/pytorch_geometric-1/Results/', help='Path to save results')
    parser.add_argument('--return_output', type=str, default=False, help='Whether output')
    parser.add_argument('--resume', '-r', type=str,default=True, help='resume from checkpoint')
    parser.add_argument('--print_device_useage', type=str, default=False, help='Whether print gpu useage')
    parser.add_argument('--print_to_logging', type=str, default=True, help='Whether print')
    parser.add_argument('--save_recurrencePlots', type=str, default=False, help='Whether print')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    path = os.path.join('data', args.dataset)
    dataset = Planetoid(path, args.dataset, T.NormalizeFeatures())
    data = dataset[0]
    best_val_acc = test_acc = 0
    train_acc_convergence=[]
    best_val_acc_convergence=[]
    test_acc__convergence=[]
    
    use_cuda = torch.cuda.is_available()
    width=ContractionLayerCoefficients(args.ConCoeff,3)
    model_to_save="{}Model{}-Coneffi_{}.pth".format(args.savepath,args.dataset,args.ConCoeff,args.Epochs)
    if args.dataset=='Cora':
        if args.resume and os.path.exists(model_to_save):
            print("Resume model")
            [model,train_acc_convergence,best_val_acc_convergence,test_acc__convergence,start_epoch]=ResumeModel(model_to_save)
            if start_epoch>=args.Epochs-1:
                pass
        else:
            model=Net(dataset,width)
    if use_cuda:
        model.cuda()
        model = DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 
    if use_cuda:
        data = data.cuda()
    with torch.no_grad():
        data = Variable(data)
    
    for epoch in range(1, args.Epochs):
        train_loss=train()
        train_acc, val_acc, tmp_test_acc = test()
        train_acc_convergence.append(train_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_val_acc_convergence.append(best_val_acc)
            test_acc__convergence.append(test_acc)
            
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    if not os.path.exists('%s' %args.savepath):
        os.makedirs('%s' %args.savepath)
    state = {
        'epoch': epoch,
        'model':model.module if use_cuda else model,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc_convergence': train_acc_convergence,
        'best_val_acc_convergence': best_val_acc_convergence,
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc__convergence': test_acc__convergence, }
    torch.save(state, model_to_save)


import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn import svm



class attDataset(Dataset):
    def __init__(self,path):
        super(attDataset,self).__init__()
        atts=[]
        with open(path,'r',encoding='utf-8') as f:
            f = f.readlines()
            for row in f:
                row = row.split()
                att = row[:-1]
                att = [int(i) for i in att]
                label = int(row[-1])
                atts.append([att,label])
            self.atts = atts
        

    def __getitem__(self, index):
        attribute,label = self.atts[index]
        
        # return attribute,label
        return torch.Tensor(attribute),torch.Tensor([label])

    def __len__(self):
        return len(self.atts)


path_train = r'/home/lthpc/lw/WeaklyObjectDetection/apascal/attribute_data/attribute_dataset.txt'
path_test = r'/home/lthpc/lw/WeaklyObjectDetection/apascal/attribute_data/attribute_dataset_test.txt'

train_set =  attDataset(path=path_train)
test_set  =  attDataset(path=path_test)

train_loader = DataLoader(dataset=train_set,batch_size=32,shuffle=True,num_workers=4)
test_loader  = DataLoader(dataset=test_set,batch_size=32,shuffle=False,num_workers=4)












#define model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64,20)
        )

    def forward(self, x):
        output = self.mlp(x)
        return output  

#if use cuda
if torch.cuda.is_available():
    mlp = MLP().cuda()
else:
    mlp = MLP()


#define optimizer and loss function
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)   # optimize all logistic parameters
# optimizer = torch.optim.SGD(mlp.parameters(),lr = 0.001, momentum = 0.9)
loss_func = nn.CrossEntropyLoss()     

max_epoch=200
correct = 0
total = 0
loss_total = 0
running_loss = 0
test_loss = 0
test_correct = 0
test_total = 0

for epoch in range(max_epoch):
    for step, (att, label) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader


        att = att.cuda()
        label = label.long().cuda()
        mlp.train()
        output = mlp(att)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label.squeeze()).sum().item()
                                            
        loss = loss_func(output, label.squeeze())
                   
        optimizer.zero_grad()                      
        loss.backward()                             
        optimizer.step()                            
        total += label.size(0)
        loss_total += loss.item()
          
    
    
    
    # path = '/home/lthpc/lw/WeaklyObjectDetection/pre_trained_mlp/mlp_model_{}.pth'.format(epoch)
    # torch.save(mlp, path)
    

    #Test
    # mlp = torch.load(path)
    # if torch.cuda.is_available():
    #     mlp = mlp.cuda()
    mlp.eval()
    for step, (att_test,label_test) in enumerate(test_loader):
        
        
        att_test = att_test.cuda()
        label_test = label_test.long().cuda()
        output = mlp(att_test)
        _, predicted = torch.max(output.data, 1)


        test_correct += (predicted == label_test.squeeze()).sum().item()
        loss = loss_func(output, label_test.squeeze())
        test_loss += loss.item()
        test_total += label_test.size(0)


    print('corrcet:{} | total:{} | loss_total:{} | test_corrcet:{} | test_total:{} | test_loss:{}'.format(correct,total,loss_total,test_correct,test_total,test_loss))
    train_acc = correct/total
    train_loss = loss_total/total
    test_acc = test_correct/test_total
    test_loss = test_loss/test_total
    print('Accuracy of the network on the train  {}: {} | test {}'.format(epoch,train_acc,test_acc))
    print('Loss of the network on the train  {}:{}  | test {}'.format(epoch,train_loss,test_loss))
    print('---------------------------------------------------------------------------------------')
    # with open('att_epoch_acc_loss.txt','a+',encoding='utf-8') as f:
    #     tmp = [epoch,train_acc,test_acc,train_loss,test_loss]
    #     tmp = [str(i) for i in tmp]
    #     f.write(' '.join(tmp))
    #     f.write('\n')
    

    correct = 0
    total = 0
    loss_total = 0
    test_total = 0
    test_loss = 0
    test_correct = 0
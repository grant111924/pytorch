import torch 
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import MLP 

def main():
    # Hyperparameters
    batch_sizes=100
    lr=1e-2
    epoches=2
 
    #data Transforms function
    data_tf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]) 

    #Load MNIST data
    trainDataSet = datasets.MNIST(root='./data',train=True,transform=data_tf, download=True)
    testDataSet = datasets.MNIST(root='./data',train=False,transform=data_tf)
    trainLoader = DataLoader(trainDataSet,batch_size=batch_sizes,shuffle=True)
    testLoader = DataLoader(testDataSet,batch_size=batch_sizes,shuffle=False)


    model = MLP.simpleNet(28*28,300,100,10)
    if torch.cuda.is_available():
        model=model.cuda()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=lr)
    #model train
    for epoch in range(epoches):
        ave_loss=0
        for step,(batch_img,batch_label) in enumerate(trainLoader) :
            batch_img=batch_img.view(batch_img.size(0),-1)
            if torch.cuda.is_available():
                imgs=Variable(batch_img).cuda()
                labels=Variable(batch_label).cuda()
            else:
                imgs=Variable(batch_img)
                labels=Variable(batch_label)
            out=model(imgs)
            loss=criterion(out,labels)
            ave_loss=ave_loss * 0.9 + loss.data[0] * 0.1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 100 == 0 or (step+1) == len(trainLoader):
                print ('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, step+1, ave_loss))

    #model predict
    model.eval()
    eval_loss=0
    eval_acc=0
    for data in testLoader :
        img, label =data
        img=img.view(img.size(0),-1)
        if torch.cuda.is_available():
            img=Variable(img,volatile=True).cuda()
            label=Variable(label,volatile=True).cuda()
        else:
            img=Variable(img,requires_grad=False)
            label=Variable(label,requires_grad=False)
        out=model(img)
        loss=criterion(out,label)
        eval_loss+=loss.data.item()*label.size(0)
        _,pred=torch.max(out,1)
        num_correct=(pred==label).sum()
        eval_acc+=num_correct.data.item()
    print("Test Loss:{:.6f},Acc:{:.6f}".format(eval_loss/(len(testDataSet)),eval_acc/(len(testDataSet))))
if __name__ == "__main__":
    main()
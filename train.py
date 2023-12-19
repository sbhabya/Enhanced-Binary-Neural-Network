import os
import sys
import time
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from birealnet import birealnet18
from KD_loss import DistributionLoss
import numpy as np
# initializing hyper parameters
CLASSES = 10
BATCH_SIZE = 8
NUM_EPOCHS = 10

 
def main():
    learning_rate = 0.001
    weight_decay = 0
    start_epoch = 0
    best_top1_acc = 0
    if not torch.cuda.is_available():
        sys.exit
        
    start_t = time.time()
    
    cudnn.benchmark = True
    cudnn.enabled=True

    resnet18_model_path = './checkpoint_resnet18/resnet18_model_best.pth'

    model_teacher = torch.load(resnet18_model_path)    
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False   
    model_teacher.eval()
    
    model_student = birealnet18()
    model_student = nn.DataParallel(model_student).cuda()
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_kd = DistributionLoss().cuda()
 
    all_parameters = model_student.parameters()
    weight_parameters = []
    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : weight_decay}],
            lr=learning_rate)

    checkpoint_tar = './models_ReactNet/reactnet_checkpoint.path.tr'
    if os.path.exists(checkpoint_tar):
        print(f'loading checkpoint {checkpoint_tar}')
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model_student.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f'loaded checkpoint {checkpoint_tar} epoch = {start_epoch}')

    end_epoch = start_epoch + NUM_EPOCHS
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/end_epoch-start_epoch
                                                                            ), last_epoch=-1)
    for epoch in range(start_epoch):
        scheduler.step()
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform_cifar10 = transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), antialias=True),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),normalize])
    train_dataset = datasets.CIFAR10(
            root = './data',
            train = True,
            download = True,
            transform=transform_cifar10)

    val_dataset = datasets.CIFAR10(
            root = './data',
            train = False,
            download = True,
            transform=transform_cifar10)  

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size= BATCH_SIZE, shuffle=True,
            num_workers= 2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset,batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True)

    acc_arr = np.zeros((end_epoch-start_epoch, 7))
    # train the model
    for epoch in range(start_epoch, end_epoch):
        top1_t, top5_t, loss_t = train(epoch,  train_loader, model_student, model_teacher, criterion_kd, optimizer, scheduler)
        top1_v, top5_v, loss_v = validate(epoch, val_loader, model_student, criterion)
        acc_arr[epoch, 0] = epoch
        acc_arr[epoch, 1] = loss_t
        acc_arr[epoch, 2] = loss_v
        acc_arr[epoch, 3] = top1_t
        acc_arr[epoch, 4] = top5_t
        acc_arr[epoch, 5] = top1_v
        acc_arr[epoch, 6] = top5_v
        
        if top1_v > best_top1_acc:
            best_top1_acc = top1_v
            if not os.path.isdir('./models_ReactNet'):
                os.mkdir('./models_ReactNet')
            state = {
            'state_dict': model_student.state_dict(),
            'acc': best_top1_acc,
            'epoch': epoch,
            'optimizer' : optimizer.state_dict()
            }
            torch.save(state,'./models_ReactNet/reactnet_checkpoint.path.tr')
            torch.save(model_student,'./models_ReactNet/reactnet_model.path.tr')

    training_time = (time.time() - start_t) / 3600
    header = "epoch, loss_t, loss_v, top1_t, top5_t, top1_v, top5_v"
    np.savetxt("epoch_data.csv", acc_arr, delimiter=",", header=header)
    print(f'total training time = {training_time} hours')


def train(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler):
    model_student.train()
    model_teacher.eval()
    top1_acc = []
    top5_acc = []
    loss_l = []
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()

        output_student = model_student(images)
        output_teacher = model_teacher(images)        
        loss = criterion(output_student, output_teacher)
        
        _, predicted_top1 = output_student.topk(1, 1, largest=True, sorted=True)
        _, predicted_top5 = output_student.topk(5, 1, largest=True, sorted=True)

        correct_top1 = predicted_top1.eq(target.view(-1, 1).expand_as(predicted_top1))
        correct_top5 = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))    

        top1_accuracy = 100*correct_top1.float().sum()/target.size(0)
        top5_accuracy = 100*correct_top5.float().sum()/target.size(0)
        top1_acc.append(top1_accuracy)
        top5_acc.append(top5_accuracy)
        loss_l.append(loss.item())
        if i%200 == 0:
            print(f'Epoch: {epoch}, [{i*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item()}, Top1_Acc: {top1_accuracy}, Top5_Acc: {top5_accuracy}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    top1_avg = sum(top1_acc)/len(top1_acc)
    top5_avg = sum(top5_acc)/len(top5_acc)  
    loss_avg = sum(loss_l)/len(loss_l)
    scheduler.step()
    return top1_avg, top5_avg, loss_avg




def validate(epoch, val_loader, model, criterion):
    model.eval()
    top1_list = []
    top5_list = []
    loss_l = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)
            _, predicted_top1 = output.topk(1, 1, largest=True, sorted=True)
            _, predicted_top5 = output.topk(5, 1, largest=True, sorted=True)

            correct_top1 = predicted_top1.eq(target.view(-1, 1).expand_as(predicted_top1))
            correct_top5 = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))
            
            top1_accuracy = 100*correct_top1.float().sum()/target.size(0)
            top5_accuracy = 100*correct_top5.float().sum()/target.size(0)  
            top1_list.append(top1_accuracy)
            top5_list.append(top5_accuracy)
            loss_l.append(loss.item())
            
    top1_avg = sum(top1_list)/len(top1_list)
    top5_avg = sum(top5_list)/len(top5_list)
    loss_avg = sum(loss_l)/len(loss_l)
    print(f'Epoch: {epoch} acc@1 {top1_avg} acc@5 {top5_avg}')

    return top1_avg, top5_avg, loss_avg


if __name__ == '__main__':
    main()
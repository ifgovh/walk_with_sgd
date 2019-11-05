__author__ = 'Chen Xing, Devansh Arpit'
import numpy as np
from models import ResNet56, vgg11, MLPNet
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl
import torchvision
import torchvision.transforms as transforms
import argparse
import torchvision.datasets as datasets

from dataloader import get_data_loaders
import resnet_models

def test(epoch, model, loader):
    model.train()
    test_loss = 0
    correct = 0
    total = 0
    optimizer=torch.optim.SGD(model.parameters(),lr=0)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    test_loss/=batch_idx
    # Save checkpoint.
    acc = 100. * correct / total
    print("Loss of "+str(epoch)+': '+str(test_loss) +"  Accuracy: "+str(acc))
    return acc, test_loss


def iteratively_interpolate_model(dir,save_dir):
    torch.nn.Module.dump_patches = True
    with open(dir + "epoch_" + args.epoch_index + '.batch_0.pt', 'rb') as f:
    #with open(dir + 'init_model.pt', 'rb') as f:
        checkpoint = torch.load(f)
        model_initial = checkpoint['net']
    model1=model_initial
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, dist_iteration_list=[],[],[],[],[]
    e=0
    for j in range(1,args.num_batches):
        with open(dir + "epoch_" + args.epoch_index + ".batch_" + str(j) + '.pt', 'rb') as f:
            checkpoint = torch.load(f)
        model2 = checkpoint['net']
        e = interpolate_between_2models(model1,model2,train_loss_list,train_acc_list, e)
        model1 = model2
        print("Iteration Number: "+str(len(train_loss_list)))
        with open(save_dir + "/train_loss.pkl", "wb") as f:
            pkl.dump(train_loss_list, f)

        with open(save_dir + "/train_acc.pkl", "wb") as f:
            pkl.dump(train_acc_list, f)

def iteratively_interpolate_model_gd(dir,save_dir):
    torch.nn.Module.dump_patches = True
    with open(dir + "epoch_1", 'rb') as f:
    #with open(dir + 'init_model.pt', 'rb') as f:
        checkpoint = torch.load(f)
        model_initial = checkpoint['net']
    model1=model_initial
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, dist_iteration_list=[],[],[],[],[]
    e=0
    for j in range(2,args.num_batches):
        with open(dir + "epoch_"  + str(j) + '.pt', 'rb') as f:
            checkpoint = torch.load(f)
        model2 = checkpoint['net']
        e = interpolate_between_2models(model1,model2,train_loss_list,train_acc_list, e)
        model1 = model2
        print("Iteration Number: "+str(len(train_loss_list)))
        with open(save_dir + "/train_loss.pkl", "wb") as f:
            pkl.dump(train_loss_list, f)

        with open(save_dir + "/train_acc.pkl", "wb") as f:
            pkl.dump(train_acc_list, f)

def interpolate_between_2models(model1,model2,train_loss_list,train_acc_list,epoch):
    model_to_call = getattr(resnet_models, args.arch)
    model = model_to_call()

    model.cuda()
    model1.eval()
    model2.eval()
    model.eval()
    alpha_list = np.arange(0, 1, 0.1)
    for alpha in alpha_list:
        new_dict = {}
        p1_params = model1.state_dict()
        p2_params = model2.state_dict()
        for p in p1_params.keys():
            if p in p2_params.keys():
                new_dict[p] = (1 - alpha) * p1_params[p] + alpha * p2_params[p]
            else:
                print(p)
        model.load_state_dict(new_dict)
        train_acc, train_loss = test(epoch,model, trainloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        epoch+=1
    return epoch

###############################################################################
# Main function
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments of "A Walk with SGD"')

    # Directories
    parser.add_argument('--data', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--model_dir', type=str, default='./resnet56128/',
                        help='')
    parser.add_argument('--save_dir', type=str, default='./resnet56128/interpo',
                        help='')
    # Hyperparams
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--mbs', type=int, default=128, metavar='N',
                        help='minibatch size')
    # Meta arguments: Tracking, resumability, CUDA
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name (cifar10)')

    parser.add_argument('--arch', type=str, default='ResNet56',
                        help='arch name (resnet, vgg11)')

    parser.add_argument('--epoch_index', type=str, default='1',
                        help='resume experiment ')
    parser.add_argument('--num_batches', type=int, default=391,
                        help='number of batches per epoch')
    parser.add_argument('--mode', type=str, default='sgd',
                        help='mode name (sgd, gd)')




    args = parser.parse_args()

    criterion = nn.CrossEntropyLoss()
    criterion_nll = nn.NLLLoss()


    print('==> Preparing data..')

    trainloader, testloader, _ = get_data_loaders(args)


    save_dir=args.model_dir+'/interpolation_'+str(args.epoch_index)+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.mode=='sgd':
        iteratively_interpolate_model(args.model_dir, args.save_dir)
    else:
        iteratively_interpolate_model_gd(args.model_dir, args.save_dir)



__author__ = 'Chen Xing, Devansh Arpit'
import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl
from models import vgg11, MLPNet
import resnet_models 

import torchvision
import torchvision.transforms as transforms

from dataloader import get_data_loaders


###############################################################################
# Training code
###############################################################################
def compute_dist():
    global model
    global param_dist_list
    with open(save_dir + '/init_model.pt', 'rb') as f:
        checkpoint = torch.load(f)
    model_init = checkpoint['net']
    d = 0.
    for param1, param2 in zip(model_init.parameters(), model.parameters()):
        param1 = param1.data.cpu().numpy()
        param2 = param2.data.cpu().numpy()
        d += np.sum((param1 - param2) ** 2)
    param_dist_list.append(np.sqrt(d))

    with open(save_dir + "/param_dist_list.pkl", "wb") as f:
        pkl.dump(param_dist_list, f)

def test(epoch, loader, valid=False):
    global best_acc
    global model
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        

    # Save checkpoint.
    acc = 100. * correct / total
    if valid and acc > best_acc:
        print('Saving best model..')
        state = {
            'net': model,
            'acc': acc,
            'epoch': epoch,
        }
        with open(save_dir + '/best_model.pt', 'wb') as f:
            torch.save(state, f)
        best_acc = acc
    return acc


def compute_angle_et_norm():
    global optimizer
    global model
    global cos_mg_list
    global grad_norm_list
    if not hasattr(compute_angle_et_norm, 'm'):
        compute_angle_et_norm.m = {}

    dot = 0
    norm1 = 0
    norm2 = 0
    for name, variable in model.named_parameters():

        g = variable.grad.data
        if not name in compute_angle_et_norm.m.keys():
            compute_angle_et_norm.m[name] = g.clone()
            dot += torch.sum(compute_angle_et_norm.m[name] * g.clone())
            norm1 += torch.sum(compute_angle_et_norm.m[name] * compute_angle_et_norm.m[name])
            norm2 += torch.sum(g.clone() * g.clone())
        else:
            dot += torch.sum(compute_angle_et_norm.m[name] * g.clone())
            norm1 += torch.sum(compute_angle_et_norm.m[name] * compute_angle_et_norm.m[name])
            norm2 += torch.sum(g.clone() * g.clone())
            compute_angle_et_norm.m[name] = g.clone()

    cos = dot / (np.sqrt(norm1 * norm2) + 0.000001)
    cos_mg_list.append(cos)
    with open(save_dir + "/cos_mg_list.pkl", "wb") as f:
        pkl.dump(cos_mg_list, f)

    grad_norm_list.append(norm2)
    with open(save_dir + "/grad_norm_list.pkl", "wb") as f:
        pkl.dump(grad_norm_list, f)

    return cos

def train(epoch):
    global trainloader
    global optimizer
    global args
    global model
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_list = []

    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()
        compute_angle_et_norm()
        compute_dist()
        optimizer.zero_grad()
        if (args.save_model_per_iter is True)&((epoch==1)|(epoch==10)|(epoch==25)|(epoch==100)):
            print('Saving intermediate model..')
            state = {
                'net': model,
                'iter': batch_idx,
            }
            with open(save_dir + '/epoch_{}.batch_{}.pt'.format(epoch, batch_idx), 'wb') as f:
                torch.save(state, f)        
        
        loss_list.append(loss.data.item())
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    print('Saving model..')
    state = {
        'net': model,
        'iter': epoch,
    }
    with open(save_dir + '/epoch_{}.pt'.format(epoch), 'wb') as f:
        torch.save(state, f)

    return sum(loss_list) / float(len(loss_list)), 100. * correct / total


###############################################################################
# Main function
###############################################################################
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Experiments of "A Walk with SGD"')
    # Directories
    parser.add_argument('--data', type=str, default='./data/',
                        help='location of the data corpus')
    # Hyperparams
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate factor that gets multiplied to the hard coded LR schedule in the code')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper epoch limit')
    parser.add_argument('--init', type=str, default="he")
    parser.add_argument('--wdecay', type=float, default=0,
                        help='weight decay applied to all weights')#1e-4

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='minibatch size')
    # Meta arguments: Tracking, resumability, CUDA
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name (cifar10, mnist)')
    parser.add_argument('--arch', type=str, default='ResNet56',
                        help='arch name (resnet, vgg11, mlp)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume experiment ')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--cluster', action='store_true', help='do not show the progress bar for batch job')

    parser.add_argument('--save_model_per_iter', type=bool, default=True,
                        help='')
    args = parser.parse_args()


    use_cuda = torch.cuda.is_available()
    ### Set the random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():    
        torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    print('==> Preparing data..')

    trainloader, testloader, _ = get_data_loaders(args)

    ###############################################################################
    # Build the model
    ###############################################################################
    save_dir = './' + args.arch + str(args.batch_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(save_dir), 'Error: no checkpoint directory found!'
        with open(save_dir + '/best_model.pt', 'rb') as f:
            checkpoint = torch.load(f)
        model = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

        lr_list = pkl.load(open(save_dir + "/LR_list.pkl", "r"))

        train_loss_list = pkl.load(open(save_dir + "/train_loss.pkl", "r"))

        train_acc_list = pkl.load(open(save_dir + "/train_acc.pkl", "r"))

        valid_acc_list = pkl.load(open(save_dir + "/valid_acc.pkl", "r"))

        param_dist_list = pkl.load(open(save_dir + "/param_dist_list.pkl", "r"))

    else:
        if os.path.isdir(save_dir):
            with open(save_dir + '/log.txt', 'w') as f:
                f.write('')
        print('==> Building model..')

        start_epoch = 1

        model_to_call = getattr(resnet_models, args.arch)
        model = model_to_call()
        
        # nb = 0
        # if args.init == 'he':
        #     for m in model.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nb += 1
        #             print('Update init of ', m)
        #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             m.weight.data.normal_(0, math.sqrt(2. / n))
        #         elif isinstance(m, nn.BatchNorm2d):
        #             print('Update init of ', m)
        #             m.weight.data.fill_(1)
        #             m.bias.data.zero_()

        best_acc = 0
        lr_list = []

        train_loss_list = []
        train_acc_list = []
        valid_acc_list = []
        param_dist_list = []
        cos_mg_list = []
        grad_norm_list = []

    if use_cuda:
        model.cuda()
    total_params = sum(np.prod(x.size()) if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
    print('Args:', args)
    with open(save_dir + '/log.txt', 'a') as f:
        f.write(str(args) + ',total_params=' + str(total_params) + '\n')

    criterion = nn.CrossEntropyLoss()
    # criterion_nll = nn.NLLLoss()

    print('Saving initial model..')
    state = {
        'net': model,
        'acc': -1,
        'epoch': 0,
    }
    with open(save_dir + '/init_model.pt', 'wb') as f:
        torch.save(state, f)

    # Loop over epochs.

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                                momentum=0, \
                                                weight_decay=args.wdecay, nesterov=False)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        loss, train_acc = train(epoch)
        print("current lr:"+str(optimizer.param_groups[0]['lr']))
        train_loss_list.append(loss)
        train_acc_list.append(train_acc)
        valid_acc = test(epoch, testloader, valid=True)
        valid_acc_list.append(valid_acc)
        with open(save_dir + "/train_loss.pkl", "wb") as f:
            pkl.dump(train_loss_list, f)
        with open(save_dir + "/train_acc.pkl", "wb") as f:
            pkl.dump(train_acc_list, f)
        with open(save_dir + "/valid_acc.pkl", "wb") as f:
            pkl.dump(valid_acc_list, f)
        lr_list.append(optimizer.param_groups[0]['lr'])
        epoch_fac = 1.
        status = 'Epoch {}/{} | Loss {:3f} | Acc {:3f} | val-acc {:.3f}| max-variance {:.3f}| LR {:4f} | BS {}'. \
            format(epoch, args.epochs * epoch_fac, loss, train_acc, valid_acc, 0, lr_list[-1], args.batch_size)
        with open(save_dir + '/log.txt', 'a') as f:
            f.write(status + '\n')

        with open(save_dir + "/LR_list.pkl", "wb") as f:
            pkl.dump(lr_list, f)
        print('-' * 89)

    # Load the best saved model.
    with open(save_dir + '/best_model.pt', 'rb') as f:
        best_state = torch.load(f)
    model = best_state['net']
    # Run on test data.
    test_acc = test(epoch, testloader, valid=True)
    print('=' * 89)
    print('| End of training | test acc {}'.format(test_acc))
    print('=' * 89)

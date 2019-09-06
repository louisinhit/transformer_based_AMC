import torch
import torch.optim as optim
import torch.nn.functional as F
import sys, os
sys.path.append("../../")
from utiles import gen_test_set, gen_train_set, gen_list
from import_m import RT
import numpy as np
import argparse
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--dropout', type=float, default=0.01)                  #####
parser.add_argument('--clip', type=float, default=1,                        #####
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=60) 
parser.add_argument('--ksize', type=int, default=8)                        #####
parser.add_argument('--n_level', type=int, default=2)                       #####
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--lr', type=float, default=0.0001)                     #####
parser.add_argument('--optim', type=str, default='Adam')                    #####
parser.add_argument('--rnn_type', type=str, default='GRU')                  #####
parser.add_argument('--d_model', type=int, default=64)                      #####
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--h', type=int, default=2)
parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--permute', action='store_true', default=False)

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda")

base_path = os.path.dirname(os.path.realpath(__file__))
#########################
name = "GOLD_XYZ_OSC.0001_1024.hdf5"        
#########################
s_dir = os.path.join(base_path,'output/')

n_classes = 24
input_channels = 2
seq_length = 1024
epochs = args.epochs
steps = 0
print(args)
                                
###################

def output_s(message, save_filename):
    print (message)
    with open(save_filename, 'a') as out:
        out.write(message + '\n')

def train(ep):
    global steps
    train_loss = 0
    model.train()
    
    trainset = gen_train_set(name, train_idx)
    train_loader = utils.DataLoader(trainset, batch_size=32)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: 
            data, target = data.cuda(), target.cuda()
            #data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]

        optimizer.zero_grad()
        output = model(data)
        target = target.squeeze_()
        loss = criterion(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        print (train_loss.item())
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            message = ('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                train_loss.item()/args.log_interval, steps))
            output_s(message, message_filename)
            train_loss = 0

def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    testset = gen_test_set(name, test_idx)
    test_loader = utils.DataLoader(testset, batch_size=32)
    
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            
            output = model(data)
            target = target.squeeze_()
            test_loss += criterion(output, target).item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        test_loss /= len(test_loader)
        message = ('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, total,
            100. * correct / total))
        output_s(message, message_filename)
        return test_loss, correct / total


if __name__ == "__main__":

    model_name = "d_{}_h_{}_t_{}_ksize_{}_level_{}_n_{}_lr_{}_dropout_{}".format(
                args.d_model, args.h, args.rnn_type, args.ksize, 
                args.n_level, args.n, args.lr, args.dropout)

    message_filename = s_dir + 'r_' + model_name + '.txt'

    with open(message_filename, 'w') as out:
        out.write('start\n')
    
    test_idx, train_idx = gen_list()

    model = RT(input_channels, args.d_model, n_classes, h=args.h, rnn_type=args.rnn_type, ksize=args.ksize, 
            n_level=args.n_level, n=args.n, dropout=args.dropout, emb_dropout=args.dropout)

    model.to(device)
    ###################
    batch_size = 32                       
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() 
    for epoch in range(1, epochs+1):
        train(epoch)
        test()

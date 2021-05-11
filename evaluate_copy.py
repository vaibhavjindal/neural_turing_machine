"""Evaluates the trained model on sequences of length 10, 20, 30, 50 and 120, 
as done in the paper and saves the output as figures"""
import argparse
import argcomplete
from wrapper_ntm import NTM
from dataloader import dataloader_copy


import torch
from torch import nn
import numpy as np

from IPython.display import Image as IPythonImage
from PIL import Image
import matplotlib.pyplot as plt


def evaluate(net, criterion, X, Y):
    """Evaluate a single batch (without training).
    """
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.initialize(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.item(),
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result

def init_arguments():
    parser = argparse.ArgumentParser(prog='evaluate.py')
    parser.add_argument('--checkpoint_path', action='store', default="./checkpoints/copy-task-1000-batch-50000.model",
                        help="Path for retrieving checkpoint data")

    parser.add_argument('--task_name',default="copy-task",type=str,help = "should be same as the one given while training")
    parser.add_argument('--controller_size',default=100, type=int,help = "should be same as the one given while training")
    parser.add_argument('--controller_layers',default=1,type=int,help = "should be same as the one given while training")
    parser.add_argument('--sequence_width',default=8, type=int,help = "should be same as the one given while training")
    parser.add_argument('--sequence_min_len',default=1,type=int,help = "should be same as the one given while training")
    parser.add_argument('--sequence_max_len',default=20, type=int,help = "should be same as the one given while training")
    parser.add_argument('--memory_n',default=128, type=int,help = "should be same as the one given while training")
    parser.add_argument('--memory_m',default=20, type=int,help = "should be same as the one given while training")
    parser.add_argument('--num_batches',default=4000, type=int,help = "should be same as the one given while training")
    parser.add_argument('--batch_size',default=1, type=int,help = "should be same as the one given while training")
    parser.add_argument('--rmsprop_lr',default=1e-4, type=float,help = "should be same as the one given while training")
    parser.add_argument('--rmsprop_momentum',default=0.9, type=float,help = "should be same as the one given while training")
    parser.add_argument('--rmsprop_alpha',default=0.95, type=float,help = "should be same as the one given while training")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

def main():
    # Initialize arguments
    args = init_arguments()

    #initialising the net
    net = NTM(args.sequence_width + 1, args.sequence_width,
                              args.controller_size, args.controller_layers,
                              args.memory_n, args.memory_m)

    #loading weights
    net.load_state_dict(torch.load(args.checkpoint_path))
    
    #save images for sequences of different lengths as done in paper
    seq_lengths = [10,20,30,50,120]
    for sl in seq_lengths:
        _, x, y = next(iter(dataloader_copy(1, 1, 8, sl, sl)))# random image of sequence length sl
        result = evaluate(net, nn.BCELoss(), x, y)
        y_out = result['y_out']

        y = y[:,0,:].T.detach().numpy()
        y_out = y_out[:,0,:].T.detach().numpy()
        
        pixel_size = 8 #one bit of our input will be displayed as an 8*8 square
        final_target = np.zeros((y.shape[0]*pixel_size,y.shape[1]*pixel_size)).astype("uint8")
        for i in range((y.shape[0])):
            for j in range((y.shape[1])):
                final_target[i*pixel_size:(i+1)*pixel_size,j*pixel_size:(j+1)*pixel_size] = 255*y[i,j]

        final_output = np.zeros((y.shape[0]*pixel_size,y.shape[1]*pixel_size)).astype("uint8")
        for i in range((y.shape[0])):
            for j in range((y.shape[1])):
                final_output[i*pixel_size:(i+1)*pixel_size,j*pixel_size:(j+1)*pixel_size] = 255*y_out[i,j]
            

        im1 = Image.fromarray(final_target)
        im1.save("target_len_"+str(sl)+".png")

        im2 = Image.fromarray(final_output)
        im2.save("output_len_"+str(sl)+".png")


if __name__ == '__main__':
    main()

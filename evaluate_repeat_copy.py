"""Evaluates the trained model on sequences of length 10, 20, 30, 50 and 120, 
as done in the paper and saves the output as figures"""
import argparse
import argcomplete
from wrapper_ntm import NTM
from dataloader import dataloader_repeat_copy


import torch
from torch import nn
import numpy as np

from PIL import Image


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
    parser.add_argument('--checkpoint_path', action='store', default="./checkpoints/repeat-copy-task-100-batch-100000.model",
                        help="Path for retrieving checkpoint data")
    parser.add_argument('--seq_len',default=10, type=int)
    parser.add_argument('--rep_len',default=10,type=int)

    #should be the same as during training
    parser.add_argument('--task_name',default="repeat-copy-task",type=str)
    parser.add_argument('--controller_size',default=100, type=int)
    parser.add_argument('--controller_layers',default=1,type=int)
    parser.add_argument('--sequence_width',default=8, type=int)
    parser.add_argument('--sequence_min_len',default=1,type=int)
    parser.add_argument('--sequence_max_len',default=10, type=int)
    parser.add_argument('--min_repeat',default=1,type=int)
    parser.add_argument('--max_repeat',default=10, type=int)
    parser.add_argument('--memory_n',default=128, type=int)
    parser.add_argument('--memory_m',default=20, type=int)
    parser.add_argument('--num_batches',default=100000, type=int)
    parser.add_argument('--batch_size',default=1, type=int)
    parser.add_argument('--rmsprop_lr',default=1e-4, type=float)
    parser.add_argument('--rmsprop_momentum',default=0.9, type=float)
    parser.add_argument('--rmsprop_alpha',default=0.95, type=float)


    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

def main():
    # Initialize arguments
    args = init_arguments()

    #initialising the net
    net = NTM(args.sequence_width + 2, args.sequence_width+1,
                              args.controller_size, args.controller_layers,
                              args.memory_n, args.memory_m)

    #loading weights
    net.load_state_dict(torch.load(args.checkpoint_path))
    
    #save images for sequences of different lengths as done in paper
    seq_lengths = [args.seq_len]
    repeat_lengths = [args.rep_len]
    for sl in seq_lengths:
        for rl in repeat_lengths:
            
            #constants according to the train settings
            reps_mean = (args.max_repeat + args.min_repeat) / 2
            reps_var = (((args.max_repeat - args.min_repeat + 1) ** 2) - 1) / 12
            reps_std = np.sqrt(reps_var)


            _, x, y = next(iter(dataloader_repeat_copy(1, 1, 8, sl, sl, rl, rl,reps_mean,reps_std)))
            result = evaluate(net, nn.BCELoss(), x, y)
            y_out = result['y_out']

            y = y[:,0,:].T.detach().numpy()
            y_out = y_out[:,0,:].T.detach().numpy()
            
            x = x[:,0,:].T.detach().numpy()

            pixel_size = 8 #one bit of our input will be displayed as an 8*8 square

            final_target = np.zeros((y.shape[0]*pixel_size,y.shape[1]*pixel_size)).astype("uint8")
            for i in range((y.shape[0])):
                for j in range((y.shape[1])):
                    final_target[i*pixel_size:(i+1)*pixel_size,j*pixel_size:(j+1)*pixel_size] = 255*y[i,j]

            final_output = np.zeros((y.shape[0]*pixel_size,y.shape[1]*pixel_size)).astype("uint8")
            for i in range((y.shape[0])):
                for j in range((y.shape[1])):
                    final_output[i*pixel_size:(i+1)*pixel_size,j*pixel_size:(j+1)*pixel_size] = 255*y_out[i,j]
            
            final_input = np.zeros((x.shape[0]*pixel_size,x.shape[1]*pixel_size)).astype("uint8")
            for i in range((x.shape[0])):
                for j in range((x.shape[1])):
                    final_input[i*pixel_size:(i+1)*pixel_size,j*pixel_size:(j+1)*pixel_size] = 255*x[i,j]

            im1 = Image.fromarray(final_target)
            im1.save("target_len_"+str(sl)+"_rep_"+str(rl)+".png")

            im2 = Image.fromarray(final_output)
            im2.save("output_len_"+str(sl)+"_rep_"+str(rl)+".png")

            im3 = Image.fromarray(final_input)
            im3.save("input_len_"+str(sl)+"_rep_"+str(rl)+".png")

if __name__ == '__main__':
    main()

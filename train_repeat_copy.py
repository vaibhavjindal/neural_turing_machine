"""Trainer for the Repeat Copy Task"""

import argparse
import json
import time
import random
import argcomplete
import torch
from torch import nn
from torch import optim
import numpy as np

from wrapper_ntm import NTM
from dataloader import dataloader_repeat_copy


def save_checkpoint(net, name, args, batch_num, losses, costs, seq_lengths):
    """
    Taken from https://github.com/loudinthecloud/pytorch-ntm
    """
    basename = "{}/{}-{}-batch-{}".format(args.checkpoint_path, name, args.seed, batch_num)
    model_fname = basename + ".model"
    print(f"Saving model checkpoint to: {model_fname}")
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    print(f"Saving model training history to: {train_fname}")
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


def clip_grads(net):
    """
    Gradient clipping to the range [10, 10] to prevent gradient overshoot
    taken from https://github.com/loudinthecloud/pytorch-ntm   
    """
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def train_repeat_copy_task(args):

    #set all seeds to same value
    if args.seed:
        seed = args.seed
    else:
        seed = np.random.randint(10000)
    
    print(f"Using seed={seed} for training")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    #our network
    net = NTM(args.sequence_width + 2, args.sequence_width+1,
                              args.controller_size, args.controller_layers,
                              args.memory_n, args.memory_m)

    #data_loader
    data_loader = dataloader_repeat_copy(args.num_batches, args.batch_size,
                          args.sequence_width,
                          args.sequence_min_len, args.sequence_max_len,
                          args.min_repeat, args.max_repeat)

    #rms-prop
    optimizer = optim.RMSprop(net.parameters(),
                             momentum=args.rmsprop_momentum,
                             alpha=args.rmsprop_alpha,
                             lr=args.rmsprop_lr)

    loss_criterion = nn.BCELoss()

    num_batches = args.num_batches
    batch_size = args.batch_size

    print(f"Training our model for {num_batches} batches with batch_size={batch_size} ...")

    losses = []
    costs = []
    seq_lengths = []
    start_ms = time.time()*1000

    for batch_num, inp, out in data_loader:
        ## out.shape -> [seq_len, batch_size, seq_width] (expected output from ntm)
        ## inp.shape -> [seq_len+1, batch_size, seq_width+1] (input to ntm)

        optimizer.zero_grad()
        inp_seq_len = inp.size(0)
        outp_seq_len, batch_size, _ = out.size()

        # New sequence
        net.initialize(batch_size)

        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            net(inp[i])

        # Read the output (no input given)
        y_out = torch.zeros(out.size())
        for i in range(outp_seq_len):
            y_out[i], _ = net()
 
        loss = loss_criterion(y_out, out)
        loss.backward()
        clip_grads(net)
        optimizer.step()

        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

        # The cost is the number of error bits per sequence
        cost = torch.sum(torch.abs(y_out_binarized - out.data))

        losses += [loss.item()]
        costs += [cost.item()/batch_size]
        seq_lengths += [out.size(0)]


        #Logging progress on terminal
        if((batch_num-1)%args.report_interval == 0):
            print("Training on batches "+str((batch_num-1))+"-"+str(batch_num-1+args.report_interval)+":")
        
        # Report
        if batch_num % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            mean_cost = np.array(costs[-args.report_interval:]).mean()
            mean_time = int(((time.time()*1000 - start_ms) / args.report_interval) / batch_size)
            print(f"Mean_loss: {mean_loss} Mean_Cost: {mean_cost} Mean_Time: {mean_time} millisec/seq")
            start_ms = time.time()*1000

        # Checkpoint
        if (args.checkpoint_interval != 0) and (batch_num % args.checkpoint_interval == 0):
            save_checkpoint(net, args.task_name, args,
                            batch_num, losses, costs, seq_lengths)

    print("Training complete")


def init_arguments():
    parser = argparse.ArgumentParser(prog='train_repeat_copy.py')
    parser.add_argument('--seed', type=int, default=100, help="Seed value")
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help="Checkpoint interval (default: 1000). Use 0 to disable checkpointing")
    parser.add_argument('--checkpoint-path', action='store', default='./',
                        help="Path for saving checkpoint data (default: './')")
    parser.add_argument('--report-interval', type=int, default=100,
                        help="Reporting interval")
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
    args.checkpoint_path = args.checkpoint_path.rstrip('/')
    return args


def main():
    # Initialize arguments
    args = init_arguments()

    #train the model
    train_repeat_copy_task(args)

if __name__ == '__main__':
    main()

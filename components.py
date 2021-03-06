"""NTM Read and Write Heads."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from utils import _split_cols, _convolve

class ReadHead(nn.Module):
    """
    ReadHead for reading from memory
    In this implementation, the NTM only has a single ReadHead
    """
    def __init__(self, memory, controller_size):
        super(ReadHead, self).__init__()

        self.memory = memory
        self.controller_size = controller_size
        self.head_type = "read"

        # Corresponding to k, beta, g, s, gamma sizes from the paper
        self.read_lengths = [memory.M, 1, 1, 3, 1]
        self.fc = nn.Linear(controller_size, sum(self.read_lengths))
        
        #initialisation values taken from other implementations
        nn.init.xavier_uniform_(self.fc.weight, gain=1.4)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, embeddings, w_prev):
        o = self.fc(embeddings)
        k, beta, g, s, gamma = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self.memory.address(k.clone(),
                                F.softplus(beta),
                                torch.sigmoid(g),
                                F.softmax(s, dim=1),
                                1 + F.softplus(gamma),
                                w_prev)
        r = self.memory.read(w)
        return r, w

class WriteHead(nn.Module):
    """
    WriteHead for writing to memory
    """
    def __init__(self, memory, controller_size):
        super(WriteHead, self).__init__()

        self.memory = memory
        self.controller_size = controller_size
        self.head_type = "write"

        # Corresponding to k, beta, g, s, gamma, e, a sizes from the paper
        self.write_lengths = [memory.M, 1, 1, 3, 1, memory.M, memory.M]
        self.fc= nn.Linear(controller_size, sum(self.write_lengths))

        #initialisation values taken from other implementations
        nn.init.xavier_uniform_(self.fc.weight, gain=1.4)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, embeddings, w_prev):
        o = self.fc(embeddings)
        k, beta, g, s, gamma, e, a = _split_cols(o, self.write_lengths)

        # e should be in [0, 1]
        e = torch.sigmoid(e)

        # Write to memory
        w = self.memory.address(k.clone(),
                                F.softplus(beta),
                                torch.sigmoid(g),
                                F.softmax(s, dim=1),
                                1 + F.softplus(gamma),
                                w_prev)

        self.memory.write(w, e, a)
        return w

class Memory(nn.Module):
    """
    Memory bank for NTM.
    Simple implementation taken from https://github.com/loudinthecloud/pytorch-ntm    
    """
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(Memory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, beta, g, s, gamma, w_prev):
        # Content focus
        wc = self._similarity(k, beta)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        w_hat = self._shift(wg, s)
        w = self._sharpen(w_hat, gamma)
        return w

    def _similarity(self, k, beta):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(beta * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size())
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, w_hat, gamma):
        w = w_hat ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w

class LSTMController(nn.Module):
    """An NTM controller based on LSTM.
    Similar to https://github.com/loudinthecloud/pytorch-ntm  
    """
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state

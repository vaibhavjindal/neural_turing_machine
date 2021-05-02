"""The wrapper NTM class combining all the components"""
import torch
from torch import nn
import torch.nn.functional as F
from components import LSTMController, Memory, ReadHead, WriteHead


class NTM(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M):
        """Initialize the NTM.

        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTM, self).__init__()

        # Save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        self.memory = Memory(N, M)
        self.controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        
        self.read_head = ReadHead(self.memory, controller_size)
        self.write_head = WriteHead(self.memory, controller_size)
        self.heads = [self.read_head,self.write_head]

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 1
        self.init_r = [torch.randn(1, self.M) * 0.01]
        self.register_buffer("read_bias", self.init_r[0].data)
        
        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [torch.zeros(batch_size, self.N), torch.zeros(batch_size, self.N)]

        return init_r, controller_state, heads_state

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.create_new_state(batch_size)

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x=None):
        """NTM forward function.

        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM
        """
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs)

        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = self.previous_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.head_type is "read":
                r, head_state = head(controller_outp, prev_head_state)
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = torch.sigmoid(self.fc(inp2))

        # Pack the current state
        self.previous_state = (reads, controller_state, heads_states)

        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params













import torch 
import numpy as np 
import random 

#these implementations are equivalent to the dataloaders given on https://github.com/loudinthecloud/pytorch-ntm  

#generator function for dataloader
def dataloader_copy(num_batches,batch_size,seq_width,min_len,max_len):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence (8 in paper)
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length (1 in paper)
    :param max_len: Sequence maximum length (20 in paper)

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch in range(num_batches):

        # All data points in same batch must have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.uniform(0,1,(seq_len, batch_size, seq_width))
        seq[seq>0.5] = 1
        seq[seq<=0.5] = 0
        seq = seq.astype('int64')
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        yield batch+1, inp.float(), outp.float()


#generator function for dataloader
def dataloader_repeat_copy(num_batches,
                            batch_size,
                            seq_width,
                            seq_min_len,
                            seq_max_len,
                            repeat_min,
                            repeat_max,
                            reps_mean = None,
                            reps_std = None):
    """Generator of random sequences for the repeat copy task.
    Creates random batches of "bits" sequences.
    All the sequences within each batch have the same length.
    The length is between `min_len` to `max_len`
    :param num_batches: Total number of batches to generate.
    :param batch_size: Batch size.
    :param seq_width: The width of each item in the sequence.
    :param seq_min_len: Sequence minimum length.
    :param seq_max_len: Sequence maximum length.
    :param repeat_min: Minimum repeatitions.
    :param repeat_max: Maximum repeatitions.
    
    NOTE: The input width is `seq_width + 2`. One additional input
    is used for the delimiter, and one for the number of repetitions.
    The output width is `seq_width` + 1, the additional input is used
    by the network to generate an end-marker, so we can be sure the
    network counted correctly.
    """
    # Some normalization constants
    if (reps_std is None) or (reps_mean is None):
        reps_mean = (repeat_max + repeat_min) / 2
        reps_var = (((repeat_max - repeat_min + 1) ** 2) - 1) / 12
        reps_std = np.sqrt(reps_var)


    for batch in range(num_batches):

        # All batches have the same sequence length and number of reps
        seq_len = random.randint(seq_min_len, seq_max_len)
        reps = random.randint(repeat_min, repeat_max)

        # Generate the sequence
        seq = np.random.uniform(0,1,(seq_len, batch_size, seq_width))
        seq[seq>0.5] = 1
        seq[seq<=0.5] = 0
        seq = seq.astype('int64')
        seq = torch.from_numpy(seq)

        # The input includes 2 additional channels, for end-of-sequence and num-reps
        inp = torch.zeros(seq_len + 2, batch_size, seq_width + 2)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0
        inp[seq_len+1, :, seq_width+1] = (reps-reps_mean)/reps_std

        # The output contain the repeated sequence + end marker
        outp = torch.zeros(seq_len * reps + 1, batch_size, seq_width + 1)
        outp[:seq_len * reps, :, :seq_width] = seq.clone().repeat(reps, 1, 1)
        outp[seq_len * reps, :, seq_width] = 1.0 # End marker

        yield batch+1, inp.float(), outp.float()
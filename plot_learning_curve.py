"""
This file plots the learning curves for a particular json file
"""
import matplotlib.pyplot as plt
import numpy as np
import json

batch_num = 50000
json_file_path = "./checkpoints/copy-task-1000-batch-50000.json"
files = [json_file_path]


# Read the metrics from the .json files
history = [json.loads(open(fname, "rt").read()) for fname in files]
training = np.array([(x['cost'], x['loss'], x['seq_lengths']) for x in history])
print("Training history (seed x metric x sequence) =", training.shape)

# Average every dv values across each (seed, metric)
dv = 1000
training = training.reshape(len(files), 3, -1, dv).mean(axis=3)

# Average the seeds
training_mean = training.mean(axis=0)
training_std = training.std(axis=0)

fig = plt.figure(figsize=(12, 5))

# X axis is normalized to thousands
x = np.arange(dv / 1000, (batch_num / 1000) + (dv / 1000), dv / 1000)

# Plot the cost per sequence curve
plt.plot(x, training_mean[0], 'o-', linewidth=2, label='Cost')
plt.ylabel('Cost per sequence (bits)')
plt.xlabel('Sequence (thousands)')
plt.title('Training Convergence', fontsize=16)
plt.savefig("cost_vs_seq.png")
plt.close()


# Plot the bce loss per sequence curve
plt.title("BCELoss")
plt.ylabel('BCE loss per sequence')
plt.xlabel('Sequence (thousands)')
plt.plot(x, training_mean[1], 'r-', label='BCE Loss')
plt.savefig("loss_vs_seq.png")
#%%
import torch
import torch.nn as nn
import comet_ml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.quasirandom import SobolEngine
import mitdeeplearning as mdl
import os
mpl.use("TkAgg")

'''
The purpose of this script is just to invert a simple 3 input 1 output function. I say it's simple because the inputs
are constrained along a line in 3d space so it's actually a 1 to 1 problem. A neural network is probably overkill here
but I have the code and it's faster for me to do train this than it is for me to code the analytic form of the line and
use a cubic spline.

Regardless, none of this will generalise well to higher dimensions so I just want to do this quickly.
'''

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = dict(
  num_training_iterations = 500,
  batch_size = 8,
  learning_rate = 5e-3,
)

COMET_API_KEY = "bF1GzyP5VGyQny3ROlnPzaTQo"

train_x = np.load("subset_X.npy")
train_y = np.load("subset_Y.npy")

training_x = torch.tensor(train_x[~np.isnan(train_x).any(axis=1)], dtype=dtype, device=device)
training_y = torch.tensor(train_y[~np.isnan(train_y)], dtype=dtype, device=device).unsqueeze(-1)

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "1d_inversion")
os.makedirs(checkpoint_dir, exist_ok=True)

def get_batch(training_x, training_y, batch_size):
    '''
    This batch will take 'batch_size' points from the training set.
    :return:
    '''
    generator = SobolEngine(dimension=1)
    idx = torch.round(generator.draw(batch_size, dtype=dtype).to(device)*len(training_x)).int()
    idx = idx.squeeze(-1).numpy()

    x_batch = training_x[idx, :].to(dtype=dtype, device=device)
    y_batch = training_y[idx,:].to(dtype=dtype, device=device)
    return x_batch, y_batch

MAE = nn.L1Loss()
def compute_loss(training_x, model_x):
    loss = MAE(training_x, model_x)
    return loss

def train_step(x, y):
    # Set the model's mode to train
    global test
    test = y
    model.train()

    # Zero gradients for every step
    optimizer.zero_grad()

    # Forward pass
    global x_hat, xhat2
    x_hat2 = model(y)
    x_hat = model(y).squeeze(-1)

    # Compute the loss
    loss = compute_loss(x, x_hat)
    print(loss)

    loss.backward()
    optimizer.step()

    return loss

def create_experiment():
    # end any prior experiments
    if 'experiment' in locals():
      experiment.end()

    # initiate the comet experiment for tracking
    experiment = comet_ml.Experiment(
                  api_key=COMET_API_KEY,
                  project_name="1d_inversion")
    # log our hyperparameters, defined above, to the experiment
    for param, value in params.items():
      experiment.log_parameter(param, value)
      experiment.flush()

    return experiment

model = nn.Sequential(
    nn.Linear(1, 128),
    nn.GELU(),
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 128),
    nn.GELU(),
    nn.Linear(128, 3)
)

model.double()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
experiment = create_experiment()

for iteration in range(params["num_training_iterations"]):
    print("iteration", iteration)
    x_batch, y_batch = get_batch(training_x, training_y, params['batch_size'])
    assert x_batch.shape == (params['batch_size'], 3), "x_batch shape is incorrect"
    assert y_batch.shape == (params['batch_size'], 1), "y_batch shape is incorrect"

    # Take a train step
    loss = train_step(x_batch, y_batch)

    # Log the loss to the Comet interface
    experiment.log_metric("loss", loss.item(), step=iter)

    # Update the progress bar and visualize within notebook/
    history.append(loss.item())
    plotter.plot(history)

torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()
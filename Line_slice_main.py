#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RBFInterpolator

mpl.use('TkAgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

'''
I have realised that I can find the slice that I need by exploiting the fact that the function is continuous.
That means that I can use a basic Bayesian Optimiser to find the maximum in the space. Then define a line
through that maxima that is parameterised by a direction vector. I define a second Bayesian optimiser to ensure
that the minima along this line is minimised. If the 1-d function that remains is injective, then great. Otherwise,
it is a straightforward problem to restrict the domain in order to make the function injective.
Note that this DOES NOT seem to naturally generalise to multi objective cases.

This can be used for training more efficiently. Optimise for the global maxima using Bayesian Optimisation.
Then define a line in input space and follow it through to the zero point. THE FULL GRID SCAN IS NOT REQUIRED

The direction that I choose might be it's own optimisation problem but I think practically, this is a very easy one
to solve.

for 1d optimisation only, the 1d slice will not be able to access the full plane of viable solutions. Intuition says
that it will require a 2d manifold but there is no similarly easy way of finding this. I am hoping that the complexity
of the function will render it automatically injective. We will see.
'''
#%% We start by loading in the model.
vals = np.load("vals.npy")
vals2 = np.load("vals2.npy")

#Note that this interpolator is just simulating us have fired many more shots and is really just enforcing a
#smoothness constraint. In reality, I will want to drop this later as the neural network should handle all of this.
interp_counts = RBFInterpolator(vals2, vals)

#%% And then we interpolate to form the grid.
plt.close("all")
'''
x_grid = np.arange(np.min(vals2[:, 0]), np.max(vals2[:,0]), 0.001) #focus
y_grid = np.arange(np.min(vals2[:, 1]), np.max(vals2[:,1]), 0.005)  #astig0
z_grid = np.arange(np.min(vals2[:, 2]), np.max(vals2[:,2]), 0.005) #astig45
'''

x_grid = np.linspace(np.min(vals2[:, 0]), np.max(vals2[:,0]), 20) #focus
y_grid = np.linspace(np.min(vals2[:, 1]), np.max(vals2[:,1]), 20)  #astig0
z_grid = np.linspace(np.min(vals2[:, 2]), np.max(vals2[:,2]), 20) #astig45

X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

V_counts = interp_counts(points)
#V_counts_grid = V_counts.reshape(X.shape)

mask = ~np.isnan(V_counts)          # shape (N,)
points_masked_counts = points[mask] # (n_valid, 3)
values_masked_counts = V_counts[mask]

Xcf, Ycf, Zcf = points_masked_counts.T
Vcf = values_masked_counts
colors_counts = Vcf

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(Xcf, Ycf, Zcf, c=colors_counts, cmap='viridis', s=10)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('charge (arb)', fontsize=20)
cbar.ax.tick_params(labelsize=12)
#ax.set_title("3", fontsize=26)
ax.set_xlabel('focus (mm)', fontsize=16)
ax.set_ylabel('astigmatism 0 (arb.)', fontsize=16)
ax.set_zlabel('astigmatism 45 (arb.)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

#%% We want to start the Newton-Raphson from the global optimum. We will sample the NN with Bayesian optimisation
#Note, this is not going to generalise well for systems with higher output dimensions but we start with the
#easy case

from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf

from typing import Optional

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    qLogExpectedImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler



warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SMOKE_TEST = False
verbose = True

bounds1 = torch.tensor(
    [
        [np.min(vals2[:, 0]), np.min(vals2[:, 1]), np.min(vals2[:, 2])] ,
        [np.max(vals2[:, 0]), np.max(vals2[:, 1]), np.max(vals2[:,2])]
    ], device=device, dtype=dtype)

NOISE_SE = 0.25
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

BATCH_SIZE = 3 if not SMOKE_TEST else 2
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32
N_TRIALS = 3 if not SMOKE_TEST else 2
N_BATCH = 20 if not SMOKE_TEST else 2
MC_SAMPLES = 256 if not SMOKE_TEST else 32


def fire(tensor_input):
    array_output = interp_counts(tensor_input.numpy())
    tensor_output = torch.from_numpy(array_output).to(device)
    return tensor_output

def generate_inital_data(bounds, evaluator, n=10, Minimise=False):
    #Generates x in real space and fires in real space
    global obj, train_x
    lower = bounds[0]
    upper = bounds[1]
    train_x = lower + (upper - lower) * torch.rand(n, bounds.shape[1], device=device, dtype=dtype)
    obj = evaluator(train_x).unsqueeze(-1)

    if Minimise:
        obj = -1*obj

    observed = obj + torch.rand_like(obj) * NOISE_SE
    best_observed_value = observed.max().item()
    best_X_value = train_x[torch.argmax(observed),:]
    return train_x, observed, best_observed_value, best_X_value

def initialise_model(train_X, train_obj, state_dict=None):
    #accepts real space inputs for train_X which are then normalised
    model = SingleTaskGP(
        train_X,
        train_obj,
        train_yvar.expand_as(train_obj),
        input_transform=Normalize(d=train_X.shape[-1]),
    ).to(train_X)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]

def optimise_and_acquire(acq_func, evaluator, bounds, Minimise=False):
    candidates, _ = optimize_acqf(
        acq_function = acq_func,
        bounds = bounds,
        q = BATCH_SIZE,
        num_restarts = NUM_RESTARTS,
        raw_samples = RAW_SAMPLES,
        options={'batch_limit':5, 'maxiter':200},
    )

    #oberve
    new_x = candidates.detach()
    #might have to unnormalise here
    obj = evaluator(new_x)
    if Minimise:
        obj = -1*obj
    observed = obj + NOISE_SE * torch.randn_like(obj)
    return new_x, observed

def single_task_optimiser(bounds, evaluator, Minimise=False, verbose=True):

    best_observed_all = []
    best_X_all = []

    t_start = time.monotonic()
    for trial in range(1, N_TRIALS+1):
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed = []
        best_X = []

        (
            train_X,
            train_obj,
            best_observed_val,
            best_X_val,
        ) = generate_inital_data(bounds=bounds, evaluator=evaluator, n=10, Minimise=Minimise)

        mll, model = initialise_model(train_X, train_obj)
        best_observed.append(best_observed_val)
        best_X.append(best_X_val)

        for iteration in range(1, N_BATCH+1):
            print("iteration: " + str(iteration))
            t0 = time.monotonic()

            fit_gpytorch_mll(mll)

            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

            qLogEI = qLogExpectedImprovement(
                model=model,
                sampler=qmc_sampler,
                best_f = train_obj.max(),
            )

            new_X, observed = optimise_and_acquire(qLogEI, evaluator, bounds=bounds, Minimise=Minimise)

            train_X = torch.cat((train_X, new_X))
            train_obj = torch.cat((train_obj, observed.unsqueeze(-1)))

            best_observed.append(train_obj.max().item())
            best_X.append(train_X[torch.argmax(train_obj), :])

            mll, model = initialise_model(train_X, train_obj)

            t1 = time.monotonic()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value  = "
                    f"({max(best_observed):>4.2f}), "
                    f"time = {t1 - t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")

        best_observed_all.append(max(best_observed))
        best_X_all.append(best_X[np.argmax(best_observed)])


    t_finish = time.monotonic()
    if verbose:
        print(
            f"\nAll iterations complete: best_value  = "
            f"({max(best_observed_all):>4.2f}), "
            f"total time = {t_finish - t_start:>4.2f} seconds.",
            end="",
        )

    return best_X_all, best_observed_all
#%% We now have the max val and the location of the max val
best_X_all, best_observed_all = single_task_optimiser(bounds1, fire, verbose=verbose)
max_Y = max(best_observed_all)
max_X = best_X_all[np.argmax(best_observed_all)]
#%% We will consider 1d slices through this point. We are looking for a slice which
#Is continuous and goes from max to zero

from scipy.ndimage import map_coordinates
X = max_X.numpy()
bounds_arr = bounds1.numpy()

#Gives the direction of the slice
a = 0.5862
b = 0.2948
c = 0.3524
direction = torch.tensor([a, b, c], device=device, dtype=dtype)
N = 500

start_points = np.array([bounds_arr[0,:]])
end_points = np.array([bounds_arr[1,:]])
length = end_points - start_points

def slice(direction):
    direc = direction.numpy()
    nhat = direc/np.linalg.norm(direc)

    #Line passing through the peak point over the box domain with the direction
    #defined above by a, b and c.
    s = np.linspace(0, 1, N)
    end_set = X + nhat*length * s[:, None]
    start_set = X - nhat*length* s[:, None]
    full_set = np.concatenate((start_set, end_set))

    mask = (start_points < full_set) & (full_set < end_points)
    cut = np.where(mask, full_set, np.nan)

    output = interp_counts(cut)
    return output, cut

def test_slice(direction):
    return np.nanmin(slice(direction)[0])

print(test_slice(direction))
#%%
plt.figure(figsize=(7,5))
plt.plot(slice(direction)[0])

#%% We define a second optimisation problem. This time, we will minimise the minima
#of the slice
bounds2 = torch.tensor([[-1.0] * 3, [1.0] * 3], device=device, dtype=dtype)
def min_slice(direction):
    #this function must be robust to receiving n tensors rather than just one.
    N1 = direction.shape[0]
    output = torch.empty(N1, device=device, dtype=dtype)
    for t in range(N1):
        out = np.nanmin(slice(direction[t,:])[0])
        output[t] = torch.tensor(out, device=device, dtype=dtype)
    return output

SMOKE_TEST = False
best_X_all2, best_observed_all2 = single_task_optimiser(bounds2, min_slice, Minimise=True, verbose=verbose)

#%% And now we extract the line subset that we desire
max_Y2 = max(best_observed_all2)
max_X2 = best_X_all2[np.argmax(best_observed_all2)]

subset_Y, subset_X = slice(max_X2)
plt.figure(figsize=(10,7))
plt.plot(subset_Y)
plt.title("line slice", fontsize=26)
plt.ylabel("charge (arb.)", fontsize=20)
plt.xlabel("500 steps along the line", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)

np.save("subset_Y.npy", subset_Y)
np.save("subset_X.npy", subset_X)






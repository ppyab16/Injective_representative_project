'''
I am using the same 1d test case as in the 1_dimensional testing from before.
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RBFInterpolator

mpl.use('TkAgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

#%% We start by loading in the model.
vals = np.load("vals.npy")
vals2 = np.load("vals2.npy")

#Note that this interpolator is just simulating us have fired many more shots and is really just enforcing a
#smoothness constraint. In reality, I will want to drop this later as the neural network should handle all of this.
interp_counts = RBFInterpolator(vals2, vals)

#%% And then we interpolate to form the grid.
plt.close("all")

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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(Xcf, Ycf, Zcf, c=colors_counts, cmap='viridis', s=10)
fig.colorbar(sc, ax=ax, label='Interpolated value')
ax.set_title("counts")
ax.set_xlabel('focus')
ax.set_ylabel('astig0')
ax.set_zlabel('astig45')

#%%
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
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

from Bayesian_grid_search.Objective_Grid_Search import ObjectiveGridSearch

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SMOKE_TEST = False
verbose = True

bounds1 = torch.tensor(
    [
        [np.min(vals2[:, 0]), np.min(vals2[:, 1]), np.min(vals2[:, 2])] ,
        [np.max(vals2[:, 0]), np.max(vals2[:, 1]), np.max(vals2[:,2])]
    ], device=device, dtype=dtype)

NOISE_SE = 0.0
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

BATCH_SIZE = 3 if not SMOKE_TEST else 2
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32
N_TRIALS = 1 if not SMOKE_TEST else 1
N_BATCH = 20 if not SMOKE_TEST else 20
MC_SAMPLES = 256 if not SMOKE_TEST else 32

N_TRIALS2 = 1 if not SMOKE_TEST else 1
N_BATCH2 = 20 if not SMOKE_TEST else 2


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
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3)),
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

    return best_X_all, best_observed_all, train_X, train_obj

def Grid_optimise_and_acquire2(model, best_f, distance, evaluator, bounds, sigma_ref):
    k=1
    certain = True
    while certain:
        acqf = ObjectiveGridSearch(
            model=model,
            best_f=best_f,
            distance=distance,
            k=k,
        )

        candidate, _ = optimize_acqf(
            acq_function = acqf,
            bounds = bounds,
            q = 1,
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES,
            options={'batch_limit':5, 'maxiter':200},
        )
        _, sigma = acqf._mean_and_sigma(candidate)
        if sigma.item() < sigma_ref:
            print("progress")
            k += 1
            if k > 20:
                return None, None, k
        else:
            print("hello?")
            certain = False

    #oberve
    new_x = candidate.detach()
    #might have to unnormalise here
    obj = evaluator(new_x)
    print("fire")
    observed = obj + NOISE_SE * torch.randn_like(obj)
    return new_x, observed, k

def Grid_optimise_and_acquire(model, best_f, distance, evaluator, bounds, sigma_ref, k_cap=15):
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    best_f = torch.as_tensor(best_f, device=device, dtype=dtype)
    bounds = bounds.to(device=device, dtype=dtype)

    k = 1
    while True:
        acqf = ObjectiveGridSearch(
            model=model,
            best_f=best_f,
            distance=distance,
            k=k,
        )

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )

        with torch.no_grad():
            post = model.posterior(candidate.unsqueeze(-2))
            sigma = post.variance.sqrt().squeeze()

        if sigma.item() < sigma_ref:
            print("progress")
            k += 1
            if k > k_cap:
                return None, None, k
            continue
        else:
            break

    with torch.no_grad():
        new_x = candidate.detach()
        print(new_x.numpy())
        obj = evaluator(new_x)
        print((obj-(best_f-k*distance)).item())
        observed = obj + NOISE_SE * torch.randn_like(obj)

    return new_x, observed, k

def evaluate_candidates(model, best_f, distance, evaluator, bounds):
    final_X = np.ones((20, 3))
    final_Y = np.ones((20,1))
    for k in range(20):
        print(k)
        acqf = ObjectiveGridSearch(
            model=model,
            best_f=best_f,
            distance=distance,
            k=k,
        )

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        new_x = candidate.detach()
        print(new_x.numpy())
        evaluated = evaluator(new_x)
        final_X[k, :] = new_x.numpy()
        final_Y[k, :] = evaluated.detach().numpy()
    return final_X, final_Y

def Grid_optimiser(bounds, evaluator, train_X, train_obj, distance, best_f, sigma_ref, verbose=True):
    mll, model = initialise_model(train_X, train_obj)
    iteration = 1
    while True:
        print("iteration: " + str(iteration))
        t0 = time.monotonic()

        fit_gpytorch_mll(mll)

        new_X, observed, k = Grid_optimise_and_acquire(
            model=model,
            best_f=best_f,
            distance=distance,
            evaluator=evaluator,
            bounds=bounds,
            sigma_ref=sigma_ref,
        )

        if k > 15:
            print("entering final loop")
            final_X, final_Y = evaluate_candidates(
                model=model,
                best_f=best_f,
                distance=distance,
                evaluator=evaluator,
                bounds=bounds,
            )

            return final_X, final_Y

        train_X = torch.cat((train_X, new_X))
        train_obj = torch.cat((train_obj, observed.unsqueeze(-1)))
        mll, model = initialise_model(train_X, train_obj)

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value  = "
                f"({k:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

        iteration += 1

best_X_all, best_observed_all, train_X, train_obj = single_task_optimiser(bounds1, fire, verbose=verbose)

final_X, final_Y = Grid_optimiser(bounds1, fire, train_X, train_obj, distance=1., best_f=best_observed_all[0], sigma_ref=0.01)

#%%
from matplotlib.ticker import MultipleLocator
plt.close("all")
plt.figure(figsize=(12,8))
x = np.linspace(1, 20, 20)
plt.plot(x, final_Y[:,0], '*')
plt.xlabel("shot number", fontsize=20)
plt.ylabel("charge (arb.)", fontsize=20)
plt.tick_params("both", labelsize=15)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(1))

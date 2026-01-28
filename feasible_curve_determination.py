import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")
import math
from typing import Optional
from scipy.interpolate import RBFInterpolator

from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel

import time
import warnings

from botorch.exceptions import BadInitialCandidatesWarning
from botorch.acquisition.objective import GenericMCObjective, ConstrainedMCObjective
from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition import qSimpleRegret

from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

#%% Loading in the multi objecgtive target as normal
obj1 = np.load("vals.npy")
#We define a second output which has peaks in a different place and apply a nonlinear function to it to change the
#landscape slightly without creating any new peaks or new structure.
obj2 = np.roll(obj1, 15) ** 1.2
points = np.load("vals2.npy")

#Note that this interpolator is just simulating us have fired many more shots and is really just enforcing a
#smoothness constraint. In reality, I will want to drop this later as the neural network should handle all of this.
interp_obj1 = RBFInterpolator(points, obj1)
interp_obj2 = RBFInterpolator(points, obj2)

# Visualise both the output functions
plt.close("all")

def visualise(points, interp):
    x_grid = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 20) #focus
    y_grid = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 20)  #astig0
    z_grid = np.linspace(np.min(points[:, 2]), np.max(points[:, 2]), 20) #astig45

    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

    set = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    V_counts = interp(set)

    mask = ~np.isnan(V_counts)  # shape (N,)
    points_masked_counts = set[mask]  # (n_valid, 3)
    values_masked_counts = V_counts[mask]

    Xcf, Ycf, Zcf = points_masked_counts.T
    Vcf = values_masked_counts
    colors_counts = Vcf

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(Xcf, Ycf, Zcf, c=colors_counts, cmap='viridis', s=10)
    fig.colorbar(sc, ax=ax, label='Interpolated mean charge (arb.)')
    ax.set_title("3d deformable mirror grid scan")
    ax.set_xlabel('focus (mm)')
    ax.set_ylabel('astig0 (arb.)')
    ax.set_zlabel('astig45 (arb.)')

    return

visualise(points, interp_obj1)
visualise(points, interp_obj2)


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

NOISE_SE = 0.0
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

DIM=2

def fire(tensor_input):
    # returns shape (N, 2)
    array_output = np.column_stack((interp_obj1(tensor_input.numpy()), interp_obj2(tensor_input.numpy())))
    tensor_output = torch.from_numpy(array_output).to(device)
    return tensor_output

def c_gate(model, candidate, target, value_gate=0.1):
    global test
    test = target
    with torch.no_grad():
        # This is only going to work on 2d output in it's current form
        post = model.posterior(candidate.unsqueeze(-2))
        sigma = post.variance.sqrt().squeeze()
        mean = post.mean.squeeze()
        normal_dist = torch.distributions.normal.Normal(mean[0], sigma[0])
        range = torch.tensor([-target.item() + value_gate, value_gate + target.item()], device=device, dtype=dtype)
        probability = normal_dist.cdf(range[1]) - normal_dist.cdf(range[0])
    return probability

def generate_initial_data(bounds, evaluator, n=10, Minimise=False):
    global test2
    # Generates x in real space and fires in real space
    upper = bounds[1]
    lower = bounds[0]
    train_x = lower + (upper - lower) * torch.rand(n, bounds.shape[1], device=device, dtype=dtype)
    obj = evaluator(train_x)

    if Minimise:
        obj = -1 * obj

    observed = obj + torch.rand_like(obj) * NOISE_SE
    test2 = observed
    return train_x, observed

def initialise_model(train_X, train_Y, state_dict=None):
    model1 = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y[:, [0]],
        input_transform=
        Normalize(d=DIM, bounds=bounds.to(train_X.device)),
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=DIM)),
    )
    model2 = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y[:, [1]],
        input_transform=
        Normalize(d=DIM, bounds=bounds.to(train_X.device)),
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=DIM)),
    )
    model = ModelListGP(model1, model2)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def optimise_and_acquire(model, training_X, sampler, f1, evaluator, Minimise=False):
    model.eval()
    objective = GenericMCObjective(obj_callable)
    constraint = constraint_callable(f1)

    acqf = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=training_X,
        sampler=sampler,
        objective=objective,
        constraints=[constraint],
    )

    print("optimise the batches")
    candidate, EI = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 100},
    )
    print("batch optimised")

    obj = evaluator(candidate)

    if Minimise:
        obj = -1 * obj

    observed = obj + torch.rand_like(obj) * NOISE_SE

    return observed, candidate, EI

def final_guess(model, f1, evaluator, sampler):
    model.eval()
    print("f1")
    print(f1.item())
    objective = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable(f1)]
    )

    acqf = qSimpleRegret(
        model=model,
        sampler=sampler,
        objective=objective,
    )

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 50, "maxiter": 200},
    )

    obj = evaluator(candidate)
    observed = obj + torch.rand_like(obj) * NOISE_SE

    return observed, candidate

def find_optimum(f1, train_X, train_Y, EI_thresh, evaluator, constraint_gate=0.99, Minimise=False, verbose=True):
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
    iteration=0
    while True:
        global testing
        testing = train_X
        t0 = time.monotonic()

        print("initialise model")
        mll, model = initialise_model(train_X, train_Y)
        print("model initialised")

        observed, candidate, EI = optimise_and_acquire(
            model=model,
            training_X=train_X,
            sampler=sampler,
            f1=f1,
            evaluator=evaluator,
            Minimise=Minimise,
        )
        probability_constraint = c_gate(model, candidate, f1)
        print(probability_constraint.item())
        if EI.item() < np.log(EI_thresh) and probability_constraint.detach().numpy()>constraint_gate:
            observed, candidate = final_guess(model, f1, evaluator, sampler)
            train_X = torch.cat((train_X, candidate))
            train_Y = torch.cat((train_Y, observed))
            return observed, train_X, train_Y

        train_X = torch.cat((train_X, candidate))
        train_Y = torch.cat((train_Y, observed))

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: EI  = "
                f"({torch.exp(EI).item():>4.2f}), "
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

        iteration += 1
        plt.pause(0.1)

def constraint_callable(f1: torch.Tensor, value_gate=0.5):
    '''
        The constraint is that the first dimensions of the output do not change with respect to the input
        with a tolerated error of some gate. IE, you will get the optimiser to return this value for a candidate
        and ensure it lies within a certain range
    '''
    def c1(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
        return torch.sqrt((Z[..., 0] - f1).pow(2).sum(dim=0)) - value_gate
    return c1

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 1]

def get_feasibility_curve(min, max, n_points, evaluator, EI_thresh=0.1, Minimise=False, verbose=True):
    global train_X
    output = torch.zeros(n_points,2)
    f1 = torch.linspace(min, max, n_points, device=device, dtype=dtype)

    train_X, observed = generate_initial_data(bounds, evaluator, n=10, Minimise=Minimise)

    for i, f in enumerate(f1):
        print("iteration: " + str(i))
        global new_X
        out, new_X, new_Y = find_optimum(
            f1=f,
            train_X=train_X,
            train_Y=observed,
            EI_thresh=EI_thresh,
            evaluator=evaluator,
            Minimise=Minimise,
            verbose=verbose,
        )
        output[i] = out

        train_X = torch.cat((train_X, new_X))
        observed = torch.cat((observed, new_Y))

    return f1, output.detach().numpy(), train_X.detach().numpy(), observed.detach().numpy()

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

bounds = torch.tensor(
    [
        [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])],
        [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]
    ], device=device, dtype=dtype)

SMOKE_TEST = False

DIM = 3
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32
N_BATCH = 20 if not SMOKE_TEST else 1
MC_SAMPLES = 256 if not SMOKE_TEST else 32
N_TRIALS = 1 if not SMOKE_TEST else 1
NUM_PARETO = 2 if SMOKE_TEST else 10
NUM_FANTASIES = 2 if SMOKE_TEST else 8

#%%
f1, output, train_X, train_Y = get_feasibility_curve(20, 30, 5, fire)

#%%
plt.close("all")
plt.figure()
plt.plot(train_Y[:,0], train_Y[:,1], 'b.')
plt.plot(output[:,0], output[:,1], 'r*')
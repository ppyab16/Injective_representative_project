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

#%% Loading in the multi objective target as normal
obj1 = np.load("vals.npy")
#We define a second output which has peaks in a different place and apply a nonlinear function to it to change the
#landscape slightly without creating any new peaks or new structure.
obj2 = np.roll(obj1, 15) ** 1
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
    cbar = fig.colorbar(sc, ax=ax, label='|electron charge| (arb.)')
    cbar.ax.set_position([0.85, 0.15, 0.03, 0.7])
    ax.set_title("3d deformable mirror grid scan")
    ax.set_xlabel('focus (mm)')
    ax.set_ylabel('0 degree astigmatism (arb.)')
    ax.set_zlabel('45 degree astigmatism (arb.)')

    return

visualise(points, interp_obj1)
visualise(points, interp_obj2)
#%%
import torch
import math
from typing import Optional
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel

from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning
)

from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.analytic import PosteriorStandardDeviation
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from torch.quasirandom import SobolEngine

from botorch.acquisition.multi_objective import (
    qLogExpectedHypervolumeImprovement
)

from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
)

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning

from scipy.interpolate import CubicSpline

from botorch.sampling.normal import SobolQMCNormalSampler


bounds = torch.tensor(
    [
        [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])],
        [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]
    ], device=device, dtype=dtype)

bounds_arr = bounds.detach().numpy()

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ref_point = torch.tensor([-600.0, -2000.0], device=device, dtype=dtype)

SMOKE_TEST = False

DIM = 3
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32
N_BATCH = 20 if not SMOKE_TEST else 1
MC_SAMPLES = 256 if not SMOKE_TEST else 32
N_TRIALS = 1 if not SMOKE_TEST else 1
NUM_PARETO = 2 if SMOKE_TEST else 10
NUM_FANTASIES = 2 if SMOKE_TEST else 8

NOISE_SE = 0.0
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

def fire(tensor_input):
    # returns shape (N, 2)
    array_output = np.column_stack((interp_obj1(tensor_input.numpy()), interp_obj2(tensor_input.numpy())))
    tensor_output = torch.from_numpy(array_output).to(device)
    return tensor_output

def generate_initial_data(bounds, evaluator, n=10, Minimise=False):
    global test2
    # Generates x in real space and fires in real space
    lower = bounds[0]
    upper = bounds[1]
    train_x = lower + (upper - lower) * torch.rand(n, bounds.shape[1], device=device, dtype=dtype)
    obj = evaluator(train_x)

    if Minimise:
        obj = -1 * obj

    observed = obj + torch.rand_like(obj) * NOISE_SE
    test2 = observed
    return train_x, observed

# initialise the model. This is what keeps retraining the model on new inputs
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

#Helper to the HVKG. This is copied exactly from the botorch tutorial
def get_current_value(
    model,
    ref_point,
    bounds,
):
    """Helper to get the hypervolume of the current hypervolume
    maximizing set.
    """
    curr_val_acqf = _get_hv_value_function(
        model=model,
        ref_point=ref_point,
        use_posterior_mean=True,
    )
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds,
        q=NUM_PARETO,
        num_restarts=20,
        raw_samples=1024,
        return_best_only=True,
        options={"batch_limit": 5},
    )
    return current_value

# Different acquisition functions can be used for training, they each have their own problems

#Finds the pareto front efficiently but we want the feasibility curve
def optimise_and_acquire_EHVI(model, mll, X_all, sampler, bounds, evaluator, Minimise=False):
    model.train()
    model.likelihood.train()
    fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()

    # Stack raw X (lists -> tensor)
    X_all = torch.stack(
        [x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=dtype)
         for x in X_all],
        dim=0,
    ).to(device=device, dtype=dtype)
    print(X_all.shape)
    with torch.no_grad():
        pred = model.posterior(X_all).mean

    # We can try NonDominatedPartioning if the Fast algorithm turns out to be quite slow
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=pred
    )

    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    # Try setting sequential to False to see if performance improves.
    candidates, EHVI = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        num_restarts=NUM_RESTARTS,
        q=1,
        sequential=True,
        raw_samples=RAW_SAMPLES,
        options={
            "maxiter": 200,
            "batch_limit": 5,
        },
    )
    ehvi_log = EHVI.item() if torch.is_tensor(EHVI) else float(EHVI)
    ehvi = np.exp(ehvi_log)

    new_x = candidates.detach()
    print("x incoming")
    print(new_x)
    obs = evaluator(new_x)

    if Minimise:
        obs = -1 * obs
    observed = obs + NOISE_SE * torch.randn_like(obs)

    return new_x, observed, ehvi

#Finds the pareto front even more efficiently but it's extremely slow to compute
def optimise_and_acquire_HVKG(model, mll, bounds, evaluator, Minimise=False):
    fit_gpytorch_mll(mll=mll)

    current_value = get_current_value(
        model=model,
        ref_point=ref_point,
        bounds=bounds,
    )

    acq_func = qHypervolumeKnowledgeGradient(
        model=model,
        ref_point=ref_point,  # use known reference point
        num_fantasies=NUM_FANTASIES,
        num_pareto=NUM_PARETO,
        current_value=current_value,
    )

    candidates, EHVI = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        num_restarts=NUM_RESTARTS,
        q=1,
        sequential=True,
        raw_samples=RAW_SAMPLES,
        options={
            "maxiter": 200,
            "batch_limit": 5,
        },
    )
    ehvi_log = EHVI.item() if torch.is_tensor(EHVI) else float(EHVI)
    ehvi = np.exp(ehvi_log)

    new_x = candidates.detach()
    print("x incoming")
    print(new_x)
    obs = evaluator(new_x)

    if Minimise:
        obs = -1 * obs
    observed = obs + NOISE_SE * torch.randn_like(obs)

    return new_x, observed, ehvi

#Samples the point with maximum uncertainty. BCD doesn't like this as it can trick the algorithm into jumping
#from one contour to another. Training like this also scales very poorly to high input dimensional spaces
def optimise_and_acquire_var(model, mll, evaluator, Minimise=False):
    fit_gpytorch_mll(mll=mll)
    pt = ScalarizedPosteriorTransform(weights=torch.ones(2, device=device, dtype=dtype))
    acqf = PosteriorStandardDeviation(model=model, posterior_transform=pt)

    candidates, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    new_x = candidates.detach()
    obs = evaluator(new_x)
    observed = obs + NOISE_SE * torch.randn_like(obs)

    return new_x, observed, acq_value

#Similar idea to the above except it minimises the expected integrated posterior variance across the entire space
#after the acquisition is made. It's extremely slow and has the same issues as above
def optimise_and_acquire_post_var(model, mll, sampler, evaluator, Minimise=False):
    fit_gpytorch_mll(mll=mll)

    sobol_engine = SobolEngine(dimension=3)
    samples = sobol_engine.draw(MC_SAMPLES)

    pt = ScalarizedPosteriorTransform(weights=torch.ones(2, device=device, dtype=dtype))
    acq_func = qNegIntegratedPosteriorVariance(
        model = model,
        mc_points = samples,
        sampler=sampler,
        posterior_transform=pt
    )

    # Try setting sequential to False to see if performance improves.
    candidates, acq_val = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        num_restarts=NUM_RESTARTS,
        q=1,
        sequential=True,
        raw_samples=RAW_SAMPLES,
        options={
            "maxiter": 200,
            "batch_limit": 5,
        },
    )

    acq = acq_val.item() if torch.is_tensor(acq_val) else float(acq_val)

    new_x = candidates.detach()
    print("x incoming")
    print(new_x)
    obs = evaluator(new_x)
    observed = obs + NOISE_SE * torch.randn_like(obs)

    return new_x, observed, acq

#The MOBO loop
def MOBO(bounds, evaluator, n_training, Minimise=False, verbose=True):
    print('beginning loop')
    state_dicts = []
    t_start = time.monotonic()
    lengthscales_final = np.ones((N_BATCH,2, DIM))
    EHVI_final = torch.ones((N_BATCH, N_TRIALS), device=device, dtype=dtype)
    train_x_final = torch.ones((N_BATCH + n_training, DIM, N_TRIALS), device=device, dtype=dtype)
    objective_final = torch.ones((N_BATCH + n_training, 2, N_TRIALS), device=device, dtype=dtype)

    for trial in range(1, N_TRIALS + 1):
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        EHVI = torch.ones((N_BATCH), device=device, dtype=dtype)

        (
            train_X,
            train_obj,
        ) = generate_initial_data(bounds=bounds, evaluator=evaluator, n=n_training, Minimise=Minimise)

        mll, model = initialise_model(train_X=train_X, train_Y=train_obj)

        for iteration in range(1, N_BATCH + 1):
            print("iteration: " + str(iteration))
            t0 = time.monotonic()


            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            new_X, observed, EHVI_new = optimise_and_acquire_EHVI(
                model=model,
                mll=mll,
                X_all=train_X,
                sampler=qmc_sampler,
                evaluator=evaluator,
                bounds=bounds,
                Minimise=Minimise
            )
            print('optimised')
            '''
            new_X, observed, EHVI_new = optimise_and_acquire_var(
                model=model,
                mll=mll,
                evaluator=evaluator,
                Minimise=Minimise
            )
            
            
            new_X, observed, EHVI_new = optimise_and_acquire_HVKG(
                model=model,
                mll=mll,
                evaluator=evaluator,
                bounds=bounds,
                Minimise=Minimise
            )
            
            new_X, observed, EHVI_new = optimise_and_acquire_post_var(
                model=model,
                mll=mll,
                sampler=qmc_sampler,
                evaluator=evaluator,
            )
            '''
            train_X = torch.cat((train_X, new_X))
            train_obj = torch.cat((train_obj, observed))
            print(train_X.shape)
            print(train_obj.shape)

            EHVI[iteration - 1] = EHVI_new

            '''
            for i, m in enumerate(model.models):
                print('2')
                m.set_train_data(train_X, train_obj[:, [i]], strict=False)
                print('3')
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            print('4')
            '''
            dict = model.state_dict()
            mll, model = initialise_model(train_X=train_X, train_Y=train_obj, state_dict=dict)
            t1 = time.monotonic()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: acq_value  = "
                    f"({EHVI_new:>4.2f}), "
                    f"time = {t1 - t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")

        dict = model.state_dict()
        state_dicts.append(dict)

        EHVI_final[:, trial - 1] = EHVI
        train_x_final[:, :, trial - 1] = train_X
        objective_final[:, :, trial - 1] = train_obj
        lengthscales = [
            m.covar_module.base_kernel.lengthscale.detach().cpu().numpy().squeeze(0)
            for m in model.models
        ]  # list of (3,)

        lengthscales_final[trial - 1, :, :] = np.stack(lengthscales, axis=0)
    t_finish = time.monotonic()
    if verbose:
        print(
            f"\nAll iterations complete:",
            f"total time = {t_finish - t_start:>4.2f} seconds.",
            end="",
        )

    return state_dicts, EHVI_final, train_x_final, objective_final, lengthscales_final


#%%
state_dicts, EHVI_final, train_x_final, objective_final, lengthscales = MOBO(bounds, fire, n_training=10)

import os
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "MOBO_master")
os.makedirs(checkpoint_dir, exist_ok=True)

torch.save({
    "state_dicts": state_dicts,
    "EHVI_final": EHVI_final,
    "train_x_final": train_x_final,
    "objective_final": objective_final,
    "bounds": bounds,
}, checkpoint_prefix)

#%% Plotting
objective_final_array = objective_final.detach().numpy()

plt.figure(figsize=(10,7))
for i in range(N_TRIALS):
    plt.plot(objective_final_array[:,0,i], objective_final_array[:,1,i], '*')
    plt.xlabel("objective1 (arb.)", fontsize=20)
    plt.ylabel("objective2 (arb.)", fontsize=20)
    plt.tick_params("both", labelsize=12)

EHVI_array = EHVI_final.detach().numpy()

plt.figure()
for i in range(N_TRIALS):
    plt.plot(EHVI_array[:,i])
    plt.xlabel("iteration")
    plt.ylabel("acquisition_value")
#%%
from two_dimensional.feasible_from_model import main
upper_feasible = main(checkpoint_prefix, (20,20,20))
upper_feasible = np.array([np.array(t) for t in dict.fromkeys(map(tuple, upper_feasible))])
obj_final = objective_final_array[:,:,0]
#%%
new_f1 = np.linspace(15, 30, 1000)
spline = CubicSpline(upper_feasible[:,0], upper_feasible[:,1])
new_f2 = spline(new_f1)

#ref_point = torch.tensor([18., spline(20.).item()], device=device, dtype=dtype)
ref_point = objective_final[8,:,0]
#ref_point = torch.tensor([28, 50], device=device, dtype=dtype)
print(ref_point.detach().numpy())

train_x = train_x_final[:,:,0]
train_y = objective_final[:,:,0]
#%% UCB for locating new points
from Bayesian_grid_search.Objective_Grid_Search import MultiObjectiveGridSearch

def Grid_optimise_and_acquire(model, reference_f, distance, evaluator, bounds, sigma_gate, value_gate, exit_counter,
                              exit_thresh, beta, k_cap=15):
    global target, post
    model.eval()

    k = 0
    while True:
        acqf = MultiObjectiveGridSearch(
            model=model,
            reference_f=reference_f,
            distance=distance,
            k=k,
            beta=beta,
            num_mc_samples=128
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
            global sigma
            post = model.posterior(candidate.unsqueeze(-2))
            sigma = post.variance.sqrt().squeeze()
            mean = post.mean.squeeze()

        target = reference_f - torch.tensor([0, k*distance])
        if (torch.sqrt(torch.mean(sigma ** 2)).item() < sigma_gate and
                torch.sqrt((mean - target)**2).item() < value_gate):
            print("progress")
            k += 1
            if k > k_cap:
                if exit_counter>exit_thresh:
                    return None, None, k
                else:
                    with torch.no_grad():
                        new_x = candidate.detach()
                        print(new_x.numpy())
                        obj = evaluator(new_x)
                        print("final_Y")
                        print(obj)
                        observed = obj + NOISE_SE * torch.randn_like(obj)
                        return new_x, observed, k
            continue
        else:
            break

    with torch.no_grad():
        new_x = candidate.detach()
        print(new_x.numpy())
        obj = evaluator(new_x)
        print("final_Y")
        print(obj)
        observed = obj + NOISE_SE * torch.randn_like(obj)

    return new_x, observed, k

def evaluate_candidates(model, reference_f, distance, evaluator, bounds, steps):
    final_X = np.ones((steps, 3))
    final_Y = np.ones((steps,2))
    for k in range(steps):
        print(k)
        acqf = MultiObjectiveGridSearch(
            model=model,
            reference_f=reference_f,
            distance=distance,
            k=k,
            beta=0.,
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

def Grid_optimiser(bounds, evaluator, train_X, train_obj, distance, feasibility_point, sigma_gate, value_gate, steps,
                   exit_thresh=2, dne_thresh=10, beta=None, verbose=True):
    '''

    :param bounds:
    :param evaluator: The evaluator function (ie the laser firing)
    :param train_X:
    :param train_obj:
    :param distance: grid spacing
    :param feasibility_point: The reference point that you aim for. The idea is that you give it the feasibility
    curve points and it follows the contour from.
    :param sigma_gate: This defines the level of certainty that is required to skip evaluating a point for a given
    k.
    :param value_gate: Same as above but it defines accepted deviation in the grid value
    :param steps: Number of values of k to iterate over
    :param exit_thresh: How many times you need to evaluate k over steps to ensure that you are interpolating rather
    than extrapolating
    :param dne_thresh: How many tries the searcher can have at finding k += 1 before it concludes the point doesn't
    exist
    :param beta: UCB beta parameter
    :param verbose:
    :return:
    '''
    global k_arr, observed_arr, observed
    mll, model = initialise_model(train_X, train_obj)
    iteration = 1
    exit_counter = 0
    print("hello world")
    k_arr=np.array([])
    observed_arr = np.array([[0.,0.]])
    trial_arr = np.array([[0.,0., 0.]])
    dne = 0
    k_prev = 0
    while True:
        print("iteration: " + str(iteration))
        t0 = time.monotonic()

        fit_gpytorch_mll(mll)

        new_X, observed, k = Grid_optimise_and_acquire(
            model=model,
            reference_f=feasibility_point,
            distance=distance,
            evaluator=evaluator,
            bounds=bounds,
            sigma_gate=sigma_gate,
            value_gate=value_gate,
            exit_counter=exit_counter,
            exit_thresh=exit_thresh,
            k_cap=steps,
            beta=beta,
        )
        k_arr = np.append(k_arr, k)

        if k > steps:
            '''
            If your value of k exceeds the number of steps that you want to take in one direction, then add
            one to the exit counter. When the counter exceeds the exit threshold, the algorithm finishes. The idea
            here is that it might evaluate k = 7 and then skip to k = 11 when the target was k = 10. You still
            might want it to evaluate k=11 to make sure that the GP interpolates k=10 rather than extrapolates.
            '''
            exit_counter += 1

        if k == k_prev:
            '''
            k_prev is the previous value of k. The idea is that if a point actually doesn't exist in output space,
            then the algorithm is going to get stuck trying to push k to access a value that doesn't exist.
            
            dne is the does not exist counter. Once it exceeds a user defined threshold, the algorithm stops.
            '''
            dne += 1
        else:
            dne = 0
        print(k)
        print(k_prev)
        print(dne)
        k_prev = k

        if dne > dne_thresh:
            print("k_prev = " + str(k_prev) + " does not exist. Firing all values up to k_prev")
            final_X, final_Y = evaluate_candidates(
                model=model,
                reference_f=feasibility_point,
                distance=distance,
                evaluator=evaluator,
                bounds=bounds,
                beta=beta,
                steps=k_prev,
            )
            return final_X, final_Y, observed_arr[1:], trial_arr[1:]


        if exit_counter > exit_thresh:
            print("entering final loop")
            final_X, final_Y = evaluate_candidates(
                model=model,
                reference_f=feasibility_point,
                distance=distance,
                evaluator=evaluator,
                bounds=bounds,
                beta=beta,
                steps=steps,
            )

            return final_X, final_Y, observed_arr[1:], trial_arr[1:]

        train_X = torch.cat((train_X, new_X))
        train_obj = torch.cat((train_obj, observed))
        observed_arr = np.concatenate((observed_arr, observed.detach().numpy()))
        trial_arr = np.concatenate((trial_arr, new_X.detach().numpy()))
        mll, model = initialise_model(train_X, train_obj)

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: k  = "
                f"({k:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

        iteration += 1

#%%
final_X, final_Y, train_obj, trial_arr = Grid_optimiser(bounds, fire, train_x, train_y, beta=-1,
                                  distance=0.5, dne_thresh=8,feasibility_point = ref_point,
                                             sigma_gate=0.5, value_gate=0.5, steps=50)

#%%
#plt.close('all')
plt.figure()
plt.plot(final_Y[:,0], final_Y[:,1], 'xk', label="final guess")
plt.plot(train_obj[:,0], train_obj[:,1], 'xr', label="learning points")
ref = ref_point.detach().numpy()
plt.plot(ref[0], ref[1], 'bo',label="reference point")
plt.xlabel("objective1 (arb.)")
plt.ylabel("objective2 (arb.)")
plt.legend()
plt.figure()
plt.plot(k_arr)
plt.xlabel("iteration")
plt.ylabel("k")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(final_X[:,0], final_X[:,1], final_X[:,2], 'xk', label="guessed points")
ax.plot(trial_arr[:,0], trial_arr[:,1], trial_arr[:,2], 'xr', label="training points")
plt.legend()
ax.set_xlabel('focus (mm)')
ax.set_ylabel('astig0 (arb.)')
ax.set_zlabel('astig45 (arb.)')

#%% EI for locating new points
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)

def obj_callable(target):
    def objective(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
        return -(Z[..., -1] - target[...,-1])**2
    return objective

def constraint_callable(target, value_gate=0.1):
    '''
        The constraint is that the first dimensions of the output do not change with respect to the input
        with a tolerated error of some gate. IE, you will get the optimiser to return this value for a candidate
        and ensure it lies within a certain range
    '''
    def c1(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
        global hello2
        hello = torch.sqrt((Z[..., :-1] - target[...,:-1]).pow(2).sum(dim=-1)) - value_gate
        dist = hello.mean(dim=0, keepdim=True)
        hello2 = torch.sqrt(torch.mean((Z[..., :-1] - target[...,:-1]) ** 2)) - value_gate
        return dist
    return c1

def reference_callable(Z:torch.Tensor, X:Optional[torch.Tensor] = None):
    return Z[...,-1]

def c_gate(model, candidate, target, value_gate=0.1):
    global test
    test = target
    print("hello")
    with torch.no_grad():
        # This is only going to work on 2d output in it's current form
        post = model.posterior(candidate.unsqueeze(-2))
        sigma = post.variance.sqrt().squeeze()
        print('1')
        mean = post.mean.squeeze()
        normal_dist = torch.distributions.normal.Normal(mean[0], sigma[0])
        print('2')
        range = torch.tensor([-target[0].item() + value_gate, value_gate + target[0].item()], device=device, dtype=dtype)
        probability = normal_dist.cdf(range[1]) - normal_dist.cdf(range[0])

    return probability

def s_gate(model, candidate, target, value_gate=0.1):
    with torch.no_grad():
        post = model.posterior(candidate.unsqueeze(-2))
        sigma = post.variance.sqrt().squeeze()
        mean = post.mean.squeeze()
        normal_dist = torch.distributions.normal.Normal(mean[1], sigma[1])
        range = torch.tensor([-target[1].item() + value_gate, value_gate + target[1].item()], device=device, dtype=dtype)
        probability = normal_dist.cdf(range[1]) - normal_dist.cdf(range[0])

    return probability

def EI_Grid_optimise_and_acquire(model, k_checkpoint, sampler, reference_f, distance, evaluator, bounds,
                                 value_gate, constraint_gate, success_gate, exit_counter,exit_thresh, k_cap=15):
    model.eval()

    k = k_checkpoint
    candidate_prev =  None
    while True:
        target = reference_f - torch.tensor([0, k * distance])
        print("target")
        print(target.detach().numpy())
        objective = GenericMCObjective(obj_callable(target))
        constraint = constraint_callable(target)
        acqf = qLogExpectedImprovement(
            model=model,
            best_f= torch.tensor([0.0], device=device, dtype=dtype),#target[...,-1],
            sampler=sampler,
            objective=objective,
            constraints=[constraint],
        )

        candidate, EI = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )

        probability_constraint = c_gate(model, candidate, target, value_gate=value_gate)
        probability_success = s_gate(model, candidate, target, value_gate=value_gate)

        print("probability_constraint")
        print(probability_constraint.detach().numpy())
        print("probability_success")
        print(probability_success.detach().numpy())
        print("EI")
        print(EI.detach().numpy())

        #EI goes to 0 when in exploration mode. We are using logEI hence the negative condition
        if (EI.detach().numpy() <= -10 and probability_constraint.detach().numpy()>constraint_gate
                and probability_success.detach().numpy()>success_gate):
            print("progress, k = :" + str(k))
            k += 1
            candidate_prev = candidate.detach()
            if k > k_cap:
                if exit_counter > exit_thresh:
                    return None, None, k
                else:
                    with torch.no_grad():
                        new_x = candidate.detach()
                        obj = evaluator(new_x)
                        #print("final_Y")
                        #print(obj)
                        observed = obj + NOISE_SE * torch.randn_like(obj)

                        if torch.sqrt((observed - target).pow(2).sum(dim=-1)).item() < value_gate:
                            print("///")
                            print("checkpoint set")
                            print("///")
                            k_checkpoint = k
                        return new_x, observed, k, k_checkpoint
            continue
        else:
            break

    with torch.no_grad():
        if candidate_prev is not None:
            print("firing prev")
            new_x=candidate_prev
            k -= 1
            target = reference_f - torch.tensor([0, k * distance])
        else:
            new_x = candidate.detach()
        obj = evaluator(new_x)
        #print("final_Y")
        #print(obj)
        observed = obj + NOISE_SE * torch.randn_like(obj)
        diff = torch.sqrt((observed - target).pow(2).sum(dim=-1)).item()
        print("difference")
        print(diff)
        if diff < value_gate:
            print("///")
            print("checkpoint set")
            print("///")
            k_checkpoint = k

    return new_x, observed, k, k_checkpoint

def EI_Grid_optimiser(s_dict, bounds, evaluator, train_X, train_obj, distance, feasibility_point, value_gate, constraint_gate,
                      success_gate, steps, exit_thresh=2, dne_thresh=10, verbose=True):
    '''

    :param bounds:
    :param evaluator: The evaluator function (ie the laser firing)
    :param train_X:
    :param train_obj:
    :param distance: grid spacing
    :param feasibility_point: The reference point that you aim for. The idea is that you give it the feasibility
    curve points and it follows the contour from.
    :param sigma_gate: This defines the level of certainty that is required to skip evaluating a point for a given
    k.
    :param value_gate: Same as above but it defines accepted deviation in the grid value
    :param steps: Number of values of k to iterate over
    :param exit_thresh: How many times you need to evaluate k over steps to ensure that you are interpolating rather
    than extrapolating
    :param dne_thresh: How many tries the searcher can have at finding k += 1 before it concludes the point doesn't
    exist
    :param beta: UCB beta parameter
    :param verbose:
    :return:
    '''
    global k_arr, observed_arr, observed
    mll, model = initialise_model(train_X, train_obj, state_dict=s_dict)
    iteration = 1
    exit_counter = 0
    print("hello world")
    k_arr = np.array([])
    observed_arr = np.array([[0., 0.]])
    trial_arr = np.array([[0., 0., 0.]])
    dne = 0
    k_prev = 0
    k_checkpoint=0
    while True:
        print("iteration: " + str(iteration))
        t0 = time.monotonic()

        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        new_X, observed, k, k_checkpoint = EI_Grid_optimise_and_acquire(
            model=model,
            sampler=sampler,
            k_checkpoint=k_checkpoint,
            reference_f=feasibility_point,
            distance=distance,
            evaluator=evaluator,
            bounds=bounds,
            value_gate=value_gate,
            constraint_gate=constraint_gate,
            success_gate = success_gate,
            exit_counter=exit_counter,
            exit_thresh=exit_thresh,
            k_cap=steps,
        )
        k_arr = np.append(k_arr, k)

        if k > steps:
            '''
            If your value of k exceeds the number of steps that you want to take in one direction, then add
            one to the exit counter. When the counter exceeds the exit threshold, the algorithm finishes. The idea
            here is that it might evaluate k = 7 and then skip to k = 11 when the target was k = 10. You still
            might want it to evaluate k=11 to make sure that the GP interpolates k=10 rather than extrapolates.
            '''
            exit_counter += 1

        if k == k_prev:
            '''
            k_prev is the previous value of k. The idea is that if a point actually doesn't exist in output space,
            then the algorithm is going to get stuck trying to push k to access a value that doesn't exist.

            dne is the does not exist counter. Once it exceeds a user defined threshold, the algorithm stops.
            '''
            dne += 1
        else:
            dne = 0
        print(k)
        print(k_prev)
        k_prev = k

        if dne > dne_thresh:
            print("k_prev = " + str(k_prev) + " does not exist. Firing all values up to k_prev")
            final_X, final_Y = evaluate_candidates(
                model=model,
                reference_f=feasibility_point,
                distance=distance,
                evaluator=evaluator,
                bounds=bounds,
                steps=k_prev,
            )
            return train_x, train_obj, final_X, final_Y, observed_arr[1:], trial_arr[1:]

        if exit_counter > exit_thresh:
            print("entering final loop")
            final_X, final_Y = evaluate_candidates(
                model=model,
                reference_f=feasibility_point,
                distance=distance,
                evaluator=evaluator,
                bounds=bounds,
                steps=steps,
            )

            return train_x, train_obj, final_X, final_Y, observed_arr[1:], trial_arr[1:]

        train_X = torch.cat((train_X, new_X))
        train_obj = torch.cat((train_obj, observed))
        observed_arr = np.concatenate((observed_arr, observed.detach().numpy()))
        trial_arr = np.concatenate((trial_arr, new_X.detach().numpy()))
        dict = model.state_dict()
        mll, model = initialise_model(train_X, train_obj, state_dict=dict)

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: k  = "
                f"({k:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

        iteration += 1

#%%
n=1
step = 1
n_steps = 5
for i in range(n):
    ref_point[0] += i*step
    X_total = np.ones((n, n_steps, 3))
    Y_total = np.ones((n, n_steps, 2))
    train_x_n = train_x
    train_y_n = train_y
    try:
        global f_X, f_Y
        print(train_x_n.shape)
        print(train_y_n.shape)
        new_train_x, new_train_obj, f_X, f_Y, train_obj_n, trial_arr = EI_Grid_optimiser(
            s_dict=state_dicts[0],
            bounds=bounds,
            evaluator=fire,
            train_X=train_x_n,
            train_obj=train_y_n,
            distance=0.1,
            dne_thresh=5,
            feasibility_point=ref_point,
            value_gate=0.2,
            constraint_gate=0.9,
            success_gate=0.5,
            steps=n_steps)

        train_x_n = new_train_x
        train_y_n = new_train_obj

        print(np.shape(f_X))
        print(f_X)
        print(np.shape(f_Y))

        X_total[i,:,:] = f_X
        Y_total[i,:,:] = f_Y
        print("///")
        print('***')
        print("success")
        print("***")
        print("///")

    except:
        print("iteration " + str(n) + " failed")

#%%
plt.close("all")
Y = Y_total
plt.close('all')
plt.figure()
for i in range(n):
    plt.plot(Y[i,:,0], Y[i,:,1], 'xk', label="final guess")
plt.xlabel("objective1 (arb.)", fontsize='16')
plt.ylabel("objective2 (arb.)", fontsize='16')
plt.legend(fontsize='16')
ref = ref_point.detach().numpy()
plt.plot(ref[0], ref[1], 'bo',label="reference point")
#%%
train_x, train_obj, final_X, final_Y, train_obj, trial_arr = EI_Grid_optimiser(
    s_dict=state_dicts[0],
    bounds=bounds,
    evaluator=fire,
    train_X=train_x,
    train_obj=train_y,
    distance=0.1,
    dne_thresh=5,
    feasibility_point = ref_point,
    value_gate=0.2,
    constraint_gate=0.6,
    success_gate=0.6,
    steps=200)

#%%
plt.close('all')
plt.figure(figsize=(10, 7))
plt.plot(train_obj[:,0], train_obj[:,1], '.r', ms=8, label="BCD training points")
plt.plot(final_Y[:,0], final_Y[:,1], '.k', ms=8, label="model output")
ref = ref_point.detach().numpy()
plt.plot(ref[0], ref[1], 'go', ms=8,label="reference point")
plt.xlabel("objective1 (arb.)", fontsize='18')
plt.ylabel("objective2 (arb.)", fontsize='18')


plt.plot(objective_final_array[:,0,0], objective_final_array[:,1,0], '.', ms=8, label="initial data")
plt.tick_params("both", labelsize=14)
plt.legend(fontsize='18')
plt.savefig("contour_descent_output.png")

#%%
plt.figure(figsize=(10,7))
plt.plot(k_arr)
plt.xlabel("iteration", fontsize='20')
plt.ylabel("$k$", fontsize="20")
plt.tick_params("both", labelsize=16)
plt.savefig("k_evolution.png")

#%%
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
ax.plot(final_X[:,0], final_X[:,1], final_X[:,2], 'xk', label="model output")
ax.plot(trial_arr[:,0], trial_arr[:,1], trial_arr[:,2], 'xr', label="BCD training points")
plt.legend(fontsize='16')
ax.set_xlabel('focus (mm)', fontsize='16')
ax.set_ylabel('astig0 (arb.)', fontsize='16')
ax.set_zlabel('astig45 (arb.)', fontsize='16')
plt.savefig("contour_descent_input.png")

#%%
def compute_rms_error(output, desired, eps=1e-8):
    rel_err = (output - desired) / (desired + eps)
    return 100.0 * np.sqrt(np.mean(rel_err ** 2)), np.std(rel_err)

start = ref_point.detach().numpy()[1]
final = final_Y[:,1]
step=-0.1
n=len(final)
target = start + step * np.arange(n)

plt.close("all")
plt.figure()
plt.plot(target)
plt.plot(final)

mean, std = compute_rms_error(target, final)
#%%
print(np.mean(target))
print(mean)
print(std)
plt.close("all")
#%%
np.save("final_X.npy", final_X)
np.save("final_Y.npy", final_Y)
np.save("trial_arr.npy", trial_arr)
np.save("k_arr.npy", k_arr)
np.save("ref.npy", ref)
np.save("train_obj.npy", train_obj)

import torch
print(torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import numpy as np
import matplotlib.pyplot as plt
from torchrbf import RBFInterpolator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.float32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
'''
dtype = torch.float32
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# --- Load data ---
obj1_np = np.load("vals.npy")   # shape (n,) or (n,1)
obj2_np = np.roll(obj1_np, 15) ** 1

points_np = np.load("vals2.npy")  # shape (n,3)

# --- Convert to torch on the right device/dtype ---
points = torch.from_numpy(points_np).to(device=device, dtype=dtype)

obj1 = torch.from_numpy(obj1_np).to(device=device, dtype=dtype)
obj2 = torch.from_numpy(obj2_np).to(device=device, dtype=dtype)

# --- Build interpolators (torch-native) ---
interp_obj1 = RBFInterpolator(points, obj1, device=device)
interp_obj2 = RBFInterpolator(points, obj2, device=device)

plt.close("all")

def visualise(points: torch.Tensor, interp, n_grid=20, title=""):
    # Make grid in torch
    x_grid = torch.linspace(points[:, 0].min(), points[:, 0].max(), n_grid, device=points.device, dtype=points.dtype)
    y_grid = torch.linspace(points[:, 1].min(), points[:, 1].max(), n_grid, device=points.device, dtype=points.dtype)
    z_grid = torch.linspace(points[:, 2].min(), points[:, 2].max(), n_grid, device=points.device, dtype=points.dtype)

    X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    query = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)  # (N,3)

    # Evaluate interpolator
    V = interp(query)  # expected (N,1) or (N,)

    # If torchrbf returns numpy, uncomment this:
    # V = torch.as_tensor(V, device=query.device, dtype=query.dtype)

    # Flatten and mask NaNs (torch-native)
    V = V.reshape(-1)
    mask = ~torch.isnan(V)

    q = query[mask]
    V = V[mask]

    # Move to CPU for matplotlib
    q_cpu = q.detach().cpu().numpy()
    V_cpu = V.detach().cpu().numpy()

    Xcf, Ycf, Zcf = q_cpu[:, 0], q_cpu[:, 1], q_cpu[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Xcf, Ycf, Zcf, c=V_cpu, cmap="viridis", s=10)
    cbar = fig.colorbar(sc, ax=ax, label="value (arb.)")

    ax.set_title(title or "3D interpolated field")
    ax.set_xlabel("focus (mm)")
    ax.set_ylabel("0 degree astigmatism (arb.)")
    ax.set_zlabel("45 degree astigmatism (arb.)")

    plt.show()

visualise(points, interp_obj1, title="Objective 1 (RBFInterpolated)")
visualise(points, interp_obj2, title="Objective 2 (RBFInterpolated)")


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
        [torch.min(points[:, 0]), torch.min(points[:, 1]), torch.min(points[:, 2])],
        [torch.max(points[:, 0]), torch.max(points[:, 1]), torch.max(points[:, 2])]
    ], device=device, dtype=dtype)

bounds_arr = bounds.cpu().detach().numpy()

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ref_point = torch.tensor([-600.0, -2000.0], device=device, dtype=dtype)

SMOKE_TEST = False

DIM = 3
NUM_RESTARTS = 5 if not SMOKE_TEST else 2
RAW_SAMPLES = 256 if not SMOKE_TEST else 32
N_BATCH = 20 if not SMOKE_TEST else 1
MC_SAMPLES = 64 if not SMOKE_TEST else 32
N_TRIALS = 1 if not SMOKE_TEST else 1
NUM_PARETO = 2 if SMOKE_TEST else 10
NUM_FANTASIES = 2 if SMOKE_TEST else 8

NOISE_SE = 0.0
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

def fire(x: torch.Tensor) -> torch.Tensor:
    # x: (N, 3)            # (N,)
    y1 = interp_obj1(x)            # (N,)
    y2 = interp_obj2(x)            # (N,)
    return torch.stack([y1, y2], dim=-1).to(dtype=dtype)

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


# Different acquisition functions can be used for training, they each have their own problems
#Finds the pareto front efficiently but we want the feasibility curve
def optimise_and_acquire_EHVI(model, mll, X_all, sampler, bounds, evaluator, Minimise=False):
    model.train()
    model.likelihood.train()
    fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()

    X_all = X_all.to(device=device, dtype=dtype)

    with torch.no_grad():
        pred = model.posterior(X_all).mean  # on GPU

    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)

    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    candidates, EHVI = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        num_restarts=NUM_RESTARTS,
        q=1,
        sequential=True,
        raw_samples=RAW_SAMPLES,
        options={"maxiter": 100, "batch_limit": 5},
    )

    ehvi_log = EHVI.item() if torch.is_tensor(EHVI) else float(EHVI)
    ehvi = float(torch.exp(torch.tensor(ehvi_log)))  # or np.exp(ehvi_log)

    new_x = candidates.detach()        # GPU
    obs = evaluator(new_x)            # GPU (torch spline)
    if Minimise:
        obs = -obs
    observed = obs + NOISE_SE * torch.randn_like(obs)

    return new_x, observed, ehvi


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
            m.covar_module.base_kernel.lengthscale.cpu().detach().numpy().squeeze(0)
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

objective_final_array = objective_final.detach().cpu().numpy()
print(np.shape(objective_final_array))
plt.figure(figsize=(10,7))
for i in range(N_TRIALS):
    plt.plot(objective_final_array[:,0,i], objective_final_array[:,1,i], '*', label="iteration " + str(i))
    plt.xlabel("objective1 (arb.)", fontsize=20)
    plt.ylabel("objective2 (arb.)", fontsize=20)
    plt.tick_params("both", labelsize=12)
plt.legend()
EHVI_array = EHVI_final.detach().cpu().numpy()

plt.figure()
for i in range(N_TRIALS):
    plt.plot(EHVI_array[:,i])
    plt.xlabel("iteration")
    plt.ylabel("acquisition_value")

train_x = train_x_final[:,:,0]
train_y = objective_final[:,:,0]
ref_point = objective_final[2,:,0]
ref_x = train_x[2,:]
print(ref_point.detach().cpu().numpy())

from Objective_Grid_Search import MultiObjectiveGridSearch
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)

def objective2(Z: torch.Tensor, target, X: Optional[torch.Tensor] = None):
    return -torch.abs((Z[..., -1] - target[...,-1]))
    #return -(Z[..., -1] - target[...,-1])**2
  

def obj_callable(target):
    def objective(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
        return -torch.abs((Z[..., -1] - target[...,-1]))
        #return -(Z[..., -1] - target[...,-1])**2
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
        return hello2
    return c1

def reference_callable(Z:torch.Tensor, X:Optional[torch.Tensor] = None):
    return Z[...,-1]

def c_gate(model, candidate, target, value_gate=0.1):
    global test
    test = target
    with torch.no_grad():
        # This is only going to work on 2d output in it's current form
        post = model.posterior(candidate.unsqueeze(-2))
        sigma = post.variance.sqrt().squeeze()
        mean = post.mean.squeeze()
        normal_dist = torch.distributions.normal.Normal(mean[0], sigma[0])
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
        evaluated = evaluator(candidate)
        new_x = candidate.detach().cpu()
        final_X[k, :] = new_x.numpy()
        final_Y[k, :] = evaluated.detach().cpu().numpy()
    return final_X, final_Y

from botorch.generation import gen_candidates_torch
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.generation.gen import get_best_candidates

def gpu_mem(tag=""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserv = torch.cuda.memory_reserved(device) / 1024**2
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    max_reserv = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"[{tag}] allocated={alloc:.1f} MiB | reserved={reserv:.1f} MiB | "
          f"max_alloc={max_alloc:.1f} MiB | max_reserv={max_reserv:.1f} MiB")


def EI_Grid_optimise_and_acquire(
    model, train_obj, k_checkpoint, sampler, reference_f, distance, evaluator, bounds,
    value_gate, constraint_gate, success_gate, exit_counter, exit_thresh, k_cap=15
):
    model.eval()

    # GPU constant to avoid recreating tensors
    e2 = torch.tensor([0.0, 1.0], device=device, dtype=dtype)

    k = k_checkpoint
    candidate_prev = None

    while True:
        print("new loop")
        target = reference_f - (float(k) * float(distance)) * e2  
        objective = GenericMCObjective(obj_callable(target))
        constraint = constraint_callable(target)

        best_f_tensor = objective2(train_obj, target)
        best_f = torch.max(best_f_tensor)
        print("best_f")
        print(best_f.item())
        acqf = qLogExpectedImprovement(
            model=model,
            best_f=best_f, #target[..., -1],
            sampler=sampler,
            objective=objective,
            constraints=[constraint],
        )


        
        #Uses slow scipy sequential optimisation
        '''
        candidate, EI = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 256, "init_batch_limit": 256, "maxiter": 200},
        )
        '''

        #Extremely fast batchable torch optimisation
        init_conds = gen_batch_initial_conditions(
            acq_function=acqf,
            bounds=bounds,                 # shape (2, d), torch tensor
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,       # e.g. 512–4096
            options={
                "batch_limit": NUM_RESTARTS,  # optional, but consistent
            },
        )
        candidates, acq_values = gen_candidates_torch(
            initial_conditions=init_conds,
            acquisition_function=acqf,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            optimizer=torch.optim.Adam, # or SGD
            options = {
                "optimizer_options": {"lr": 0.01},
                "stopping_criterion_options": {"maxiter": 200},
            }
        )

        candidate = get_best_candidates(candidates, acq_values).detach()
        EI = acq_values.max().detach()
        
        
        probability_constraint = c_gate(model, candidate, target, value_gate=value_gate)
        probability_success = s_gate(model, candidate, target, value_gate=value_gate)

        ei_val = EI.item()
        pc = probability_constraint.item()
        print(probability_constraint)
        print(ei_val)
        ps = probability_success.item()

        if (ei_val <= -10.0 and pc > constraint_gate and ps > success_gate):
            k += 1
            candidate_prev = candidate#.detach()
            if k > k_cap:
                if exit_counter > exit_thresh:
                    return None, None, k
                else:
                    with torch.no_grad():
                        obj = evaluator(candidate)  
                        observed = obj + NOISE_SE * torch.randn_like(obj)

                        if torch.sqrt((observed - target).pow(2).sum(dim=-1)).item() < value_gate:
                            k_checkpoint = k
                        return candidate, observed, k, k_checkpoint
            continue
        else:
            break

    with torch.no_grad():
        if candidate_prev is not None:
            candidate = candidate_prev
            k -= 1
            target = reference_f - (float(k) * float(distance)) * e2

        obj = evaluator(candidate)
        observed = obj + NOISE_SE * torch.randn_like(obj)

        diff = torch.sqrt((observed - target).pow(2).sum(dim=-1)).item()
        if diff < value_gate:
            k_checkpoint = k

    return candidate, observed, k, k_checkpoint

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

    post = model.posterior(ref_x.unsqueeze(-2))      # X: shape [n, d]
    mean = post.mean               # shape [n, m]  (m = number of objectives)
    var  = post.variance
    print("train y:", ref_point.detach().cpu().numpy(), "mean:", mean.detach().cpu().numpy(), "std:", var.detach().cpu().numpy())
    
    iteration = 1
    exit_counter = 0
    observed_log = []
    trial_log = []
    k_log = []
    dne = 0
    k_prev = 1
    k_checkpoint=1
    while True:
        print("iteration: " + str(iteration))
        t0 = time.monotonic()

        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        new_X, observed, k, k_checkpoint = EI_Grid_optimise_and_acquire(
            model=model,
            train_obj=train_obj,
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
        k_log.append(k)


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
            observed_arr = torch.cat(observed_log, dim=0).cpu().numpy()
            trial_arr = torch.cat(trial_log, dim=0).cpu().numpy()
            k_arr = np.array(k_log)
            return train_X, train_obj, final_X, final_Y, observed_arr[1:], trial_arr[1:]

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

            observed_arr = torch.cat(observed_log, dim=0).cpu().numpy()
            trial_arr = torch.cat(trial_log, dim=0).cpu().numpy()
            k_arr = np.array(k_log)
            return train_X, train_obj, final_X, final_Y, observed_arr[1:], trial_arr[1:]

        train_X = torch.cat((train_X, new_X))
        train_obj = torch.cat((train_obj, observed))
        observed_log.append(observed.detach())
        trial_log.append(new_X.detach())
        model.train()
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
SMOKE_TEST = True

NUM_RESTARTS = 100 if not SMOKE_TEST else 5
RAW_SAMPLES = 1024 if not SMOKE_TEST else 512
MC_SAMPLES = 216 if not SMOKE_TEST else 216
N_TRIALS = 1 if not SMOKE_TEST else 1

tx = train_x#.detach().requires_grad_(False)
ty = train_y#.detach().requires_grad_(False)
print(tx.shape)
print(ty.shape)
train_X, train_Obj, final_X, final_Y, train_obj, trial_arr = EI_Grid_optimiser(
    s_dict=state_dicts[0],
    bounds=bounds,
    evaluator=fire,
    train_X=tx,
    train_obj=ty,
    distance=0.5,
    dne_thresh=20,
    feasibility_point = ref_point,
    value_gate=0.2,
    constraint_gate=0.8,
    success_gate=0.0,
    steps=5)
print("done!")

plt.figure()
plt.plot(train_obj[:,0], train_obj[:,1], 'xr', label="learning points")
plt.plot(final_Y[:,0], final_Y[:,1], 'xk', label="final guess")
ref = ref_point.detach().cpu().numpy()
plt.plot(ref[0], ref[1], 'bo',label="reference point")
plt.xlabel("objective1 (arb.)", fontsize='16')
plt.ylabel("objective2 (arb.)", fontsize='16')
plt.legend(fontsize='16')
plt.figure()
plt.plot(k_arr)
plt.xlabel("iteration", fontsize='16')
plt.ylabel("k", fontsize="16")

fig = plt.figure()
ax = plt.axes(projection='3d')
ref = ref_x.detach().cpu().numpy()
ax.plot(ref[0], ref[1], ref[2], 'bo', label="reference_point")
ax.plot(final_X[:,0], final_X[:,1], final_X[:,2], 'xk', label="guessed points")
ax.plot(trial_arr[:,0], trial_arr[:,1], trial_arr[:,2], 'xr', label="training points")
plt.legend(fontsize='16')
ax.set_xlabel('focus (mm)', fontsize='16')
ax.set_ylabel('astig0 (arb.)', fontsize='16')
ax.set_zlabel('astig45 (arb.)', fontsize='16')
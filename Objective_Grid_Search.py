'''
In this script, I write my first attempt at my Objective Grid Search acquisition function. I will try and use the
same structure as the botorch EI acquisition function. This means that I will use AnalyticAcquisitionFunction
structure and will not (at least initially) bother with the mc q sampling.
'''

from botorch.models.model import Model
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor
import torch

from botorch.acquisition.objective import PosteriorTransform

class ObjectiveGridSearch(AnalyticAcquisitionFunction):
    r"""
    This acquisition function is designed to find the points in input space that will return
    a grid in objective space. In order to use this acquisition function, you first need to know the global
    maximum of your function (if single objective) or the curve defining feasible solutions  (if multi objective).

    This acquisition function starts at the global maximum (the model should know this value with high certainty),
    and then looks for grid points near the maximum. This process propagates outwards until the full grid is known.

    This is a first try at this type of acquisition function. There is almost certainly a more intelligent
    statistically informed way of sampling the grid. If the basic premise shows promise, then I will undertake
    this in the future.
    """
    def __init__(
            self,
            model: Model,
            best_f: float | Tensor,
            distance: float,
            k: float,
            beta: float = 1.,
            X_prev: Tensor = None,
            posterior_transform: PosteriorTransform | None = None,
    ):
        r"""Single-outcome objective grid search (analytic).

                Args:
                    model: A fitted single-outcome model.
                    best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                        the best function value observed so far (assumed noiseless).
                    distance: The grid spacing
                    sigma: The tolerable evaluation uncertainty
                    beta: Increase this to make the algorithm more exploitative. 1.0 is a good default.
                    posterior_transform: A PosteriorTransform. If using a multi-output model,
                        a PosteriorTransform that transforms the multi-output posterior into a
                        single-output posterior is required.
                """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.distance = distance
        self.beta = beta
        self.X_prev = X_prev
        self.k = k

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the candidate set X to determine the posterior mean and
        variance at every point. We try and incentivise selections that are in points
        of low model uncertainty.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Posterior mean and variance are computed for each point individually.

        Returns:
            A `(b1 x ... bk) x 1`-dim tensor of posterior means at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        target = (self.best_f - self.k * self.distance)
        acq_val = -((mean - target) ** 2) - self.beta*sigma
        return acq_val.squeeze(-1)



class MultiObjectiveGridSearch(AcquisitionFunction):
    r"""
    This acquisition function is designed to find the points in input space that will return
    a grid in objective space. In order to use this acquisition function, you first need to know the global
    maximum of your function (if single objective) or the curve defining feasible solutions  (if multi objective).

    This acquisition function starts at the global maximum (the model should know this value with high certainty),
    and then looks for grid points near the maximum. This process propagates outwards until the full grid is known.

    This is a first try at this type of acquisition function. There is almost certainly a more intelligent
    statistically informed way of sampling the grid. If the basic premise shows promise, then I will undertake
    this in the future.
    """
    def __init__(
            self,
            model: Model,
            reference_f: Tensor,
            distance: float|Tensor,
            k: float,
            beta:float = 1.,
            num_mc_samples=256):
        r"""Multi-outcome objective grid search (analytic).

                Args:
                    model: A fitted single-outcome model.
                    reference_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                        the point in output space from which you wish to begin your grid..
                    distance: The grid spacing
                    sigma: The tolerable evaluation uncertainty
                    beta: Positive beta penalises any exploration. Negative promotes it. -0.5 seems to be
                    a good default. 0 should be used for final evaluation.
                    posterior_transform: A PosteriorTransform. If using a multi-output model,
                        a PosteriorTransform that transforms the multi-output posterior into a
                        single-output posterior is required.
                """
        super().__init__(model)
        self.register_buffer("reference_f", torch.as_tensor(reference_f))
        self.distance = distance; self.k = k
        self.beta=beta
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))

    def forward(self, X: Tensor) -> Tensor:
        #the MC acquisiton function works by sampling N_mc possible candidates from the model for each X
        posterior = self.model.posterior(X)
        Y = self.sampler(posterior)

        ref = self.reference_f.view(*([1] * (Y.ndim - 1)), -1) #ensure correct shape
        target_last = ref[..., -1] - self.k * self.distance

        #squared error for the first n dimensions as we are following a contour
        se_first = (Y[..., :-1] - ref[..., :-1])**2
        #Loss for the first dimensions
        loss_first = se_first.sum(dim=-1)

        #Loss over the last dimension
        loss_last  = (Y[..., -1] - target_last)**2

        #expected loss is the sum of the two losses, and then the mean over all the MC samples associated with that X
        exp_loss   = (loss_first + loss_last).mean(dim=0)

        #uncertainty penalty
        unc_penalty = self.beta * Y.std(dim=0, unbiased=False).sum(dim=-1)

        #Negative because you want to minimise the loss
        output = -(exp_loss + unc_penalty)
        return output.squeeze(-1)

class MultiObjectiveExpectedGridSearch(AcquisitionFunction):
    '''
    The idea is similar to the MOGS acquisition function, except we will locate new points using an
    expected improvement rather than a UCB style. This is motivated by my struggles tuning hyperparameters
    and experience with EI methods converging much faster than UCB methods.
    '''
    def __init__(
            self,
            model: Model,
            reference_f: Tensor,
            distance: float|Tensor,
            k: float,
            beta:float = 1.,
            num_mc_samples=256):
        r"""Multi-outcome objective grid search (analytic).

                Args:
                    model: A fitted single-outcome model.
                    reference_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                        the point in output space from which you wish to begin your grid..
                    distance: The grid spacing
                    sigma: The tolerable evaluation uncertainty
                    beta: Positive beta penalises any exploration. Negative promotes it. -0.5 seems to be
                    a good default. 0 should be used for final evaluation.
                    posterior_transform: A PosteriorTransform. If using a multi-output model,
                        a PosteriorTransform that transforms the multi-output posterior into a
                        single-output posterior is required.
                """
        super().__init__(model)
        self.register_buffer("reference_f", torch.as_tensor(reference_f))
        self.distance = distance; self.k = k
        self.beta=beta
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))


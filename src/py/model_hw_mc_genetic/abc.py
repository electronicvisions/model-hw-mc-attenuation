'''
Approximated Bayesian Computuation

Collect functions which are related to perform approximated Bayesian
computation for an experiment. The algorithms include sequential neural
estimator algorithms (SNxE) as well as  Monte-Carlo ABC (MCABC).
'''
from enum import Enum, auto
from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import torch

from sbi.inference import SNPE, SNRE, MCABC, prepare_for_sbi, simulate_for_sbi


class Algorithm(Enum):
    SNPE = auto()
    # SNLE = auto() Not supported due to error during training
    SNRE = auto()
    MCABC = auto()


algorithms = {Algorithm.SNPE: SNPE,
              # Algorithm.SNLE: SNLE,
              Algorithm.SNRE: SNRE,
              Algorithm.MCABC: MCABC}


def perform_sequential_estimation(
    algorithm: Algorithm,
    proposal: torch.distributions.Distribution,
    simulator: Callable,
    target: np.ndarray, *,
    simulations: Optional[List[int]] = None,
    density_estimator: Optional[Union[str, Callable]] = None,
) -> Tuple[np.ndarray, List]:
    '''
    Perform a sequential neural posterior estimation.

    :param algorithm: Sequential algorithm to use.
    :param proposal: Proposal distribution for the first round of the
        algorithm. This is typically the prior distribution.
    :param simulator: Simulation functions which yields the observation given
        the parameters as an input.
    :param target: Target observation used to condition the posterior on
        starting from the second round.
    :param simulations: List of simulations in each round.
    :param density_estimator: Neural density estimator for the SNPE algorithm.
    :returns: Tuple of array with sample information and List of posteriors for
        each round. The array contains the proposal samples drawn in each
        round, the observations and the corresponding round.
    '''
    if simulations is None:
        simulations = [1000] * 10

    # Prepare inference
    simulator, proposal = prepare_for_sbi(simulator, proposal)

    if algorithm == Algorithm.SNRE:
        inference = algorithms[algorithm](prior=proposal)
    else:
        inference = algorithms[algorithm](prior=proposal,
                                          density_estimator=density_estimator)

    logs = []
    posteriors = []
    # Perform inference
    for curr_round, n_simulations in enumerate(simulations):
        theta, obs = simulate_for_sbi(simulator, proposal,
                                      num_simulations=n_simulations)

        logs.append(
            np.hstack([theta, obs, np.full((len(obs), 1), curr_round)]))

        if algorithm == Algorithm.SNPE:
            inference.append_simulations(theta, obs, proposal=proposal).train()
        else:
            inference.append_simulations(theta, obs).train()

        posterior = inference.build_posterior()
        posteriors.append(posterior)
        proposal = posterior.set_default_x(target)

    return np.concatenate(logs), posteriors


def perform_mcabc(
    prior: torch.distributions.Distribution,
    simulator: Callable,
    target: np.ndarray,
    eps: float,
    simulations: Optional[int] = 10000,
) -> np.ndarray:
    '''
    Perform Monte-Carlo ABC.

    :param prior: Probability distribution from which model parameters are
        drawn.
    :param simulator: Simulation functions which yields the observation given
        the parameters as an input.
    :param target: Target observation to which simulation results are compared.
    :param eps: Epsilon for which parameters are accepted.
    :param simulations: Number of simulations to perform.
    :returns: Array which contains the proposal samples drawn in each
        round and the observations.
    '''
    # MCABC wants a batch dimension
    def batch_simulator(params):
        return torch.Tensor(simulator(params[0]))[None, :]

    mcabc = MCABC(batch_simulator, prior)
    theta, summary = mcabc(torch.tensor(target),
                           simulations,
                           eps=eps,
                           return_summary=True)

    return np.hstack([theta, summary['x']])

'''
Experiment agnostic helper functions.

Provide for example functions which can perform a grid search of the parameter
space.
'''
from itertools import product
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from model_hw_mc_genetic.attenuation import Observation, fit_length_constant
from model_hw_mc_genetic.attenuation.base import Base as AttenuationExperiment


def grid_search(chain_experiment: AttenuationExperiment,
                g_leak: Tuple[float, float, float],
                g_icc: Tuple[float, float, float],
                ) -> pd.DataFrame:
    '''
    Perform a grid search of the axial and inter-compartment conductance for
    an AttenuationExperiment on BSS-2.

    :param chain_experiment: Attenuation Experiment.
    :param g_leak: Tuple (min, max, steps) used for the leak conductance
        during the grid search. The number of steps is converted to an integer.
    :param g_leak: Tuple (min, max, steps) used for the inter-compartment
        conductance during the grid search. The number of steps is converted to
        an integer.
    :returns: DataFrame filled with tested parameters and measured amplitudes.
    '''
    # Determine measurement points
    g_leak_icc = np.array(list(product(
        np.linspace(g_leak[0], g_leak[1], int(g_leak[-1])),
        np.linspace(g_icc[0], g_icc[1], int(g_icc[-1])))))

    # Create DataFrame, fill with parameter values and add meta data
    length = chain_experiment.length
    param_col = pd.MultiIndex.from_product([['parameters'],
                                            ['g_leak', 'g_icc']])
    cols = [f"A_{i}{j}" for i, j in product(range(length), range(length))]
    amp_col = pd.MultiIndex.from_product([['amplitudes'], cols])
    columns = pd.MultiIndex.from_tuples(list(param_col) + list(amp_col))

    result = pd.DataFrame(np.zeros([len(g_leak_icc), 2 + length**2]),
                          columns=columns)
    result['parameters'] = g_leak_icc

    result.attrs['length'] = length
    result.attrs['date'] = str(datetime.now())

    # Perform experiment
    for row in range(len(result)):
        g_leak, g_icc = result.loc[row, 'parameters']
        res = chain_experiment.measure_response(np.array([g_leak, g_icc]))
        result.loc[row, 'amplitudes'] = res.flatten()

    return result


def record_variations(experiment: AttenuationExperiment,
                      repetitions: int) -> pd.DataFrame:
    '''
    Repeat the same attenuation experiment several times and log the recorded
    PSP amplitudes in a DataFrame.

    :param experiment: Attenuation Experiment.
    :param repetitions: Number of times the experiment should be repeated.
    :returns: DataFrame with PSP amplitudes for each repetition.
    '''

    results = []
    for _ in range(repetitions):
        result = experiment.measure_response()
        results.append(result.flatten())

    result = pd.DataFrame(np.array(results))
    result.attrs['length'] = experiment.length
    result.attrs['date'] = str(datetime.now())

    return result


def get_bounds(data_frame: pd.DataFrame) -> AttenuationExperiment:
    '''
    Get the bounds for an exponential fit to the PSP amplitudes.

    The bounds vary due to the noise between experiments on BSS-2 and arbor.
    Supplying appropriate bounds is important since the recorded amplitudes
    are subject to modulations due to a finite chain length.

    :param data_frame: DataFrame with recorded amplitudes.
    :returns: Bounds to use for fitting the amplitudes with an exponential
        decay.
    '''
    attrs = data_frame.attrs
    if attrs['experiment'] == 'attenuation_bss':
        # - tau: The typical length constants are in the range of 1-8
        # - offset: We subtract the leak potential before fitting. The noise on
        #   the measured potentials has a standard deviation of around 3 LSB
        # - scaling factor: Restrict to the 10-bit range of the MADC
        bounds = ([0, -10, 0],
                  [20, 10, 1022])
        return bounds
    if attrs['experiment'] == 'attenuation_arbor':
        offset = attrs['noise'] * 3 if 'noise' in attrs else 1e-5
        # bounds for tau, offset, scaling factor
        return ([0, -offset, 0],
                [np.inf, offset, np.inf])

    raise RuntimeError(f'The experiment of type "{attrs["experiment"]}" is '
                       'not supported.')


def extract_observation(amplitudes: pd.DataFrame,
                        observation: Observation,
                        target_amplitudes: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    '''
    Extract the given observation from the given amplitudes.

    :param amplitudes: DataFrame with PSP amplitudes.
    :param observation: Type of observation to extract.
    :param atrget_amplitudes: One-dimensional array with atrget amplitudes.
    :returns: The given observation for each row in the DataFrame.
    '''
    if observation == Observation.AMPLITUDES:
        return amplitudes.values
    if observation == Observation.AMPLITUDE_00:
        return amplitudes.values[:, 0]
    if observation == Observation.AMPLITUDES_DISTANCE:
        if target_amplitudes is None:
            raise ValueError('You need to supply `target_amplitudes` for the '
                             f'observation {observation.name}.')
        return np.linalg.norm(amplitudes.values - target_amplitudes, axis=1)

    length = np.sqrt(amplitudes.shape[1]).astype(int)
    if observation == Observation.AMPLITUDES_FIRST:
        return amplitudes.values.reshape(-1, length, length)[:, :, 0]
    if observation == Observation.LENGTH_CONSTANT:
        bounds = get_bounds(amplitudes)

        def get_length_constant(row: pd.Series) -> float:
            return fit_length_constant(
                row.values.reshape(length, length)[:, 0], bounds=bounds)
        return amplitudes.apply(get_length_constant, axis=1).values

    raise ValueError(f'The observation "{observation}" is not supported.')


# Avoid BSS/arbor specific imports at the top-levl such that they can be
# executed individually without installing the dependencies for the other
# experiment.
# pylint: disable=import-outside-toplevel
def get_experiment(target: pd.DataFrame) -> AttenuationExperiment:
    '''
    Create an attenuation experiment with the same parameters which were used
    to record the target.

    :param target: DataFrame with recorded amplitudes which will be used to
        extract a target.
    :returns: AttenuationExperiment with the same parameterization as used for
        recording the target.
    '''
    attrs = target.attrs

    if attrs['experiment'] == 'attenuation_bss':
        from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment \
            as AttenuationBSS
        return AttenuationBSS(Path(attrs['calibration']),
                              length=attrs['length'],
                              input_neurons=attrs['input_neurons'],
                              input_weight=attrs['input_weight'])

    if attrs['experiment'] == 'attenuation_arbor':
        from model_hw_mc_genetic.attenuation.arbor import \
            AttenuationExperiment as AttenuationArbor
        return AttenuationArbor(length=attrs['length'])

    raise RuntimeError(f'The experiment of type "{attrs["experiment"]}" is '
                       'not supported.')

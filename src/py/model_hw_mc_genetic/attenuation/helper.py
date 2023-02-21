'''
Experiment agnostic helper functions.

Provide for example functions which can perform a grid search of the parameter
space.
'''
from itertools import product
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Sequence

import pandas as pd
import numpy as np

from model_hw_mc_genetic.helper import AttributeNotIdentical, \
    get_identical_attr
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


# Avoid BSS/arbor specific imports at the top-levl such that they can be
# executed individually without installing the dependencies for the other
# experiment.
# pylint: disable=import-outside-toplevel
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
        from model_hw_mc_genetic.attenuation.bss import integration_bounds
        return integration_bounds
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

    def get_length_constant(row: pd.Series) -> float:
        return fit_length_constant(
            row.values.reshape(length, length)[:, 0], bounds=bounds)

    if observation == Observation.LENGTH_CONSTANT:
        bounds = get_bounds(amplitudes)
        return amplitudes.apply(get_length_constant, axis=1).values
    if observation == Observation.LENGTH_AND_AMPLITUDE:
        bounds = get_bounds(amplitudes)
        return np.array(
            [amplitudes.apply(get_length_constant, axis=1).values,
             amplitudes.values.reshape(-1, length, length)[:, 0, 0]]).T

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
                              input_weight=attrs['input_weight'],
                              n_average=attrs['n_average'])

    if attrs['experiment'] == 'attenuation_arbor':
        from model_hw_mc_genetic.attenuation.arbor import \
            AttenuationExperiment as AttenuationArbor
        return AttenuationArbor(length=attrs['length'])

    raise RuntimeError(f'The experiment of type "{attrs["experiment"]}" is '
                       'not supported.')


def extract_original_parameters(dfs: Sequence[pd.DataFrame]) -> np.ndarray:
    '''
    Extract the original parameters from the given DataFrames with samples.

    Try to extract the parameters which were used to measure the target on
    which the posteriors were conditioned.


    :param dfs: DataFrames with samples drawn during the approximation or drawn
        from the posteriors. From their attributes the initial target is
        extracted and from the target the parameter.
    :raises RuntimeError: If the original parameters can not be extracted.
    '''
    try:
        target_dfs = [pd.read_pickle(df.attrs['target_file']) for df in dfs]
    except (KeyError, FileNotFoundError) as err:
        raise RuntimeError('Target can not be extracted for all '
                           'posterior files.') from err
    try:
        return get_identical_attr(target_dfs, 'parameters')
    except AttributeNotIdentical as err:
        raise RuntimeError('Posterior files do not have a common target.'
                           ) from err

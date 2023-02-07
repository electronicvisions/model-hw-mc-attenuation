'''
Experiment agnostic helper functions.

Provide for example functions which can perform a grid search of the parameter
space.
'''
from itertools import product
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np

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
    measurements = [str(idx) for idx in np.arange(length**2)]
    amp_col = pd.MultiIndex.from_product([['amplitudes'], measurements])
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

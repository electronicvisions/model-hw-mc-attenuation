'''
Helper functions which extract the type of experiment (BSS vs Arbor) from a
DataFrame to perform a given action.
In order to prevent cyclic imports, these functions can not be part of the
`helper` submodule.
'''
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from model_hw_mc_attenuation import Observation, fit_length_constant, \
    AttenuationExperiment


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
        from model_hw_mc_attenuation.bss import AttenuationExperiment \
            as AttenuationBSS
        # n_average was added later -> some saved experiments do not have
        # the n_average attribute -> set equal to 1
        return AttenuationBSS(Path(attrs['calibration']),
                              length=attrs['length'],
                              input_neurons=attrs['input_neurons'],
                              input_weight=attrs['input_weight'],
                              n_average=attrs.get('n_average', 1))

    if attrs['experiment'] == 'attenuation_arbor':
        from model_hw_mc_attenuation.arbor import \
            AttenuationExperiment as AttenuationArbor
        return AttenuationArbor(length=attrs['length'])

    raise RuntimeError(f'The experiment of type "{attrs["experiment"]}" is '
                       'not supported.')


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
        from model_hw_mc_attenuation.bss import integration_bounds
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

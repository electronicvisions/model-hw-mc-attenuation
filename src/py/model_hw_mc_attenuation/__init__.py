from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Union, Optional
import numpy as np
import neo
from scipy.optimize import curve_fit
import quantities as pq
import torch


class Observation(Enum):
    AMPLITUDES = auto()  # all amplitudes
    AMPLITUDES_FIRST = auto()  # amplitudes in the first compartment
    # length constant of exponential fit to amplitudes_first
    LENGTH_CONSTANT = auto()
    AMPLITUDE_00 = auto()  # response in first comp. to input to first comp.
    AMPLITUDES_DISTANCE = auto()  # Euclidean distance between two amplitudes
    # length constant of exponential fit to amplitudes_first + amplitude in
    # first compartment as a result from an input to the first compartment
    LENGTH_AND_AMPLITUDE = auto()


def extract_psp_heights(
        traces: List[neo.IrregularlySampledSignal]) -> np.ndarray:
    '''
    Extract the PSP height from membrane recordings.

    :param traces: Membrane recording for each compartment in the chain.
    :return: Height of PSP (difference between maximum and baseline).
        The heights are calculated for each compartment and for each input
        spike. The outer dimension iterates over compartments, the inner
        over input spikes.
    '''
    heights = []
    for sig in traces:
        heights_comp = []
        spike_times = sig.annotations['input_spikes'].rescale(pq.ms)

        start_stop = np.concatenate([
            sig.t_start.rescale(pq.ms)[np.newaxis],
            spike_times[:-1] + np.diff(spike_times) / 2,
            sig.t_stop.rescale(pq.ms)[np.newaxis]]) * pq.ms

        for n_spike, (start, stop) in enumerate(zip(start_stop[:-1],
                                                    start_stop[1:])):
            cut_sig = sig.time_slice(start, stop)
            # use membrane voltage before first spike as baseline
            baseline = cut_sig.time_slice(cut_sig.t_start,
                                          spike_times[n_spike]).mean()
            heights_comp.append(cut_sig.max() - baseline)
        heights.append(heights_comp)
    return np.asarray(heights)


def exponential_decay(location: Union[float, np.ndarray],
                      tau: Union[float, np.ndarray],
                      offset: Union[float, np.ndarray],
                      scaling_factor: Union[float, np.ndarray]
                      ) -> Union[float, np.ndarray]:
    '''
    Calculate an expoenential decay.

    :param location: Free variable.
    :param tau: Decay constant.
    :param offset: Constant y offset.
    :param scaling factor: Amplitude of the decay.
    :returns: scaling_factor * exp( -location / tau ) + offset.
    '''
    return scaling_factor * np.exp(- location / tau) + offset


def fit_exponential(heights: np.ndarray, **kwargs) -> float:
    '''
    Fit an exponential decay to the PSP height in the given array.

    The keyword arguments are passed to :func:`curve_fit`.

    If the initial guess of the parameters (p0) are not passed as a keyword
    argument, they are estimated from the given heights. This estimate
    assumes that the offset of the exponential is zero.

    Furthermore, the bounds (bounds) are set as broad as possible if not
    supplied as a keyword argument.

    :param heights: PSP heights to which an exponential should be fitted.
    :return: All fit parameters of the exponential fit (tau, offset,
        scaling_factor).
    '''
    compartments = np.arange(len(heights))

    # assume zero offset, i.e. y = A * exp( -x / tau)
    guessed_tau = 1 if heights[1] == 0 else 1 / np.log(heights[0] / heights[1])

    # order of variables in fit function:  tau, offset, scaling factor
    default_kwargs = {}
    default_kwargs['p0'] = [guessed_tau, 0, heights[0]]
    default_kwargs['bounds'] = ([0, -np.inf, 0],
                                [np.inf, np.inf, np.inf])
    default_kwargs.update(kwargs)

    popt = curve_fit(exponential_decay, compartments, heights,
                     **default_kwargs)[0]

    return popt


def fit_length_constant(heights: np.ndarray, **kwargs) -> float:
    '''
    Fit an exponential decay to the PSP height in the given array.

    The keyword arguments are passed to :func:`fit_exponential`.

    :param heights: PSP heights to which an exponential should be fitted.
    :return: Length constant (exponential fit to PSP height as function of
        compartment)
    '''
    return fit_exponential(heights, **kwargs)[0]


class AttenuationExperiment(ABC):
    '''
    Experiment which measures the attenuation of a PSP or a constant current
    in a chain of compartments.

    :ivar length: Number of compartments in the chain.
    '''

    def __init__(self, length: int) -> None:
        self.length = length

    @abstractmethod
    def set_parameters_individual(self, params: torch.Tensor):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Array with leak conductance and inter-compartment
            conductance for each compartment individually. The leak conductance
            is be expected to be in the first entries (one entry for each
            compartment) followed by the inter-compartment conductances.
        '''
        raise NotImplementedError

    @abstractmethod
    def set_parameters_global(self, params: np.ndarray):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Global values of the leak conductance and
            inter-compartment conductance to set (one value for each parameter
            is shared for all compartments).
        '''
        raise NotImplementedError

    def set_parameters(self, parameters: np.ndarray) -> None:
        '''
        Set leak and inter-compartment conductance.

        :parameters: Parameters of the leak and inter compartment conductance
            to set. The parameters can be provided globally for all
            compartments (parameters has length 2) or for each compartment
            individually (parameters has length 'chain length - 1').
            If the parameters are set individually the leak conductance
            is be expected to be in the first entries (one entry for each
            compartment) followed by the inter-compartment conductances.
        '''
        if parameters is not None and len(parameters) == 2:
            self.set_parameters_global(parameters)
        elif parameters is not None:
            self.set_parameters_individual(parameters)

    @abstractmethod
    def measure_response(self, parameters: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        '''
        Measure the PSP heights or the deflection due to a constant current.

        :parameters: Parameters of the leak and inter compartment conductance
            to set. The parameters can be provided globally for all
            compartments (parameters has length 2) or for each compartment
            individually (parameters has length 'chain length - 1').
        :return: PSP heights or deflection in the different compartments. The
            first row contains the response in the first compartment, the
            second the response in the second and so on. The different columns
            are the responses to different input sites.
        '''
        raise NotImplementedError

    @property
    @abstractmethod
    def default_limits(self) -> np.ndarray:
        '''
        Return default limits of the parameters of this experiment.

        :return: Limits of the experiment parameters. The different rows
            represent different parameters, the left column the lower limit
            and the right column the upper limit.
        '''
        raise NotImplementedError

    @property
    def default_parameters(self) -> np.ndarray:
        '''
        Return default parameters of this experiment.

        :return: Parameters at the center of the experiment limits.
        '''
        return self.default_limits.mean(1)

    def expand_parameters(self, parameters: Optional[np.ndarray] = None
                          ) -> np.ndarray:
        '''
        Expand the given parameters to individual parameters or return default
        values.

        :parameters: Parameters of the leak and inter compartment conductance
            to expand if needed. If no parameters are supplied the default
            parameters are returned.
        :returns: Parameters for each compartment/inter-compartment connection
            individually.
        '''
        if parameters is not None:
            if len(parameters) == 2 * self.length - 1:
                return parameters
            if len(parameters) == 2:
                return np.repeat(parameters, self.length)[:-1]
            raise ValueError('The length of the supplied parameters does not '
                             'match the expected length of 2 or '
                             f'{self.length}.')

        # return default values
        return self.default_parameters

    def parameter_names(self, global_names: bool = False) -> List[str]:
        '''
        Get the names of the parameters which can be configured for the
        experiment.

        The parameters are the leak and inter-compartment conductance. The
        number of parameters depends on the length of the chain.

        :param global_names: Leak and inter-compartment conductances are set
            globally. Do return the global parameter names.
        :returns: Names for all parameters of the experiment.
        '''
        return parameter_names(None if global_names else self.length)


def parameter_names(length: Optional[int] = None) -> List[str]:
    '''
    Get the names of the parameters which can be configured for an attenuation
    experiment.

    The parameters are the leak and inter-compartment conductance. The
    number of parameters depends on the length of the chain.

    :param length: Length of the chain. If not set the global parameter names
        (leak and inter-compartment conductance) are returned.
    :returns: Names for all parameters of the experiment.
    '''
    parameter_base_names = ['g_leak', 'g_icc']

    # Global evaluation
    if length is None:
        return parameter_base_names

    parameters = [f'{parameter_base_names[0]}_{comp}' for comp in
                  range(length)]
    parameters.extend(
        [f'{parameter_base_names[1]}_{comp}_{comp + 1}' for comp in
         range(length - 1)])

    return parameters

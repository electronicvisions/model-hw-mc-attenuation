from enum import Enum, auto
from typing import List
import numpy as np
import neo
from scipy.optimize import curve_fit
import quantities as pq


class Observation(Enum):
    AMPLITUDES = auto()  # all amplitudes
    AMPLITUDES_FIRST = auto()  # amplitudes in the first compartment
    # length constant of exponential fit to amplitudes_first
    LENGTH_CONSTANT = auto()
    AMPLITUDE_00 = auto()  # response in first comp. to input to first comp.
    AMPLITUDES_DISTANCE = auto()  # Euclidean distance between two amplitudes


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

    def fitfunc(location, tau, offset, scaling_factor):
        return scaling_factor * np.exp(- location / tau) + offset

    # assume zero offset, i.e. y = A * exp( -x / tau)
    guessed_tau = 1 if heights[1] == 0 else 1 / np.log(heights[0] / heights[1])

    # order of variables in fit function:  tau, offset, scaling factor
    default_kwargs = {}
    default_kwargs['p0'] = [guessed_tau, 0, heights[0]]
    default_kwargs['bounds'] = ([0, -np.inf, 0],
                                [np.inf, np.inf, np.inf])
    default_kwargs.update(kwargs)

    popt = curve_fit(fitfunc, compartments, heights, **default_kwargs)[0]

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

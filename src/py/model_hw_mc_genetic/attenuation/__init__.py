from typing import List
import numpy as np
import neo
from scipy.optimize import curve_fit
import quantities as pq


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


def fit_exponential(heights: np.ndarray) -> float:
    '''
    Fit an exponential decay to the PSP height in the given array.

    :param heights: PSP heights to which an exponential should be fitted.
    :return: All fit parameters of the exponential fit (tau, offset,
        scaling_factor).
    '''
    compartments = np.arange(len(heights))

    def fitfunc(location, tau, offset, scaling_factor):
        return scaling_factor * np.exp(- location / tau) + offset

    # initial guess for fit parameters
    guessed_tau = np.argmin(np.abs((heights - heights[-1])
                                   - (heights[0] - heights[-1]) / np.e))
    p_0 = {'tau': guessed_tau, 'offset': heights[-1],
           'scaling_factor': heights[0] - heights[-1]}
    bounds = {'tau': [0, np.inf],
              'offset': [-np.inf, np.inf],
              'scaling_factor': [0, np.inf]}

    popt = curve_fit(
        fitfunc,
        compartments,
        heights,
        p0=[p_0['tau'], p_0['offset'], p_0['scaling_factor']],
        bounds=([bounds['tau'][0], bounds['offset'][0],
                 bounds['scaling_factor'][0]],
                [bounds['tau'][1], bounds['offset'][1],
                 bounds['scaling_factor'][1]]))[0]

    return popt


def fit_length_constant(heights: np.ndarray) -> float:
    '''
    Fit an exponential decay to the PSP height in the given array.

    :param heights: PSP heights to which an exponential should be fitted.
    :return: Length constant (exponential fit to PSP height as function of
        compartment)
    '''
    return fit_exponential(heights)[0]

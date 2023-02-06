from typing import List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import neo
import torch
from scipy.optimize import curve_fit
import quantities as pq

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

from model_hw_si_nsc_dendrites.helper import get_license_and_chip

from model_hw_si_nsc_dendrites.compartment_chain import CompartmentChain
from model_hw_si_nsc_dendrites.helper import set_compartment_conductance


class AttenuationExperiment:
    '''
    Experiment which measures the attenuation of a PSP as it travels along a
    chain of comaprtments.

    A synaptic input is injected in the first compartment of a chain and the
    PSPs in all compartments are measured one after another.
    The class offers functions to extract the heights of the PSPs and
    deterimining a length constant with an exoinential fit.

    :ivar runtime: Experiment runtime in ms (hw domain).
    :ivar chain: CompartmentChain object used to perform the experiments.
    '''

    def __init__(self, calibration: Path, length: int = 5,
                 *,
                 input_neurons: int = 15,
                 input_weight: int = 63,
                 enable_leak_divsion: bool = False):
        interval = 0.2 * pq.ms  # time between spikes
        self.runtime = interval * length
        self.spike_times = np.arange(length) * interval + interval / 2

        self.chain = self._setup(calibration=calibration,
                                 length=length,
                                 input_neurons=input_neurons,
                                 input_weight=input_weight,
                                 enable_leak_divsion=enable_leak_divsion)

    def _setup(self, calibration: Path, length: int,
               *,
               input_neurons: int, input_weight: int,
               enable_leak_divsion: bool) -> CompartmentChain:
        '''
        Setup the Experiment.

        Create a chain, a input population and a projection from this
        population to the first compartment.

        '''
        pynn.setup(initial_config=pynn.helper.chip_from_file(calibration))
        chain = CompartmentChain(length)

        # disable multiplication for leak and set division based on argument
        for comp in chain.compartments:
            comp.set(leak_enable_multiplication=False,
                     leak_enable_division=enable_leak_divsion)

        # Inject inputs in one compartment after another
        pop_in = []
        for spike_time in self.spike_times:
            spike_source = pynn.cells.SpikeSourceArray(
                spike_times=[float(spike_time.rescale(pq.ms))])
            pop_in.append(pynn.Population(input_neurons, spike_source))

        for pop, compartment in zip(pop_in, chain.compartments):
            pynn.Projection(pop, compartment,
                            pynn.AllToAllConnector(),
                            synapse_type=StaticSynapse(weight=input_weight))
        return chain

    def set_parameters_individual(self, params: torch.Tensor):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Array with leak conductance and inter-compartment
            conductance for each compartment individually. The leak conductance
            is be expected to be in the first entries (one entry for each
            compartment) followed by the inter-compartment conductances.
        '''
        length = len(self.chain.compartments)

        if params.size != 2 * length - 1:
            raise ValueError('Provide a value of the leak conductance and the '
                             'inter-compartment conductance for each '
                             'compartment. I.e. the array has to have the '
                             'size 2 * "length of chain" - 1.')

        for comp, g_leak in zip(self.chain.compartments, params[:length]):
            comp.set(leak_i_bias=g_leak)
        for comp, g_icc in zip(self.chain.compartments[1:], params[length:]):
            set_compartment_conductance(comp, g_icc)

    def set_parameters_global(self, params: np.ndarray):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Global values of the leak conductance and
            inter-compartment conductance to set (one value for each parameter
            is shared for all compartments).
        '''

        if params.size != 2:
            raise ValueError('Provide one value for the leak conductance and '
                             'one value for the inter-compartment '
                             'conductance, e.g. `np.array([500, 500])`')
        for comp in self.chain.compartments:
            comp.set(leak_i_bias=params[0])
            set_compartment_conductance(comp, params[1])

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

    def record_membrane_traces(self) -> List[neo.IrregularlySampledSignal]:
        '''
        Measure the membrane potential in one compartment after another.

        :return: Analog recordings of membrane potentials in each compartment.
        '''
        results = []
        for n_comp, comp in enumerate(self.chain.compartments):
            comp.record(['v'])

            pynn.run(float(self.runtime.rescale(pq.ms)))

            # Extract hight
            segments = comp.get_data(clear=True).segments
            sig = segments[-1].irregularlysampledsignals[0]
            sig.annotate(compartment=n_comp, input_spikes=self.spike_times)
            results.append(sig)

            comp.record(None)
            pynn.reset()

        return results

    def measure_response(self, parameters: Optional[np.ndarray]) -> np.ndarray:
        '''
        Measure the PSP heights in response to synaptic inputs in the different
        compartments.

        :parameters: Parameters of the leak and inter compartment conductance
            to set. The parameters can be provided globally for all
            compartments (parameters has length 2) or for each compartment
            indivudually (parameters has lenght 'chain length - 1').
        :return: PSP heights in the different compartments. The first row
            conatins the response in the first compartment, the second the
            response in the second and so on. The different columns are the
            responses to different input sites.
        '''
        self.set_parameters(parameters)
        return extract_psp_heights(self.record_membrane_traces())


def record_data(calibration: Path, parameters: np.ndarray,
                length: int = 5,
                input_neurons: Optional[int] = None,
                input_weight: Optional[int] = None) -> neo.Block:
    '''
    Record input traces for different comaprtments in a chain of compartments.

    :param input_neurons: Number of synchronous inputs (BSS only).
    :param input_weight: Input weight for each neuron (BSS only).
    '''
    # configure chain
    chain_experiment = AttenuationExperiment(calibration, length,
                                             input_weight=input_weight,
                                             input_neurons=input_neurons)
    chain_experiment.set_parameters(parameters)

    # record data
    trace = chain_experiment.record_membrane_traces()

    # save data in block
    block = neo.Block()
    block.segments.append(neo.Segment())
    block.segments[0].irregularlysampledsignals.extend(trace)
    block.annotate(calibration=str(calibration.resolve()),
                   length=length,
                   date=str(datetime.now()),
                   hicann=get_license_and_chip(),
                   input_neurons=input_neurons,
                   input_weight=input_weight,
                   parameters=parameters,
                   spike_times=chain_experiment.spike_times)

    return block


def extract_psp_heights(
        traces: List[neo.IrregularlySampledSignal]) -> np.ndarray:
    '''
    Extract the PSP height from membrane recordings.

    :param traces: Analog signal for each compartment in the chain.
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

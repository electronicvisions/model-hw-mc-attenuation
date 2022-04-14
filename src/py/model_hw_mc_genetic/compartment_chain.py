from typing import List
from pathlib import Path
import numpy as np
import neo
import torch
from scipy.optimize import curve_fit

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

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
                 input_neurons: int = 20,
                 input_weight: int = 63):
        self.runtime = 0.5  # ms (hw)
        self.spike_times = [self.runtime / 4]

        self.chain = self._setup(calibration=calibration,
                                 length=length,
                                 input_neurons=input_neurons,
                                 input_weight=input_weight)

    def _setup(self, calibration: Path, length: int,
               input_neurons: int, input_weight: int) -> CompartmentChain:
        '''
        Setup the Experiment.

        Create a chain, a input population and a projection from this
        population to the first compartment.

        '''
        pynn.setup(initial_config=pynn.helper.chip_from_file(calibration))
        chain = CompartmentChain(length)

        # Inject single stimulus in first compartment and measure responses
        pop_in = pynn.Population(input_neurons, pynn.cells.SpikeSourceArray(
            spike_times=self.spike_times))

        pynn.Projection(pop_in, chain.compartments[0],
                        pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=input_weight))
        return chain

    def set_parameters(self, params: torch.Tensor):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Array with leak concutance and inter-compartment
            conductance for each comaprtment individually. The leak condctance
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
        for comp, g_icc in zip(self.chain.compartments[1:], params[:length]):
            set_compartment_conductance(comp, g_icc)

    def set_parameters_global(self, params: np.ndarray):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Global values of the leak concutance and
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

    def measure_result(self) -> List[neo.IrregularlySampledSignal]:
        '''
        Measure the membrane potential in one compartment after another.

        :return: Analog recordings of membrane potentials in each compartment.
        '''
        results = []
        for n_comp, comp in enumerate(self.chain.compartments):
            comp.record(['v'])

            pynn.run(self.runtime)
            pynn.reset()

            # Extract hight
            sig = comp.get_data().segments[-1].irregularlysampledsignals[0]
            sig.annotate(compartment=n_comp)
            results.append(sig)

            comp.record(None)

        return results

    def __del__(self):
        pynn.end()


def extract_psp_heights(
        traces: List[neo.IrregularlySampledSignal]) -> np.ndarray:
    '''
    Extract the PSP height from membrane recordings.

    :param traces: Analog signal for each compartment in the chain.
    :return: Height of PSP (difference between maximum and baseline).
    '''
    heights = []
    for sig in traces:
        # Take first quarter as base line
        heights.append(sig.max() - sig[:int(len(sig) / 4)].mean())
    return np.asarray(heights)


def fit_length_constant(
        traces: List[neo.IrregularlySampledSignal]) -> np.ndarray:
    '''
    Fit an exponential decay to the PSP height in the compartments.

    :param traces: Analog signal for each compartment in the chain.
    :return: Length constant (exponential fir to PSP height as function of
        compartment)
    '''
    heights = extract_psp_heights(traces)

    norm_height = heights / heights[0]
    compartments = np.arange(len(heights))

    def fitfunc(location, tau, offset):
        return np.exp(- location / tau) + offset

    # initial guess for fit parameters
    guessed_tau = len(traces) if norm_height[-1] > 1 / np.e else \
        compartments[np.argmin(norm_height > 1 / np.e)]
    p_0 = {'tau': guessed_tau, 'offset': norm_height[-1]}
    bounds = {'tau': [0, np.inf],
              'offset': p_0['offset'] + np.array([-1, 1])}

    popt = curve_fit(
        fitfunc,
        compartments,
        norm_height,
        p0=[p_0['tau'], p_0['offset']],
        bounds=([bounds['tau'][0], bounds['offset'][0]],
                [bounds['tau'][1], bounds['offset'][1]]))[0]

    return np.array([popt[0]])

import argparse
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import neo
import torch
import quantities as pq

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

from model_hw_si_nsc_dendrites.compartment_chain import CompartmentChain
from model_hw_si_nsc_dendrites.helper import get_license_and_chip, \
    set_compartment_conductance

from model_hw_mc_genetic.helper import set_leak_conductance
from model_hw_mc_genetic.attenuation import extract_psp_heights
from model_hw_mc_genetic.attenuation.base import Base


class AttenuationExperiment(Base):
    '''
    Experiment which measures the attenuation of a PSP as it travels along a
    chain of compartments on the BSS-2 system.

    A synaptic input is injected in the first compartment of a chain and the
    PSPs in all compartments are measured one after another.

    :ivar calibration: Path to portable binary calibration.
    :ivar input_neurons: Number of synchronous inputs.
    :ivar input_weight: Synaptic weight of each input.

    :ivar runtime: Experiment runtime in ms (hw domain).
    :ivar spike_times: Times at which the different inputs are injected.
    :ivar chain: CompartmentChain object used to perform the experiments.
    '''

    def __init__(self, calibration: Path, length: int = 5,
                 *,
                 input_neurons: int = 15,
                 input_weight: int = 63) -> None:
        super().__init__(length)

        self.calibration = calibration.resolve()
        self.input_neurons = input_neurons
        self.input_weight = input_weight

        interval = 0.2 * pq.ms  # time between spikes
        self.runtime = interval * length
        self.spike_times = np.arange(length) * interval + interval / 2

        self.chain = self._setup()

    def _setup(self) -> CompartmentChain:
        '''
        Setup the Experiment.

        Create a chain, a input population and a projection from this
        population to the first compartment.
        '''
        pynn.setup(initial_config=pynn.helper.chip_from_file(self.calibration))
        chain = CompartmentChain(self.length)

        # Inject inputs in one compartment after another
        pop_in = []
        for spike_time in self.spike_times:
            spike_source = pynn.cells.SpikeSourceArray(
                spike_times=[float(spike_time.rescale(pq.ms))])
            pop_in.append(pynn.Population(self.input_neurons, spike_source))

        for pop, compartment in zip(pop_in, chain.compartments):
            pynn.Projection(
                pop, compartment, pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.input_weight))
        return chain

    def set_parameters_individual(self, params: torch.Tensor):
        length = self.length

        if params.size != 2 * length - 1:
            raise ValueError('Provide a value of the leak conductance and the '
                             'inter-compartment conductance for each '
                             'compartment. I.e. the array has to have the '
                             'size 2 * "length of chain" - 1.')

        for comp, g_leak in zip(self.chain.compartments, params[:length]):
            set_leak_conductance(comp, g_leak)
        for comp, g_icc in zip(self.chain.compartments[1:], params[length:]):
            set_compartment_conductance(comp, g_icc)

    def set_parameters_global(self, params: np.ndarray):
        if params.size != 2:
            raise ValueError('Provide one value for the leak conductance and '
                             'one value for the inter-compartment '
                             'conductance, e.g. `np.array([500, 500])`')
        for comp in self.chain.compartments:
            set_leak_conductance(comp, params[0])
        for comp in self.chain.compartments[1:]:
            set_compartment_conductance(comp, params[1])

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

    def measure_response(self, parameters: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        '''
        Measure the PSP heights in response to synaptic inputs in the different
        compartments.

        :parameters: Parameters of the leak and inter compartment conductance
            to set. The parameters can be provided globally for all
            compartments (parameters has length 2) or for each compartment
            individually (parameters has length 'chain length - 1').
        :return: PSP heights in the different compartments. The first row
            contains the response in the first compartment, the second the
            response in the second and so on. The different columns are the
            responses to different input sites.
        '''
        if parameters is not None:
            self.set_parameters(parameters)
        return extract_psp_heights(self.record_membrane_traces())

    def record_data(self, parameters: np.ndarray) -> neo.Block:
        '''
        Record membrane traces in each compartment.

        The recorded traces and meta data relavant for the experiment are
        saved in a :class:`neo.Block`.

        :param parameters: Leak and inter-compartment conductance. Can be set
            globally or locally, compare
            :meth:`~AttenuationExperiment.set_parameters`.
        '''
        # configure chain
        self.set_parameters(parameters)

        # record data
        trace = self.record_membrane_traces()

        # save data in block
        block = neo.Block()
        block.segments.append(neo.Segment())
        block.segments[0].irregularlysampledsignals.extend(trace)
        block.annotate(calibration=str(self.calibration),
                       length=self.length,
                       date=str(datetime.now()),
                       hicann=get_license_and_chip(),
                       input_neurons=self.input_neurons,
                       input_weight=self.input_weight,
                       parameters=parameters,
                       spike_times=self.spike_times,
                       experiment='attenuation_bss')

        return block

    @property
    def default_limits(self) -> np.ndarray:
        return default_conductance_limits.repeat(self.length, axis=0)[:-1]


# default limits for leak and inter-compartment conductance
default_conductance_limits = np.array([[0, 1022], [0, 1022]])


def add_bss_psp_args(parser: argparse.ArgumentParser
                     ) -> argparse._ArgumentGroup:
    '''
    Add arguments which are related to an attenuation experiment on BSS-2
    which use synaptic inputs to measure the attenuation.

    Add arguments '-input_neurons' and '-input_weight'.

    :param parser: Parser to which the arguments are added.
    :returns: Group with the added options.
    '''
    group = parser.add_argument_group(
        'BSS-2 - Attenuation of EPSPs',
        description='Options which control how an experiment on BSS-2 which '
                    'measures the attenuation of excitatory post-synaptic '
                    'is configured.')
    parser.add_argument('-calibration',
                        type=str,
                        help='Path to portable binary calibration. If not '
                             'provided the latest nightly calibration is '
                             'used.')
    group.add_argument("-input_neurons",
                       help="Number of synchronous inputs.",
                       type=int,
                       default=10)
    group.add_argument("-input_weight",
                       help="Input weight for each neuron.",
                       type=int,
                       default=30)
    return group

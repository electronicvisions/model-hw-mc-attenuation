import argparse
from typing import List, Optional, Sequence
from pathlib import Path
from datetime import datetime
import numpy as np
import neo
import torch
import quantities as pq

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

from model_hw_mc_attenuation.compartment_chain import CompartmentChain
from model_hw_mc_attenuation.helper import get_license_and_chip, \
    set_axial_conductance, set_leak_conductance
from model_hw_mc_attenuation import extract_psp_heights
from model_hw_mc_attenuation import AttenuationExperiment as Base


class AttenuationExperiment(Base):
    '''
    Experiment which measures the attenuation of a PSP as it travels along a
    chain of compartments on the BSS-2 system.

    A synaptic input is injected in the first compartment of a chain and the
    PSPs in all compartments are measured one after another.

    :ivar calibration: Path to portable binary calibration.
    :ivar input_neurons: Number of synchronous inputs.
    :ivar input_weight: Synaptic weight of each input.
    :ivar n_average: Number of times the experiment is executed and the results
        are averaged over.

    :ivar runtime: Experiment runtime in ms (hw domain).
    :ivar spike_times: Times at which the different inputs are injected.
    :ivar chain: CompartmentChain object used to perform the experiments.
    '''

    def __init__(self, calibration: Path, length: int = 5,
                 *,
                 input_neurons: int = 15,
                 input_weight: int = 63,
                 n_average: int = 1) -> None:
        super().__init__(length)

        self.calibration = calibration.resolve()
        self.input_neurons = input_neurons
        self.input_weight = input_weight
        if n_average < 1:
            raise ValueError(
                f"n_average must be greater than 0. Provided was {n_average}.")
        self.n_average = n_average

        self.interval = 0.2 * pq.ms  # time between spikes
        self.runtime = self.interval * length * n_average
        self.spike_times = np.arange(length) * self.interval \
            + self.interval / 2

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
            spike_times = np.arange(
                spike_time.rescale(pq.ms), self.runtime.rescale(pq.ms),
                self.length * self.interval.rescale(pq.ms))
            spike_source = pynn.cells.SpikeSourceArray(spike_times=spike_times)
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
            set_axial_conductance(comp, g_icc)

    def set_parameters_global(self, params: np.ndarray):
        if params.size != 2:
            raise ValueError('Provide one value for the leak conductance and '
                             'one value for the inter-compartment '
                             'conductance, e.g. `np.array([500, 500])`')
        for comp in self.chain.compartments:
            set_leak_conductance(comp, params[0])
        for comp in self.chain.compartments[1:]:
            set_axial_conductance(comp, params[1])

    def _record_raw_traces(self) -> List[neo.IrregularlySampledSignal]:
        '''
        Measure the membrane potential in one compartment after another.

        :return: Analog recordings of membrane potentials in each compartment.
            The traces are not averaged over the different repetitions.
        '''
        results = []
        for n_comp, comp in enumerate(self.chain.compartments):
            comp.record(['v'])

            pynn.run(float(self.runtime.rescale(pq.ms)))

            # Extract hight
            segments = comp.get_data(clear=True).segments
            sig = segments[-1].irregularlysampledsignals[0]
            sig.annotate(compartment=n_comp, input_spikes=self.spike_times,
                         n_average=self.n_average)
            results.append(sig)

            comp.record(None)
            pynn.reset()

        return results

    def record_membrane_traces(self) -> List[neo.IrregularlySampledSignal]:
        '''
        Measure the membrane potential in one compartment after another.

        :return: Analog recordings of membrane potentials in each compartment.
        '''
        if self.n_average > 1:
            return [_average_traces(_split_traces(trace, self.n_average)) for
                    trace in self._record_raw_traces()]
        return self._record_raw_traces()

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
                       chip_id=get_license_and_chip(),
                       input_neurons=self.input_neurons,
                       input_weight=self.input_weight,
                       n_average=self.n_average,
                       parameters=parameters,
                       spike_times=self.spike_times,
                       experiment='attenuation_bss')

        return block

    @property
    def default_limits(self) -> np.ndarray:
        return default_conductance_limits.repeat(self.length, axis=0)[:-1]


# default limits for leak and inter-compartment conductance
default_conductance_limits = np.array([[0, 1022], [0, 1022]])


def _split_traces(trace: neo.IrregularlySampledSignal, n_average: int
                  ) -> List[neo.IrregularlySampledSignal]:
    '''
    Split the provided membrane trace in `n_average` equally sized traces.

    :param trace: Recording of a compartment's membrane trace.
    :param n_average: Number of experiment repetitions.
    :returns: List containing equally sized splits of the provided trace.
    '''
    intervals = trace.duration / n_average
    splitted_traces = []
    for n_split in range(n_average):
        splitted_traces.append(trace.time_slice(
            trace.t_start + n_split * intervals,
            trace.t_start + (n_split + 1) * intervals))

    # Make length of all splits equal
    max_common_len = np.min([len(trace) for trace in splitted_traces])
    common_len_signal = [trace[:max_common_len] for trace in splitted_traces]

    return common_len_signal


def _average_traces(traces: Sequence[neo.IrregularlySampledSignal]
                    ) -> neo.IrregularlySampledSignal:
    '''
    Calculate the average trace from the provided traces.

    The times are not averaged but the time of the first trace is used for the
    output signal. All traces must have the same number of samples.

    :param traces: Traces that will be averaged to one trace.
    :returns: Average of the given traces. The annotation and times are
        extracted from the first trace.
    '''
    times = traces[0].times
    traces_magnitude = [s.magnitude for s in traces]
    mag = np.mean(traces_magnitude, axis=0)

    signal = neo.IrregularlySampledSignal(times, mag * traces[0].units)
    signal.annotate(**traces[0].annotations)

    return signal


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
    group.add_argument("-n_average",
                       help="Number of times the experiment is executed and "
                            "the results are averaged over.",
                       type=int,
                       default=1)
    return group


# Bounds for the exponential fit for amplitudes recorded on BSS-2
# - tau: The typical length constants are in the range of 1-8
# - offset: We subtract the leak potential before fitting. The noise on
#   the measured potentials has a standard deviation of around 3 LSB
# - scaling factor: Restrict to the 10-bit range of the MADC
integration_bounds = ([0, -10, 0],
                      [20, 10, 1022])

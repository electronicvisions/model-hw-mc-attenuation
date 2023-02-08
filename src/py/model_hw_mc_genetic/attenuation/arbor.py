#!/usr/bin/env python3

from datetime import datetime
from typing import List, Optional

import arbor
import neo
import numpy as np
import quantities as pq
import torch

from model_hw_mc_genetic.attenuation import extract_psp_heights
from model_hw_mc_genetic.attenuation.base import Base


class ChainRecipe(arbor.recipe):
    '''
    Create a chain of compartments and insert a synaptic stimulus in the first
    compartment.
    :cvar delay_between_inputs: Time (in ms) between inputs.
    '''
    delay_between_inputs = 150  # ms

    def __init__(self,
                 length: int,
                 input_weight: float = 0.10,
                 capacitance: pq.Quantity = 125 * pq.pF) -> None:
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)

        interval = 200 * pq.ms  # time between spikes
        self.runtime = interval * length
        self.spike_times = np.arange(length) * interval + interval / 2

        self.weight = input_weight

        self.length = length

        self.neuron_params = {'capacitance': capacitance,
                              'leak_potential': -65,  # mV
                              'comp_length': 1000 * pq.um,
                              'comp_diameter': 4 * pq.um}

        self.ncells = 1

        self._params = [self.tau_mem_to_g_leak(10 * pq.ms)] * self.length + \
            [self.tau_icc_to_g_axial(10 * pq.ms)] * (self.length - 1)

    def set_parameters(self, params: np.ndarray):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Array with leak conductance (in S / cm^2) and
            inter-compartment conductance (in S / cm) for each compartment
            individually. The leak conductance is expected to be in the
            first entries (one entry for each compartment) followed by the
            inter-compartment conductances.
        '''
        if params.size != 2 * self.length - 1:
            raise ValueError('Provide a value of the leak conductance and the '
                             'inter-compartment conductance for each '
                             'compartment. I.e. the array has to have the '
                             'size `2 * length - 1`.')
        self._params = np.asarray(params)

    def _labels(self) -> arbor.label_dict:
        '''
        Create labels for compartments and center of compartments.

        :return: Label dictionary.
        '''
        # create label for center of compartments
        branch = 0
        labels = {}
        for curr_comp in range(self.length):
            labels[f'center_comp_{curr_comp}'] = \
                f'(location {branch} {(curr_comp + 0.5) / self.length})'

        # combine all centers to an single label
        labels['comp_centers'] = f"(join {' '.join(labels.values())})"

        # create label for compartments
        start_stop = np.linspace(0, 1, self.length + 1)
        for curr_comp in range(self.length):
            start = start_stop[curr_comp]
            stop = start_stop[curr_comp + 1]
            labels[f'comp_{curr_comp}'] = f"(cable {branch} {start} {stop})"

        return arbor.label_dict(labels)

    def _set_conductances(self, decor):
        '''
        Set leak conductance as well as inter-compartment conductance.
        '''
        # leak conductance
        for curr_comp, g_leak in enumerate(self._params[:self.length]):
            # [g_leak] = S / cm^2
            leak_potential = self.neuron_params['leak_potential']
            density_mech = arbor.density(f'pas/e={leak_potential}',
                                         {'g': g_leak})
            decor.paint(f'"comp_{curr_comp}"', density_mech)

        branch = 0
        start_stop = (np.arange(self.length) + 0.5) / self.length
        # inter-compartment conductance: draw axial conductance from
        # center of one compartment to center of other compartment
        for n_cond, g_icc in enumerate(self._params[self.length:]):
            # [g_icc] = S / cm
            start = start_stop[n_cond]
            stop = start_stop[n_cond + 1]
            decor.paint(f'(cable {branch} {start} {stop})',
                        rL=1 / g_icc)

        return decor

    def cell_description(self, gid: int) -> arbor.cable_cell:
        '''
        Return the cell description of a single neuron which has the
        morphology of a compartment chain.

        :param gid: Global ID of the cell which should be created.
        :returns: Cable cell in form of a compartment chain.
        '''
        self.check_gid(gid)

        total_length = self.neuron_params['comp_length'] * self.length
        tree = arbor.segment_tree()
        tree.append(arbor.mnpos,
                    arbor.mpoint(0, 0, 0, self.neuron_params['comp_diameter']),
                    arbor.mpoint(total_length, 0, 0,
                                 self.neuron_params['comp_diameter']),
                    tag=3)

        decor = arbor.decor()

        # capacitance and leak potential
        cap_per_area = \
            self.capacitance_to_arbor(self.neuron_params['capacitance'],
                                      self.neuron_params['comp_length'],
                                      self.neuron_params['comp_diameter'])
        decor.set_property(cm=cap_per_area,
                           Vm=self.neuron_params['leak_potential'])

        self._set_conductances(decor)

        # Place synapses at centers of first compartment
        for n_comp in range(self.length):
            decor.place(f'"center_comp_{n_comp}"',
                        arbor.synapse('expsyn', {'tau': 10}),
                        f'synapse_{n_comp}')

        # Split segment in compartments
        decor.discretization(arbor.cv_policy_fixed_per_branch(self.length))

        return arbor.cable_cell(tree, decor, self._labels())

    @staticmethod
    def capacitance_to_arbor(capacitance: pq.Quantity,
                             length: pq.Quantity,
                             diameter: pq.Quantity) -> float:
        '''
        Convert capacitance from farad to farad per area.

        :param capacitance: Membrane capacitance.
        :param lenght: Length of a compartment.
        :param diameter: Diameter of a compartment.
        :return: Capacitance in F/m^2.
        '''

        surface_area = np.pi * diameter * length
        cap_per_area = capacitance / surface_area

        return cap_per_area.rescale(pq.F / pq.m**2).magnitude

    def tau_mem_to_g_leak(self, tau_mem: pq.Quantity) -> float:
        '''
        Convert membrane time constant to leak conductivity per area.

        :param tau_mem: Membrane time constant.
        :return: Leak conductance in S/cm^2.
        '''

        total_cond = self.neuron_params['capacitance'] / tau_mem
        surface_area = np.pi * self.neuron_params['comp_diameter'] * \
            self.neuron_params['comp_length']
        cond_per_area = total_cond / surface_area

        return cond_per_area.rescale(pq.S / pq.cm**2).magnitude

    def tau_icc_to_g_axial(self, tau_icc: pq.Quantity) -> float:
        '''
        Convert inter-compartment time constant to inter-compartment
        resistance in S / cm.

        :param tau_icc: Inter-compartment time constant.
        :return: Inter-compartment conductance in S/cm.
        '''

        total_cond = self.neuron_params['capacitance'] / tau_icc
        face_area = np.pi * self.neuron_params['comp_diameter']**2 / 4
        comp_length = self.neuron_params['comp_length']
        cond_per_length = total_cond / face_area * comp_length

        return cond_per_length.rescale(pq.siemens / pq.cm).magnitude

    def num_cells(self) -> int:
        '''
        Number of cells which are simulated.

        :returns: Number of cells in the simulation.
        '''
        return self.ncells

    def cell_kind(self, gid) -> arbor.cell_kind:
        '''
        Cell type for different cells in the simulation.

        :param gid: Global ID of cell.
        :returns: Cell type of the cell with the given GID.
        '''
        self.check_gid(gid)
        return arbor.cell_kind.cable

    def event_generators(self, gid: int) -> List[arbor.event_generator]:
        '''
        Create generators which will inject spikes in one compartment after
        another.

        :param gid: Global ID of cell.
        :returns: Event generators connected to the cell with the given GID.
        '''
        self.check_gid(gid)
        generators = []
        for n_comp, spike_time in enumerate(self.spike_times):
            generators.append(
                arbor.event_generator(f'synapse_{n_comp}',
                                      self.weight,
                                      arbor.explicit_schedule([spike_time])))
        return generators

    def probes(self, gid: int) -> List[arbor.probe]:
        '''
        Place probes at center of compartments.

        :param gid: Global ID of cell.
        :returns: Probes connected to the cell with the given GID.
        '''
        self.check_gid(gid)
        probes = [arbor.cable_probe_membrane_voltage(
            f'"center_comp_{n_comp}"') for n_comp in range(self.length)]
        return probes

    # pylint: disable=no-self-use
    def global_properties(self, kind: arbor.cell_kind
                          ) -> arbor.cable_global_properties:
        '''
        Properties of the given cell type.

        :param kind: Kind of cell.
        :returns: Properties of the given cell kind.
        '''
        if kind != arbor.cell_kind.cable:
            raise RuntimeError('No properties for the requested cell kind.')
        return arbor.neuron_cable_properties()

    def check_gid(self, gid: int):
        '''
        Check that the given GID is registered in this recipe.

        :param gid: Global ID of cell.
        :raises RuntimeError: When GID is not registered in recipe.
        '''
        if gid >= self.ncells:
            raise RuntimeError(f'This recipe only contains {self.ncells} '
                               f'cells. `gid` {gid} can not be accessed.')


class AttenuationExperiment(Base):
    '''
    Experiment which simulates the attenuation of a PSP as it travels along a
    chain of compartments in arbor.

    A synaptic input is injected in the first compartment of a chain and the
    PSPs in all compartments are measured one after another.

    :ivar recipe: Arbor recipe of the simulation.
    '''
    def __init__(self, length: int = 4, input_weight: float = 0.05) -> None:
        super().__init__(length)
        self.recipe = ChainRecipe(length, input_weight)

    @property
    def spike_times(self):
        return self.recipe.spike_times

    def set_parameters_individual(self, params: torch.Tensor):
        params = np.asarray(params)
        self.recipe.set_parameters(params)

    def set_parameters_global(self, params: torch.Tensor):
        if params.size != 2:
            raise ValueError('Provide one value for the leak conductance and '
                             'one value for the inter-compartment '
                             'conductance, e.g. `np.array([500, 500])`')

        chain_length = self.recipe.length
        expanded = [float(params[0])] * chain_length + \
            [float(params[1])] * (chain_length - 1)

        self.set_parameters(torch.tensor(expanded))

    def record_membrane_traces(self) -> List[neo.IrregularlySampledSignal]:
        '''
        Measure the membrane potential in one compartment after another.

        :return: Recordings of membrane potentials in each compartment.
        '''
        sampling_period = 0.1 * pq.ms

        sim = arbor.simulation(self.recipe)

        # membrane recordings
        scheduler = arbor.regular_schedule(sampling_period)
        handles = [sim.sample((0, n_comp), scheduler)
                   for n_comp in range(self.recipe.length)]
        # Run simulation
        sim.run(self.recipe.runtime)

        # Use same format as PyNN
        anasigs = []
        for n_comp, handle in enumerate(handles):
            time_voltage = sim.samples(handle)[0][0]
            voltage = time_voltage[:, 1]
            anasigs.append(neo.IrregularlySampledSignal(
                time_voltage[:, 0] * pq.ms,
                voltage * pq.dimensionless,
                compartment=n_comp,
                input_spikes=self.spike_times))
        return anasigs

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
        :returns: :class:`~neo.Block` with the recorded membrane traces and
            experiment meta data.
        '''
        self.set_parameters(parameters)

        # record data
        results = self.record_membrane_traces()

        # save data in block
        block = neo.Block()
        block.segments.append(neo.Segment())
        block.segments[0].irregularlysampledsignals.extend(results)
        block.annotate(length=self.length,
                       date=str(datetime.now()),
                       license='arbor',
                       parameters=parameters,
                       spike_times=self.spike_times)
        return block


def time_constants_to_conductance(experiment: AttenuationExperiment,
                                  time_constants: np.ndarray) -> np.ndarray:
    '''
    Convert membrane time constant and inter-compartment time constant in
    conductances.

    If no time constants are supplied a default value is returned.

    :param experiment: AttenuationExperiment for which to determine the
        convert the time constants to conductances.
    :param time_constants: Time constants to convert. The time constants can
        represent global values (length of array is two) or local values
        (size of array is `2 * length - 1`). If not set default values are
        returned.
    :returns: Leak and inter-compartment conductance. One leak conductance for
        each compartment in the chain and the inter-compartment conductance
        for each connection between compartments.
        The leak conductance is given in S/cm^2, the inter-compartment
        conductance in S/cm.
    '''
    if time_constants is None:
        # chose defaults in the center of the translated conductances.
        min_values = time_constants_to_conductance(
            experiment, default_tau_limits[:, 1] * pq.ms)
        max_values = time_constants_to_conductance(
            experiment, default_tau_limits[:, 0] * pq.ms)

        return np.mean([min_values, max_values], axis=0)

    length = experiment.length
    if len(time_constants) == 2:
        time_constants = np.repeat(time_constants, length)[:-1]

    leak_cond = experiment.recipe.tau_mem_to_g_leak(time_constants[:length])
    icc_cond = experiment.recipe.tau_icc_to_g_axial(time_constants[length:])

    return np.concatenate([leak_cond, icc_cond])


# default limits for leak and inter-compartment time constants (in ms)
default_tau_limits = np.array([[1, 20], [1, 20]])

from typing import Optional
import numpy as np

import pynn_brainscales.brainscales2 as pynn
from model_hw_mc_genetic.helper import set_axial_conductance


class CompartmentChain:
    """
    Creates a chain of neuron compartments which are connected via tunable
    conductances. Each compartment is made up of two neuron circuits.
    The spiking threshold is disabled for all compartments.
    It is assumed that logical neurons are placed in a linear fashion on one
    quadrant of the chip.

    :ivar all_neurons: pynn.Population containing all neurons which are used to
        construct the chain.
    :ivar compartments: List of pynn.PopulationViews to the circuits which
        control the properties of the corresponding compartment.
    """

    def __init__(self, length: int, conductance: Optional[int] = None):
        """
        Creates a chain of compartments. Each compartment is made up of two
        neuron circuits. The first neuron circuit will use a resistor (R)
        to connect to the somatic line while the second connects directly
        to the line. The first and last neuron circuits are not connected to
        the somatic line. The somatic line has to be connected to the right
        for every uneven neuron circuit.
        For every uneven neuron circuit we deactivate the capacitor as well as
        the leak such that the dynamics of the compartment are determined by
        the other neuron circuit (C).

        The circuit diagram displays the configuration on hardware. The number
        at the top denotes the neuron circuit; the number in the bottom the
        number of the compartment.

         0     1     2     3     4     5  ... ...    -2    -1

        -+-   -+-----+-   -+-----+-   -+- ... ... ----+-   -+-
         |     |     |     |     |     |              |     |
               |     R     |     R     |              R
         |     |     |     |     |     |              |     |
        -C-----+-   -C-----+-   -C-----+-            -C-----+-

            1           2           3     ... ...     length

        Note: Each neuron circuit will be presented by one neuron in a PyNN
              population.

        :param length: Length of the chain.
        :param conductance: Conductance between different compartments. If
            a value is given, the calibration is overwritten.
            Depending on the exact value the resistor is operated in the
            division, normal or multiplication mode. Compare
            `helpers.set_axial_conductance`.
        """
        # Create neurons
        pop = pynn.Population(length * 2, pynn.cells.HXNeuron())

        # Combine two neuron circuits to one compartment
        pynn.PopulationView(pop, np.arange(0, 2 * length, 2)).set(
            multicompartment_connect_right=True)

        # disable capacitor as well as leak conductance of uneven circuits
        pynn.PopulationView(pop, np.arange(1, 2 * length, 2)).set(
            leak_i_bias=0,
            leak_enable_division=True,
            membrane_capacitance_capacitance=0)

        # Enable direct connection to somatic line and close connection to the
        # right for uneven neurons (don't connect last neuron circuit)
        pynn.PopulationView(pop, np.arange(1, 2 * length - 1, 2)).set(
            multicompartment_connect_soma=True,
            multicompartment_connect_soma_right=True)

        # Connect resistor to somatic line for even circuits (don't connect
        # first neuron circuit)
        if conductance is not None:
            set_axial_conductance(
                pynn.PopulationView(pop, np.arange(2, 2 * length, 2)),
                conductance)
        else:
            pynn.PopulationView(pop, np.arange(2, 2 * length, 2)).set(
                multicompartment_enable_conductance=True)

        # Disable spiking
        pop.set(threshold_enable=False)

        self.all_neurons = pop
        # Every even neuron circuit controls the capacitance, resistance,
        # leak, ... of a single compartment. Save views on these circuits as
        # compartments
        self.compartments = [pynn.PopulationView(pop, [n]) for n in
                             range(0, 2 * length, 2)]

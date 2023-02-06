import pynn_brainscales.brainscales2 as pynn


def get_neuron_population(size: int) -> pynn.Population:
    """
    Minimal helper function:
    Construct a neuron population to be used within this experiment.
    :param size: Population size
    :return: Neuron population
    """
    return pynn.Population(size, pynn.cells.HXNeuron())

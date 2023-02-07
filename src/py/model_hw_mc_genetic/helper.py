from pyNN.common import BasePopulation


def set_leak_conductance(pop: BasePopulation, value: int):
    '''
    Sets the leak conductance of populations of HXNeurons. Depending on the
    supplied conductance the division/multiplication mode is enabled.

    :param pop: Population of HXNeurons for which the leak conductance should
        be set.
    :param value: Conductance between 0 and 3068. If the conductance is below
        1023 the low conductance mode is enabled. If it is higher than 2045,
        the high conductance mode is enabled.
    '''
    multiplication = value > 2045
    division = value < 1023
    pop.set(leak_enable_multiplication=multiplication,
            leak_enable_division=division,
            leak_i_bias=value % 1023)

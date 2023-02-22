from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import torch


class Base(ABC):
    '''
    Experiment which measures the attenuation of a PSP or a constant current
    in a chain of compartments.

    :ivar length: Number of compartments in the chain.
    '''

    def __init__(self, length: int) -> None:
        self.length = length

    @abstractmethod
    def set_parameters_individual(self, params: torch.Tensor):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Array with leak conductance and inter-compartment
            conductance for each compartment individually. The leak conductance
            is be expected to be in the first entries (one entry for each
            compartment) followed by the inter-compartment conductances.
        '''
        raise NotImplementedError

    @abstractmethod
    def set_parameters_global(self, params: np.ndarray):
        '''
        Adjust leak conductance and inter-compartment conductance.

        :param params: Global values of the leak conductance and
            inter-compartment conductance to set (one value for each parameter
            is shared for all compartments).
        '''
        raise NotImplementedError

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

    @abstractmethod
    def measure_response(self, parameters: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        '''
        Measure the PSP heights or the deflection due to a constant current.

        :parameters: Parameters of the leak and inter compartment conductance
            to set. The parameters can be provided globally for all
            compartments (parameters has length 2) or for each compartment
            individually (parameters has length 'chain length - 1').
        :return: PSP heights or deflection in the different compartments. The
            first row contains the response in the first compartment, the
            second the response in the second and so on. The different columns
            are the responses to different input sites.
        '''
        raise NotImplementedError

    @property
    @abstractmethod
    def default_limits(self) -> np.ndarray:
        '''
        Return default limits of the parameters of this experiment.

        :return: Limits of the experiment parameters. The different rows
            represent different parameters, the left column the lower limit
            and the right column the upper limit.
        '''
        raise NotImplementedError

    @property
    def default_parameters(self) -> np.ndarray:
        '''
        Return default parameters of this experiment.

        :return: Parameters at the center of the experiment limits.
        '''
        return self.default_limits.mean(1)

    def expand_parameters(self, parameters: Optional[np.ndarray] = None
                          ) -> np.ndarray:
        '''
        Expand the given parameters to individual parameters or return default
        values.

        :parameters: Parameters of the leak and inter compartment conductance
            to expand if needed. If no parameters are supplied the default
            parameters are returned.
        :returns: Parameters for each compartment/inter-compartment connection
            individually.
        '''
        if parameters is not None:
            if len(parameters) == 2 * self.length - 1:
                return parameters
            if len(parameters) == 2:
                return np.repeat(parameters, self.length)[:-1]
            raise ValueError('The length of the supplied parameters does not '
                             'match the expected length of 2 or '
                             f'{self.length}.')

        # return default values
        return self.default_parameters

    def parameter_names(self, global_names: bool = False) -> List[str]:
        '''
        Get the names of the parameters which can be configured for the
        experiment.

        The parameters are the leak and inter-compartment conductance. The
        number of parameters depends on the length of the chain.

        :param global_names: Leak and inter-compartment conductances are set
            globally. Do return the global parameter names.
        :returns: Names for all parameters of the experiment.
        '''
        parameter_base_names = ['g_leak', 'g_icc']

        # Global evaluation
        if global_names:
            return parameter_base_names

        parameters = [f'{parameter_base_names[0]}_{comp}' for comp in
                      range(self.length)]
        parameters.extend(
            [f'{parameter_base_names[1]}_{comp}_{comp + 1}' for comp in
             range(self.length - 1)])

        return parameters

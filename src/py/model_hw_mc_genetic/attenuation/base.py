from abc import ABC, abstractmethod
from typing import Optional
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
    def measure_response(self, parameters: Optional[np.ndarray]) -> np.ndarray:
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

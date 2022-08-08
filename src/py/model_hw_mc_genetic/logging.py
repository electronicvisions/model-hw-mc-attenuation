from typing import List
import pandas as pd


class ExperimentLogger:
    '''
    Log parameters, results and meta-parameter of each experiment execution.

    This can be helpful for SBI, where the tested parameters and results are
    not easily accesible.
    '''

    def __init__(self,
                 parameters: List[str],
                 observables: List[str],
                 meta_data: List[str]):
        """
        :param parameters: Names of the parameters which are passed to
            `log_statistics()`.
        :param observales: Names of observables which are passed to
            `log_statistics()`.
        :param meta_data: Names of meta_data which are passed to
            `log_statistics()`.
        """

        self._parameter_names = parameters
        self._observable_names = observables
        self._meta_data_names = meta_data

        self._results = \
            {'parameters': {name: [] for name in parameters},
             'observables': {name: [] for name in observables},
             'meta_data': {name: [] for name in meta_data}}

    def log_statistics(self, parameters, observables, meta_data):
        '''
        Log all statistics: parameters, observables and meta data.
        '''

        # Log parameters
        for name, value in zip(self._parameter_names, parameters):
            self._results['parameters'][name].append(value)

        # Log observables
        for name, value in zip(self._observable_names, observables):
            self._results['observables'][name].append(value)

        # Log meta data
        for name, value in zip(self._meta_data_names, meta_data):
            self._results['meta_data'][name].append(value)

    def get_data_frame(self) -> pd.DataFrame:
        '''
        Return data frame with all logged data.

        :return: All logged data.
        '''
        sub_dfs = {name: pd.DataFrame(value) for name, value
                   in self._results.items()}
        return pd.concat(sub_dfs, axis=1)

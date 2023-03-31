from typing import List, Optional, Tuple, Sequence
import os
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from pyNN.common import BasePopulation

from model_hw_mc_attenuation import AttenuationExperiment


class AttributeNotIdentical(Exception):
    '''
    Exception used py :func:`get_identical_attr` if the requested attribute
    is not identical for all DataFrames.
    '''


def conductance_to_capmem(value: int) -> Tuple[int, bool, bool]:
    '''
    Map the given conductance to a CapMem value and decide if
    multiplication/division should be enabled.

    :param value: Conductance between 0 and 3068. If the conductance is below
        1023 the low conductance mode should be enabled. If it is higher than
        2045, the high conductance mode should be enabled.
    :returns: Tuple of (CapMem value, enable division, enable multiplication).
    '''
    multiplication = value > 2045
    division = value < 1023
    return int(value % 1023), division, multiplication


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
    norm_value, division, multiplication = conductance_to_capmem(value)
    pop.set(leak_enable_multiplication=multiplication,
            leak_enable_division=division,
            leak_i_bias=norm_value)


def set_axial_conductance(pop: BasePopulation, value: int):
    '''
    Enables and sets the inter-compartment conductance of populations of
    HXNeurons.

    Depending on the supplied conductance the division/multiplication mode is
    enabled.

    :param pop: Population of HXNeurons for which the conductance should be set
    :param value: Conductance between 0 and 3068. If the conductance is below
        1023 the low conductance mode is enabled. If it is higher than 2045,
        the high conductance mode is enabled.
    '''
    norm_value, division, multiplication = conductance_to_capmem(value)
    pop.set(multicompartment_connect_soma=False,
            multicompartment_enable_conductance=True,
            multicompartment_enable_conductance_multiplication=multiplication,
            multicompartment_enable_conductance_division=division,
            multicompartment_i_bias_nmda=norm_value)


def get_identical_attr(sample_dfs: List[pd.DataFrame], attribute: str) -> List:
    '''
    Compare the given attribute of different DataFrames and return its value
    if its identical in all DataFrames.

    Follows the annotations to the DataFrames to find the DataFrame from which
    the target was extracted.

    :param samples_dfs: List of DataFrames for which to extract the identical
        attribute.
    :param attribute: Name of the attribute to extract.
    :return: Value of the attribute which is identical in all DataFrames.
    :raises AttributeNotIdentical: If the attribute is not identical in all
        DataFrames, i.e. it does not exist in all DataFrame or differs in at
        least two.
    '''
    value = None

    for sample_df in sample_dfs:
        if attribute not in sample_df.attrs:
            raise AttributeNotIdentical()
        curr_value = sample_df.attrs[attribute]

        # set value if unset, else compare to set values
        if value is None:
            value = curr_value
        if np.any(value != curr_value):
            raise AttributeNotIdentical()

    return value


def get_chip_serial() -> Optional[int]:
    '''
    Extract chip serial from environment.

    Parse the handwritten chip serial from the environment variable
    `SLURM_HWDB_YAML`.

    :return: Handwritten chip serial if information is available in environment
        else nothing is returned.
    '''
    if 'SLURM_HARDWARE_LICENSES' not in os.environ or 'SLURM_HWDB_YAML' \
            not in os.environ:
        return None
    fpga = int(os.environ.get('SLURM_HARDWARE_LICENSES')[-1])
    hwdb = yaml.safe_load(os.environ.get('SLURM_HWDB_YAML'))['fpgas']
    for fpga_data in hwdb:
        if fpga_data['fpga'] == fpga:
            return fpga_data['handwritten_chip_serial']
    return None


def get_license_and_chip() -> str:
    '''
    Create a string which uniquely describes a BSS-2 chip and the used FPGA.

    :return: String 'W{wafer}F{fpga}C{chip_serial}' if the information is
        available in the environment, else an empty string.
    '''
    if 'SLURM_HARDWARE_LICENSES' not in os.environ or 'SLURM_HWDB_YAML' \
            not in os.environ:
        return ''
    return f"{os.environ.get('SLURM_HARDWARE_LICENSES')}C{get_chip_serial()}"


def grid_search(chain_experiment: AttenuationExperiment,
                g_leak: Tuple[float, float, float],
                g_icc: Tuple[float, float, float],
                ) -> pd.DataFrame:
    '''
    Perform a grid search of the axial and inter-compartment conductance for
    an AttenuationExperiment on BSS-2.

    :param chain_experiment: Attenuation Experiment.
    :param g_leak: Tuple (min, max, steps) used for the leak conductance
        during the grid search. The number of steps is converted to an integer.
    :param g_leak: Tuple (min, max, steps) used for the inter-compartment
        conductance during the grid search. The number of steps is converted to
        an integer.
    :returns: DataFrame filled with tested parameters and measured amplitudes.
    '''
    # Determine measurement points
    g_leak_icc = np.array(list(product(
        np.linspace(g_leak[0], g_leak[1], int(g_leak[-1])),
        np.linspace(g_icc[0], g_icc[1], int(g_icc[-1])))))

    # Create DataFrame, fill with parameter values and add meta data
    length = chain_experiment.length
    param_col = pd.MultiIndex.from_product([['parameters'],
                                            ['g_leak', 'g_icc']])
    cols = [f"A_{i}{j}" for i, j in product(range(length), range(length))]
    amp_col = pd.MultiIndex.from_product([['amplitudes'], cols])
    columns = pd.MultiIndex.from_tuples(list(param_col) + list(amp_col))

    result = pd.DataFrame(np.zeros([len(g_leak_icc), 2 + length**2]),
                          columns=columns)
    result['parameters'] = g_leak_icc

    result.attrs['length'] = length
    result.attrs['date'] = str(datetime.now())

    # Perform experiment
    for row in range(len(result)):
        g_leak, g_icc = result.loc[row, 'parameters']
        res = chain_experiment.measure_response(np.array([g_leak, g_icc]))
        result.loc[row, 'amplitudes'] = res.flatten()

    return result


def record_variations(experiment: AttenuationExperiment,
                      repetitions: int) -> pd.DataFrame:
    '''
    Repeat the same attenuation experiment several times and log the recorded
    PSP amplitudes in a DataFrame.

    :param experiment: Attenuation Experiment.
    :param repetitions: Number of times the experiment should be repeated.
    :returns: DataFrame with PSP amplitudes for each repetition.
    '''

    results = []
    for _ in range(repetitions):
        result = experiment.measure_response()
        results.append(result.flatten())

    result = pd.DataFrame(np.array(results))
    result.attrs['length'] = experiment.length
    result.attrs['date'] = str(datetime.now())

    return result


def extract_original_parameters(dfs: Sequence[pd.DataFrame]) -> np.ndarray:
    '''
    Extract the original parameters from the given DataFrames with samples.

    Try to extract the parameters which were used to measure the target on
    which the posteriors were conditioned.

    :param dfs: DataFrames with samples drawn during the approximation or drawn
        from the posteriors. From their attributes the initial target is
        extracted and from the target the parameter.
    :raises RuntimeError: If the original parameters can not be extracted.
    '''
    try:
        target_dfs = [pd.read_pickle(df.attrs['target_file']) for df in dfs]
    except (KeyError, FileNotFoundError) as err:
        raise RuntimeError('Target can not be extracted for all '
                           'posterior files.') from err
    try:
        return get_identical_attr(target_dfs, 'parameters')
    except AttributeNotIdentical as err:
        raise RuntimeError('Posterior files do not have a common target.'
                           ) from err

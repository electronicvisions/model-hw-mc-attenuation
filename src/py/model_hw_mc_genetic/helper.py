from typing import List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import yaml
from pyNN.common import BasePopulation


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

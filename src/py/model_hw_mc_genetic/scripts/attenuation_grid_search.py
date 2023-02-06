#!/usr/bin/env python3
'''
Record all samples in a grid of parameter values and save height as well as
attenuation in a DataFrame.
'''
from itertools import product
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import quantities as pq

from model_hw_si_nsc_dendrites.helper import get_license_and_chip

from model_hw_mc_genetic.attenuation.base import Base as AttenuationExperiment

from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment as \
    AttenuationBSS
from model_hw_mc_genetic.attenuation.arbor import \
    AttenuationExperiment as AttenuationArbor


def main(chain_experiment: AttenuationExperiment, g_leak_icc: np.ndarray,
         ) -> pd.DataFrame:
    '''
    Perform a grid search of the axial and inter-compartment conductance for
    an AttenuationExperiment on BSS-2.

    :param chain_experiment: Attenuation Experiment.
    :param g_leak_icc: Array of leak and inter-compartment conductances to
        test.
    :returns: DataFrame filled with tested parameters and measured EPSP height.
    '''
    length = chain_experiment.length
    # Create DataFrame, fill with parameter values and add meta data
    param_col = pd.MultiIndex.from_product([['parameters'],
                                            ['g_leak', 'g_icc']])
    psp_col = pd.MultiIndex.from_product([['psp_heights'],
                                          np.arange(length**2)])
    columns = pd.MultiIndex.from_tuples(list(param_col) + list(psp_col))

    result = pd.DataFrame(np.zeros([len(g_leak_icc), 2 + length**2]),
                          columns=columns)
    result['parameters'] = g_leak_icc

    result.attrs['length'] = length
    result.attrs['date'] = str(datetime.now())

    # Perform experiment
    for row in range(len(result)):
        g_leak, g_icc = result.loc[row, 'parameters']
        res = chain_experiment.measure_response(np.array([g_leak, g_icc]))
        result.loc[row, 'psp_heights'] = res.flatten()

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform the chain attenuation experiment with parameters '
                    'located on a grid in the two-dimensional parameter '
                    'space. Leak and inter-compartment conductance are set '
                    'globally for all compartments/conductances.')
    parser.add_argument("-mode",
                        help="Perform the experiment on BSS-2 or in arbor.",
                        type=str,
                        default='bss',
                        choices=['bss', 'arbor'])
    parser.add_argument('-calibration',
                        type=str,
                        help='Path to binary calibration. This is only needed '
                             'if the experiment is executed on BSS-2.')
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=5)
    parser.add_argument("-leak_param",
                        help="Leak conductance (CapMem values for Bss-2) or "
                             "membrane time constant (in ms for arbor). "
                             "Provide as a list with the following values: "
                             "(lower_bound, upper bound, num_steps). The "
                             "steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=float,
                        default=[10, 1000, 100])
    parser.add_argument("-icc_param",
                        help="Inter-compartment conductance (CapMem values "
                             "for Bss-2) or inter-compartment time constant "
                             "(in ms for arbor). "
                             "Provide as a list with the following values: "
                             "(lower_bound, upper bound, num_steps). The "
                             "steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=float,
                        default=[10, 1000, 100])
    parser.add_argument("-input_neurons",
                        help="Number of synchronous inputs (BSS only).",
                        type=int,
                        default=10)
    parser.add_argument("-input_weight",
                        help="Input weight for each neuron (BSS only).",
                        type=int,
                        default=30)
    args = parser.parse_args()

    if args.mode == 'arbor':
        attenuation_experiment = AttenuationArbor(length=args.length)
        g_leak_lim = attenuation_experiment.recipe.tau_mem_to_g_leak(
            [args.leak_param[1], args.leak_param[0]] * pq.ms)
        g_icc_lim = attenuation_experiment.recipe.tau_icc_to_g_axial(
            [args.icc_param[1], args.icc_param[0]] * pq.ms)
    else:
        if args.calibration is None:
            raise RuntimeError('You are performing the experiment on BSS- 2 '
                               'but you did not provide a calibration.')
        attenuation_experiment = AttenuationBSS(
            Path(args.calibration), length=args.length,
            input_weight=args.input_weight, input_neurons=args.input_neurons)
        g_leak_lim = [args.leak_param[0], args.leak_param[1]]
        g_icc_lim = [args.icc_param[0], args.icc_param[1]]

    parameters = np.array(list(product(
        np.linspace(g_leak_lim[0], g_leak_lim[1], int(args.leak_param[-1])),
        np.linspace(g_icc_lim[0], g_icc_lim[1], int(args.icc_param[-1])))))

    data = main(attenuation_experiment, parameters)
    if args.mode == 'bss':
        data.attrs['calibration'] = str(Path(args.calibration).resolve())
        data.attrs['license'] = get_license_and_chip()
        data.attrs['input_neurons'] = args.input_neurons
        data.attrs['input_weight'] = args.input_weight
    elif args.mode == 'arbor':
        data.attrs['license'] = 'arbor'

    data.to_pickle('attenuation_grid_search.pkl')

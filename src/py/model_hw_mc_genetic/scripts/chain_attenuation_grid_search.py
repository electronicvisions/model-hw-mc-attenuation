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

from model_hw_si_nsc_dendrites.helper import get_license_and_chip

from model_hw_mc_genetic.compartment_chain import AttenuationExperiment

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform the chain attenuation experiment with parameters '
                    'located on a grid in the two-dimensional parameter '
                    'space. Leak and inter-compartment conductance are set '
                    'globally for all compartments/conductances.')
    parser.add_argument('calibration',
                        type=str,
                        help='Path to binary calibration')
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=5)
    parser.add_argument("-g_leak",
                        help="Leak conductance. Provide as a list with the "
                             "following values: (lower_bound, upper bound, "
                             "num_steps). The steps will be distributed "
                             "evenly over the parameter space.",
                        nargs=3,
                        type=int,
                        default=[10, 1000, 100])
    parser.add_argument("-g_icc",
                        help="Conductance between compartments. Provide as a "
                             "list with the following values: (lower_bound, "
                             "upper bound, num_steps). The steps will be "
                             "distributed evenly over the parameter space.",
                        nargs=3,
                        type=int,
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

    # Experiment
    chain_experiment = AttenuationExperiment(Path(args.calibration),
                                             args.length,
                                             input_weight=args.input_weight,
                                             input_neurons=args.input_neurons)

    # Create DataFrame, fill with parameter values and add meta data
    param_col = pd.MultiIndex.from_product([['parameters'],
                                            ['g_leak', 'g_icc']])
    psp_col = pd.MultiIndex.from_product([['psp_heights'],
                                          np.arange(args.length**2)])
    columns = pd.MultiIndex.from_tuples(list(param_col) + list(psp_col))
    parameters = np.array(list(product(np.linspace(*args.g_leak),
                                       np.linspace(*args.g_icc))))

    result = pd.DataFrame(np.zeros([len(parameters), 2 + args.length**2]),
                          columns=columns)
    result['parameters'] = parameters

    result.attrs['calibration'] = str(Path(args.calibration).resolve())
    result.attrs['length'] = args.length
    result.attrs['date'] = str(datetime.now())
    result.attrs['license'] = get_license_and_chip()
    result.attrs['input_neurons'] = args.input_neurons
    result.attrs['input_weight'] = args.input_weight

    # Perform experiment
    for row in range(len(result)):
        g_leak, g_icc = result.loc[row, 'parameters']
        data = chain_experiment.measure_response(np.array([g_leak, g_icc]))
        result.loc[row, 'psp_heights'] = data.flatten()

    result.to_pickle('chain_attenuation_grid_search.pkl')

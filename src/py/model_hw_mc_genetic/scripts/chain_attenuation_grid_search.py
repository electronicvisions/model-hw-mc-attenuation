#!/usr/bin/env python3
'''
Record all samples in a grid of parameter values and save height as well as
attenuation in a DataFrame.
'''
from itertools import product
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

from model_hw_mc_genetic.compartment_chain import AttenuationExperiment, \
    extract_psp_heights, fit_length_constant

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
    args = parser.parse_args()

    # Experiment
    chain_experiment = AttenuationExperiment(Path(args.calibration),
                                             args.length)

    results = defaultdict(list)
    for g_leak, g_icc in product(np.linspace(*args.g_leak),
                                 np.linspace(*args.g_icc)):
        chain_experiment.set_parameters(np.array([g_leak, g_icc]))
        data = chain_experiment.measure_result()

        results['g_leak'].append(g_leak)
        results['g_icc'].append(g_icc)
        results['heights'].append(extract_psp_heights(data))
        results['length_constant'].append(fit_length_constant(data)[0])

    df = pd.DataFrame(results)
    df.to_pickle('chain_attenuation_grid_search.pkl')

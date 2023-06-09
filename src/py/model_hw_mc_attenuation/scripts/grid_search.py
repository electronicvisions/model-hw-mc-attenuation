#!/usr/bin/env python3
'''
Record all samples in a grid of parameter values and save height as well as
attenuation in a DataFrame.
'''
from pathlib import Path

from pynn_brainscales.brainscales2.helper import nightly_calib_path

from model_hw_mc_attenuation.helper import get_license_and_chip, grid_search

from model_hw_mc_attenuation.bss import AttenuationExperiment, \
    default_conductance_limits, add_bss_psp_args


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform the chain attenuation experiment with parameters '
                    'located on a grid in the two-dimensional parameter '
                    'space. Leak and inter-compartment conductance are set '
                    'globally for all compartments/conductances. The '
                    'experiment is emulated on BSS-2.')
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=5)
    parser.add_argument("-g_leak",
                        help="Leak conductance (in CapMem values). "
                             "Provide as a list with the following values: "
                             "(lower_bound, upper bound, num_steps). The "
                             "steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=float,
                        default=default_conductance_limits[0].tolist() + [10])
    parser.add_argument("-g_icc",
                        help="Inter-compartment conductance (in CapMem "
                             "values). Provide as a list with the following "
                             "values: (lower_bound, upper bound, num_steps). "
                             "The steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=float,
                        default=default_conductance_limits[1].tolist() + [10])
    add_bss_psp_args(parser)
    args = parser.parse_args()

    if args.calibration is None:
        calibration = Path(nightly_calib_path())
    else:
        calibration = Path(args.calibration)
    attenuation_experiment = AttenuationExperiment(
        calibration, length=args.length,
        input_weight=args.input_weight, input_neurons=args.input_neurons,
        n_average=args.n_average)

    data = grid_search(attenuation_experiment, args.g_leak, args.g_icc)

    data.attrs['calibration'] = str(calibration.resolve())
    data.attrs['chip_id'] = get_license_and_chip()
    data.attrs['input_neurons'] = args.input_neurons
    data.attrs['input_weight'] = args.input_weight
    data.attrs['n_average'] = args.n_average
    data.attrs['experiment'] = 'attenuation_bss'

    data.to_pickle('attenuation_grid_search.pkl')

#!/usr/bin/env python3
'''
Record all samples in a grid of parameter values and save height as well as
attenuation in a DataFrame.
'''
import quantities as pq

from model_hw_mc_attenuation.helper import grid_search

from model_hw_mc_attenuation.arbor import AttenuationExperiment, \
    default_tau_limits


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform the chain attenuation experiment with parameters '
                    'located on a grid in the two-dimensional parameter '
                    'space. Leak and inter-compartment conductance are set '
                    'globally for all compartments/conductances. The '
                    'experiment is simulated in arbor.')
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=5)
    parser.add_argument("-tau_mem",
                        help="Membrane time constant (in ms). "
                             "Provide as a list with the following values: "
                             "(lower_bound, upper bound, num_steps). The "
                             "steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=float,
                        default=default_tau_limits[0].tolist() + [100])
    parser.add_argument("-tau_icc",
                        help="Inter-compartment time constant (in ms). "
                             "Provide as a list with the following values: "
                             "(lower_bound, upper bound, num_steps). The "
                             "steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=float,
                        default=default_tau_limits[1].tolist() + [100])
    args = parser.parse_args()

    attenuation_experiment = AttenuationExperiment(length=args.length)

    g_leak_lim = attenuation_experiment.recipe.tau_mem_to_g_leak(
        [args.tau_mem[1], args.tau_mem[0]] * pq.ms)
    g_icc_lim = attenuation_experiment.recipe.tau_icc_to_g_axial(
        [args.tau_icc[1], args.tau_icc[0]] * pq.ms)

    data = grid_search(attenuation_experiment,
                       (g_leak_lim[0], g_leak_lim[1], int(args.tau_mem[-1])),
                       (g_icc_lim[0], g_icc_lim[1], int(args.tau_icc[-1]))
                       )
    data.attrs['experiment'] = 'attenuation_arbor'

    data.to_pickle('attenuation_grid_search.pkl')

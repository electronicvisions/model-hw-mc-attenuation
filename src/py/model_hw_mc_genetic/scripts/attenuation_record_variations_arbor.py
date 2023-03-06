#!/usr/bin/env python3
import argparse
from typing import Optional, Tuple

import pandas as pd

from model_hw_mc_genetic.attenuation.arbor import AttenuationExperiment
from model_hw_mc_genetic.attenuation.helper import record_variations


def main(length: int, repetitions: int,
         parameters: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    '''
    Repeat the same attenuation experiment several times in arbor and log the
    recorded PSP amplitudes in a DataFrame.

    :param length: Number of compartments in the chain.
    :param repetitions: Number of times the experiment should be repeated.
    :param parameters: Leak and inter-compartment conductance.
    :returns: DataFrame with PSP amplitudes for each repetition.
    '''

    experiment = AttenuationExperiment(length)
    params = experiment.time_constants_to_conductance(parameters)
    experiment.set_parameters(params)

    result = record_variations(experiment, repetitions)
    result.attrs['parameters'] = params
    result.attrs['experiment'] = 'attenuation_arbor'

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform the ChainAttenuation experiment several times '
                    'and save the PSP heights in a pickled DataFrame.')
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=4)
    parser.add_argument("-n_repetitions",
                        help="Number of times the experiment is performed.",
                        type=int,
                        default=100)
    parser.add_argument("-parameters",
                        help="Leak and inter-compartment time constant "
                             "(in ms).",
                        nargs=2,
                        type=float)
    args = parser.parse_args()

    data = main(args.length, args.n_repetitions, parameters=args.parameters)
    data.to_pickle('attenuation_variations.pkl')

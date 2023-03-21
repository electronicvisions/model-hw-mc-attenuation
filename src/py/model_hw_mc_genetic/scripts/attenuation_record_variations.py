#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from pynn_brainscales.brainscales2.helper import nightly_calib_path
from model_hw_mc_genetic.helper import get_license_and_chip

from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment, \
    add_bss_psp_args
from model_hw_mc_genetic.attenuation.helper import record_variations


def main(length: int, repetitions: int,
         parameters: Optional[Tuple[float, float]] = None, *,
         input_neurons: int = 10,
         input_weight: int = 30,
         n_average: int = 1,
         calibration: Optional[str] = None
         ) -> pd.DataFrame:
    '''
    Repeat the same attenuation experiment several times on BSS-2 and log the
    recorded PSP amplitudes in a DataFrame.

    :param length: Number of compartments in the chain.
    :param repetitions: Number of times the experiment should be repeated.
    :param parameters: Leak and inter-compartment conductance.
    :param input_neurons: Number of synchronous inputs.
    :param input_weight: Synaptic weight of each input.
    :param calibration: Path to portable binary calibration. If not provided
        the latest nightly calibration is used.
    :returns: DataFrame with PSP amplitudes for each repetition.
    '''
    calibration = Path(nightly_calib_path() if calibration is None
                       else calibration)

    experiment = AttenuationExperiment(calibration,
                                       length=length,
                                       input_neurons=input_neurons,
                                       input_weight=input_weight,
                                       n_average=n_average)

    params = experiment.expand_parameters(parameters)
    experiment.set_parameters(params)

    result = record_variations(experiment, repetitions)
    result.attrs['chip_id'] = get_license_and_chip()
    result.attrs['parameters'] = params
    result.attrs['calibration'] = str(calibration.resolve())
    result.attrs['input_neurons'] = input_neurons
    result.attrs['input_weight'] = input_weight
    result.attrs['n_average'] = n_average
    result.attrs['experiment'] = 'attenuation_bss'

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform the ChainAttenuation experiment several times '
                    'and save the PSP heights in a pickled DataFrame.')
    parser.add_argument("-mode",
                        help="Perform the experiment on BSS-2 or in arbor.",
                        type=str,
                        default='bss',
                        choices=['bss', 'arbor'])
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=4)
    parser.add_argument("-n_repetitions",
                        help="Number of times the experiment is performed.",
                        type=int,
                        default=100)
    parser.add_argument("-parameters",
                        help="Leak/inter-compartment conductance (CapMem "
                             "values). The first value is for the leak, the "
                             "second for the inter-compartment conductance.",
                        nargs=2,
                        type=float)
    add_bss_psp_args(parser)
    args = parser.parse_args()

    data = main(args.length, args.n_repetitions, args.parameters,
                input_neurons=args.input_neurons,
                input_weight=args.input_weight,
                calibration=args.calibration,
                n_average=args.n_average
                )
    data.to_pickle('attenuation_variations.pkl')

#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional, Tuple

import neo

from pynn_brainscales.brainscales2.helper import nightly_calib_path

from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment, \
    expand_params, add_bss_psp_args


def main(length: int,
         parameters: Optional[Tuple[float, float]] = None, *,
         input_neurons: Optional[int] = 10,
         input_weight: Optional[int] = 30,
         calibration: Optional[str] = None
         ) -> neo.Block():
    '''
    Perform an attenuation experiment on BSS-2 and record the membrane traces.

    :param length: Number of compartments in the chain.
    :param parameters: Leak and inter-compartment conductance.
    :param input_neurons: Number of synchronous inputs.
    :param input_weight: Synaptic weight of each input.
    :param calibration: Path to portable binary calibration. If not provided
        the latest nightly calibration is used.
    :returns: Block of recorded membrane traces.
    '''
    calibration = Path(nightly_calib_path() if calibration is None
                       else calibration)

    experiment = AttenuationExperiment(calibration,
                                       length=length,
                                       input_neurons=input_neurons,
                                       input_weight=input_weight)

    return experiment.record_data(expand_params(experiment, parameters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform the ChainAttenuation experiment and record '
                    'the membrane potentials. The recorded membrane traces '
                    'are saved in a pickled `neo.Block`.')
    parser.add_argument("-mode",
                        help="Perform the experiment on BSS-2 or in arbor.",
                        type=str,
                        default='bss',
                        choices=['bss', 'arbor'])
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=4)
    parser.add_argument("-parameters",
                        help="Leak and inter-compartment conductance (in "
                             "CapMem values).",
                        nargs=2,
                        type=float)
    add_bss_psp_args(parser)
    args = parser.parse_args()

    result = main(args.length,
                  parameters=args.parameters,
                  input_neurons=args.input_neurons,
                  input_weight=args.input_weight,
                  calibration=args.calibration)
    neo.PickleIO('membrane_traces.pkl').write(result)
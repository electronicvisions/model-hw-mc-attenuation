#!/usr/bin/env python3

import argparse
from typing import Optional, Tuple
import quantities as pq
import neo

from model_hw_mc_attenuation.arbor import AttenuationExperiment


def main(length: int,
         parameters: Optional[Tuple[float, float]] = None) -> neo.Block():
    '''
    Simulate an attenuation experiment in arbor and record the membrane traces.

    :param length: Number of compartments in the chain.
    :param parameters: Membrane and inter-compartment time constants. If not
        provided, the default parameters are used.
    :returns: Block of recorded membrane traces.
    '''
    experiment = AttenuationExperiment(length)

    params = experiment.time_constants_to_conductance(
        None if parameters is None else parameters * pq.ms)

    return experiment.record_data(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform the ChainAttenuation experiment and record '
                    'the membrane potentials. The recorded membrane traces '
                    'are saved in a pickled `neo.Block`.')
    parser.add_argument("-length",
                        help="Length of compartment chain.",
                        type=int,
                        default=4)
    parser.add_argument("-parameters",
                        help="Membrane and inter-compartment time constant "
                             "(in ms).",
                        nargs=2,
                        type=float)
    args = parser.parse_args()

    result = main(args.length, parameters=args.parameters)
    neo.PickleIO('membrane_traces.pkl').write(result)

#!/usr/bin/env python3
from typing import List
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from model_hw_mc_genetic.helper import concat_dfs


def draw_samples(posteriors: List,
                 abc_samples: pd.DataFrame,
                 posterior_idx: int = -1,
                 n_samples: int = 10000):
    '''
    Return DataFrame with samples drawn from a posterior.

    :param posteriors: List of approximated posteriors. From which samples can
        be drawn.
    :param abc_samples: DataFrame which contains samples drawn during ABC.
        The meta data such as parameter names are extracted from this
        DataFrame.
    :param posterior_idx: Index of posterior from which to draw the samples.
    :param n_samples: Number of samples to draw from the posterior.
    :returns: DataFrame with samples drawn from posterior.
    '''
    samples = posteriors[posterior_idx].sample((n_samples,)).numpy()

    columns = pd.MultiIndex.from_product(
        [['parameters'], abc_samples['parameters'].columns])
    samples = pd.DataFrame(samples, columns=columns)

    # meta data
    samples.attrs['posterior_idx'] = np.arange(len(posteriors))[posterior_idx]

    samples.attrs['length'] = abc_samples.attrs['length']
    samples.attrs['limits'] = abc_samples.attrs['limits']
    samples.attrs['target_file'] = abc_samples.attrs['target_file']
    samples.attrs['target'] = abc_samples.attrs['target']
    samples.attrs['observation'] = abc_samples.attrs['observation']

    return samples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Draw samples from the provided posterior. The samples "
                    "will be drawn from the posterior specified by the "
                    "argument `index_posterior` and are saved as a pickled "
                    "DataFrame in the file 'posterior_samples_{n_pos}.pkl' "
                    "where 'n_pos' is the index of the posterior. If the "
                    "file already exists, the samples are appended.")
    parser.add_argument("posterior_file",
                        type=str,
                        help="Path to pickled file with list of posteriors.")
    parser.add_argument("abc_samples_file",
                        type=str,
                        help="Path to pickled DataFrame which contains "
                             "samples drawn during ABC. The meta data such as "
                             "parameter names are extracted from this file.")
    parser.add_argument("-n_samples",
                        type=int,
                        default=10000,
                        help="Number of samples to draw from the posterior.")
    parser.add_argument("-index_posterior",
                        type=int,
                        default=-1,
                        help="Index of posterior from which to draw the "
                             "samples.")
    args = parser.parse_args()

    with open(args.posterior_file, 'rb') as handle:
        approx_posteriors = pickle.load(handle)

    new_df = draw_samples(approx_posteriors,
                          pd.read_pickle(args.abc_samples_file),
                          args.index_posterior,
                          n_samples=args.n_samples)
    new_df.attrs['posterior_file'] = str(Path(args.posterior_file).resolve())
    new_df.attrs['abc_samples_file'] = \
        str(Path(args.abc_samples_file).resolve())

    out_file = Path(f'posterior_samples_{new_df.attrs["posterior_idx"]}.pkl')
    if out_file.exists():
        new_df = concat_dfs(pd.read_pickle(out_file), new_df)

    new_df.to_pickle(out_file)

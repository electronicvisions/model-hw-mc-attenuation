'''
Custom mutation operators.
'''
from typing import Tuple, Sequence

import numpy as np


# pylint:disable=invalid-name  # adhering to upstream deap naming
def mutCustomBitFlip(individual: Sequence, low: np.ndarray, up: np.ndarray,
                     indpb: float) -> Tuple[int]:
    '''
    Mutate the provided individual by randomly adding/subtracting a power of
    two to/from a gene.

    Each gene of the individual will be mutated with probability `indpb`.
    A mutation randomly adds (or subtracts) 2^x to (or from) the current value
    of the gene, where x is drawn uniformly from [0, :math:`log_2`(low - up)].
    If a new generated value falls outside the boundaries the sample is
    rejected and a new one is drawn until it resides within the boundaries.

    :param individual: Individual which will be mutated.
    :param low: Lower boundary accepted as value for the respective gene.
    :param up: Upper boundary accepted as value for the repsective gene.
    :param indpb: Probability with which a gene is mutated.
    :returns: Mutated individual.
    '''
    rng = np.random.default_rng()

    for gene_idx, gene in enumerate(individual):
        if rng.random() <= indpb:
            new_gene = np.inf
            while new_gene > up[gene_idx] or new_gene < low[gene_idx]:
                # Upper integer bound is excluded therefore add 1.
                random_int = rng.integers(
                    low=0,
                    high=int(np.log2(up[gene_idx] - low[gene_idx])) + 1)
                shift = (1 << random_int)
                sign = rng.integers(low=0, high=2) * 2 - 1
                new_gene = gene + sign * shift
            individual[gene_idx] = int(new_gene)
    return (individual,)

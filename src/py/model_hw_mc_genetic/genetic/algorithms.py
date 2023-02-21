'''
    This file contains code modified from DEAP and is therefore, like DEAP,
    published under the GNU Lesser General Public License as published by the
    Free Software Foundation version 3 of the License.

    This code is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
    for more details, <http://www.gnu.org/licenses/>.

The function `ea_elite` is based on :func:`deap.algorithms.eaSimple` [1]_.

.. [1] https://github.com/DEAP/deap/blob/1.3.0/deap/algorithms.py
'''
from typing import Sequence, Optional

from deap import base, tools
from deap.algorithms import eaSimple
from deap.algorithms import varAnd, eaMuPlusLambda


# This is a modification of deap's eaSimple incorparating an elitism mechanism
def ea_elite(population: Sequence, toolbox: base.Toolbox, *,
             cxpb: float, mutpb: float, ngen: int, n_elites: int,
             stats: Optional[tools.Statistics] = None,
             verbose: bool = __debug__):
    """
    This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_ and extends it by an elitism
    mechanism [De75]_.

    This function is based on :func:`deap.algorithms.eaSimple` and extends it
    by an elitism mechanism. This is done by always curating a Hall of Fame
    and directly transferring these individuals to the next generation's
    population.
    Furthermore, all individuals are evaluated in each generation to counter
    fixed pattern noise which is present on analog neuromorphic hardware.
    In pseudo-code the algorithm works as follow :

    evaluate(population)
    for g in range(ngen):
        elites = get_fittest(population, n_elites)
        population = select(population, len(population) - n_elites)
        offspring = varAnd(population, toolbox, cxpb, mutpb)
        offspring.extend(elites)
        evaluate(offspring)
        population = offspring

    For further explanation see func:`deap.algorithms.eaSimple`.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
        operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param n_elites: Number of elites directly passed to the next generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
        inplace, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
        evolution

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    .. [De75] Kenneth Alan De Jong. An analysis of the behavior of a class
       of genetic adaptive systems. PhD thesis, 1975.
    """
    logbook = tools.Logbook()
    # --- begin added ---------------------------------------------------------
    logbook.header = ['gen'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    for ind, fit in zip(
            population, toolbox.map(toolbox.evaluate, population)):
        ind.fitness.values = fit

    halloffame = tools.HallOfFame(n_elites)
    halloffame.update(population)
    # --- end added -----------------------------------------------------------

    record = stats.compile(population) if stats else {}
    # --- begin modified ------------------------------------------------------
    logbook.record(gen=0, **record)
    # --- end modified --------------------------------------------------------
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        # --- begin modified --------------------------------------------------
        offspring = toolbox.select(population, len(population) - n_elites)
        # --- end modified ----------------------------------------------------

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # --- begin added -----------------------------------------------------
        # add the halloffame individuals back to population
        offspring.extend(halloffame.items)

        # Evaluate all individuals
        for ind, fit in zip(
                offspring, toolbox.map(toolbox.evaluate, offspring)):
            ind.fitness.values = fit
        # --- end added -------------------------------------------------------

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        # --- begin modified --------------------------------------------------
        logbook.record(gen=gen, **record)
        # --- end modified ----------------------------------------------------
        if verbose:
            print(logbook.stream)

    return population, logbook


algorithms = {'ea_elite': ea_elite,
              'eaSimple': eaSimple,
              'eaMuPlusLambda': eaMuPlusLambda}

"""Elitist NSGA-II for binary graph chromosomes, with injectable fitness."""
import functools

from GE.population import Population
from GE.utils import NSGA2Utils


class Evolution:
    def __init__(self, pop=50, iter=30, cross_pro=0.8, mutation_pro=0.2, matrix_size=1024,
                 tourna_size=5, index=None, index_group=None, his_len=None, pre_len=None,
                 fitness_evaluator=None, initial_genomes=None, mandatory_mask=None,
                 undirected=False, seed=None, device=None, unique_population=False):
        if iter < 0:
            raise ValueError("Generation count cannot be negative")
        self.utils = NSGA2Utils(pop=pop, iter=iter, cross_pro=cross_pro, mutation_pro=mutation_pro,
                               matrix_size=matrix_size, tourna_size=tourna_size, index=index,
                               index_group=index_group, fitness_evaluator=fitness_evaluator,
                               initial_genomes=initial_genomes, mandatory_mask=mandatory_mask,
                               undirected=undirected, seed=seed, device=device, unique_population=unique_population)
        self.population = None
        self.num_of_generations = int(iter)
        self.on_generation_finished, self.history = [], []
        self.his_len, self.pre_len = his_len, pre_len
        self.completed_generations, self.stop_reason = 0, "generation_budget"

    def register_on_new_generation(self, fun):
        self.on_generation_finished.append(fun)

    def _rank(self, population):
        self.utils.fast_nondominated_sort(population)
        for front in population.fronts:
            self.utils.calculate_crowding_distance(front)

    def evolve(self, other_adj=None):
        self.population = self.utils.create_initial_population(other_adj, self.his_len, self.pre_len)
        self.num_of_individuals = len(self.population)
        self._rank(self.population)
        for generation in range(self.num_of_generations):
            children = self.utils.create_children(self.population.population, self.his_len, self.pre_len)
            if self.utils.unique_population and not children:
                self.stop_reason = "search_space_exhausted"
                break
            combined = Population()
            combined.extend(self.population.population + children)
            if self.utils.unique_population:
                combined.population = self.utils.deduplicate(combined.population)
            self._rank(combined)
            selected = Population()
            for front in combined.fronts:
                remaining = self.num_of_individuals - len(selected)
                if remaining == 0:
                    break
                if len(front) <= remaining:
                    selected.extend(front)
                else:
                    ordered = sorted(front, key=functools.cmp_to_key(self.utils.crowding_operator))
                    selected.extend(ordered[:remaining])
                    break
            self.population = selected
            self._rank(self.population)
            self.completed_generations = generation + 1
            for callback in self.on_generation_finished:
                value, kind = callback(self.population, generation)
                if kind == "his":
                    self.history.append(value)
        # Include the last offspring and handle zero generations. No unused
        # extra generation is trained after final survivor selection.
        return list(dict.fromkeys(self.population.fronts[0]))

    def acc_best(self, fronts):
        if not fronts:
            raise ValueError("No Pareto graphs available")
        return [min(fronts, key=lambda ind: (ind.objectives[m], tuple(ind.objectives)))
                for m in range(len(fronts[0].objectives))]


def print_genetation(population, genetation):
    return {genetation: [p.objectives for p in dict.fromkeys(population.fronts[0])]}, "his"

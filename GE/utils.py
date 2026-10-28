"""NSGA-II graph operators with injectable fitness and independent search RNG."""
import random
from pathlib import Path

import numpy as np
import torch

from GE.individual import Individual
from GE.population import Population


class NSGA2Utils:
    def __init__(self, pop=30, iter=30, cross_pro=0.7, mutation_pro=0.05, matrix_size=1024,
                 tourna_size=5, index=None, index_group=None, fitness_evaluator=None,
                 initial_genomes=None, mandatory_mask=None, undirected=False, seed=None, device=None,
                 unique_population=False):
        if pop < 2 or tourna_size < 1 or matrix_size < 2:
            raise ValueError("Population >=2, tournament >=1 and matrix_size >=2 are required")
        if not (0 <= cross_pro <= 1 and 0 <= mutation_pro <= 1):
            raise ValueError("Crossover and mutation probabilities must be in [0,1]")
        self.pop_num, self.iter_num = int(pop), int(iter)
        self.cross_pro, self.mutation_pro = cross_pro, mutation_pro
        self.matrix_size, self.tourna_size = int(matrix_size), int(tourna_size)
        self.index_group, self.fitness_evaluator = index_group, fitness_evaluator
        self.undirected = bool(undirected)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        # Fitness resets global seeds for fair proxy training; GA randomness is isolated.
        seed = int(np.random.randint(0, 2**32 - 1)) if seed is None else int(seed)
        self.rng, self.py_rng = np.random.default_rng(seed), random.Random(seed)
        self.initial_genomes = initial_genomes
        self.unique_population = bool(unique_population)
        self.duplicate_rejections, self.immigrants = 0, 0
        self._legacy_indices = None
        if index is None and index_group is not None:
            first, last = map(int, index_group)
            if not 1 <= first <= last:
                raise ValueError("Invalid legacy KNN index range")
            directory = Path(__file__).resolve().parents[1] / "data/cnn_spatial_align_slide/index"
            self._legacy_indices = [np.load(directory / f"index_{k}.npy", allow_pickle=False)
                                    for k in range(first, last + 1)]
            index = self._legacy_indices[-1]
        if index is None:
            index = np.stack(np.triu_indices(matrix_size, k=1), axis=1)
        self.max_index = np.asarray(index, dtype=np.int64).copy()
        if (self.max_index.ndim != 2 or self.max_index.shape[1] != 2 or not len(self.max_index)
                or (self.max_index < 0).any() or (self.max_index >= matrix_size).any()
                or (self.max_index[:, 0] == self.max_index[:, 1]).any()
                or len(np.unique(self.max_index, axis=0)) != len(self.max_index)):
            raise ValueError("Candidate edge index must contain unique non-self edges in range")
        if undirected and not (self.max_index[:, 0] < self.max_index[:, 1]).all():
            raise ValueError("Undirected genes must encode upper-triangle edges only")
        self.max_index.setflags(write=False)
        self.lens = len(self.max_index)
        self.mandatory_mask = (np.zeros(self.lens, dtype=bool) if mandatory_mask is None
                               else np.asarray(mandatory_mask, dtype=bool).copy())
        if self.mandatory_mask.shape != (self.lens,):
            raise ValueError("mandatory_mask must have one bit per candidate edge")
        self._fitness_cache = {}
        self.fitness_evaluations, self.cache_hits = 0, 0
        self.Fitness, self.hist_fit = [], []
        # Sequential: one GPU and isolated seeds/artifacts. Old MultiPro omitted
        # his_len/pre_len and assumed three GPUs even on single-GPU hosts.
        self.use_parallel = False

    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count, individual.dominated_solutions = 0, []
            for other in population:
                if individual is other:
                    continue
                if individual.dominates(other):
                    individual.dominated_solutions.append(other)
                elif other.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while population.fronts[i]:
            following = []
            for individual in population.fronts[i]:
                for other in individual.dominated_solutions:
                    other.domination_count -= 1
                    if other.domination_count == 0:
                        other.rank = i + 1
                        following.append(other)
            population.fronts.append(following)
            i += 1

    def calculate_crowding_distance(self, front):
        for individual in front:
            individual.crowding_distance = 0.0
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float("inf")
            return
        for objective in range(len(front[0].objectives)):
            ordered = sorted(front, key=lambda individual: individual.objectives[objective])
            extent = ordered[-1].objectives[objective] - ordered[0].objectives[objective]
            if extent <= 0:
                continue
            ordered[0].crowding_distance = ordered[-1].crowding_distance = float("inf")
            for i in range(1, len(ordered) - 1):
                ordered[i].crowding_distance += ((ordered[i + 1].objectives[objective]
                                                  - ordered[i - 1].objectives[objective]) / extent)

    def crowding_operator(self, individual, other_individual):
        a = (individual.rank, -float(individual.crowding_distance or 0))
        b = (other_individual.rank, -float(other_individual.crowding_distance or 0))
        return -1 if a < b else 1 if a > b else 0

    def _individual(self, genome):
        genome = np.asarray(genome)
        if genome.shape != (self.lens,) or not np.isin(genome, (0, 1)).all():
            raise ValueError("Graph genome must be a binary vector matching candidate edges")
        individual = Individual()
        individual.compression = genome.astype(np.uint8, copy=True)
        individual.compression[self.mandatory_mask] = 1
        individual.index, individual.matrix_size = self.max_index, self.matrix_size
        individual.undirected = self.undirected
        return individual

    def individual_key(self, individual):
        signature = None
        if individual.specialty:
            projected = self._individual(individual.compression).index_to_matrix()
            if not np.array_equal(projected, individual.adj):
                signature = np.ascontiguousarray(individual.adj, dtype=np.float64).tobytes()
        return individual.compression.astype(np.uint8).tobytes(), signature

    def deduplicate(self, individuals):
        unique, seen = [], set()
        for individual in individuals:
            key = self.individual_key(individual)
            if key not in seen:
                seen.add(key)
                unique.append(individual)
            else:
                self.duplicate_rejections += 1
        return unique

    def fill_unique(self, individuals, target, excluded=()):
        """Bounded immigrant generation, including finite/exhausted spaces."""
        result, blocked = [], set(excluded)
        for individual in individuals:
            key = self.individual_key(individual)
            if key not in blocked and len(result) < target:
                result.append(individual)
                blocked.add(key)
        optional = np.flatnonzero(~self.mandatory_mask)
        # Random immigrants preserve exploration without retrying forever.
        for _ in range(8 * target if len(optional) else 0):
            if len(result) >= target:
                break
            genome = (self.rng.random(self.lens) < self.rng.uniform(.1, .9)).astype(np.uint8)
            individual = self._individual(genome)
            key = self.individual_key(individual)
            if key not in blocked:
                result.append(individual)
                blocked.add(key)
                self.immigrants += 1
        # Among blocked+target assignments at least target are available unless
        # the binary space is exhausted. Enumerate only this bounded prefix.
        limit = min(1 << min(len(optional), 62), len(blocked) + target)
        for code in range(limit):
            if len(result) >= target:
                break
            genome = self.mandatory_mask.astype(np.uint8)
            for bit in range(min(len(optional), code.bit_length())):
                genome[optional[bit]] = (code >> bit) & 1
            individual = self._individual(genome)
            key = self.individual_key(individual)
            if key not in blocked:
                result.append(individual)
                blocked.add(key)
                self.immigrants += 1
        return result

    def create_initial_population(self, other_adj=None, his_len=None, pre_len=None):
        genomes = [] if self.initial_genomes is None else list(self.initial_genomes)
        specialty_graphs = []
        if self._legacy_indices is not None and self.initial_genomes is None:
            lookup = {tuple(edge): i for i, edge in enumerate(self.max_index)}
            for edges in self._legacy_indices:
                genome = np.zeros(self.lens, dtype=np.uint8)
                for edge in edges:
                    genome[lookup[tuple(edge)]] = 1
                genomes.append(genome)
            self.pop_num = len(genomes)
        if other_adj is not None:
            for adjacency in other_adj:
                adjacency = np.asarray(adjacency)
                if adjacency.shape != (self.matrix_size, self.matrix_size):
                    raise ValueError("Seed graph has the wrong node count")
                adjacency = adjacency.copy()
                np.fill_diagonal(adjacency, 0)
                if not np.isfinite(adjacency).all() or (adjacency < 0).any():
                    raise ValueError("Seed graph must be finite and nonnegative")
                if self.undirected and not np.array_equal(adjacency, adjacency.T):
                    raise ValueError("An undirected seed graph must be symmetric")
                fixed = self.max_index[self.mandatory_mask]
                adjacency[fixed[:, 0], fixed[:, 1]] = 1
                if self.undirected:
                    adjacency[fixed[:, 1], fixed[:, 0]] = 1
                individual = self._individual((adjacency[self.max_index[:, 0], self.max_index[:, 1]] > 0).astype(np.uint8))
                individual.specialty, individual.adj = True, adjacency
                specialty_graphs.append(individual)
            self.pop_num = max(self.pop_num, len(genomes) + len(specialty_graphs))
        while len(genomes) + len(specialty_graphs) < self.pop_num:
            genomes.append((self.rng.random(self.lens) < self.rng.uniform(.1, .9)).astype(np.uint8))
        population = Population()
        individuals = [self._individual(g) for g in genomes[:self.pop_num - len(specialty_graphs)]] + specialty_graphs
        if self.unique_population:
            individuals = self.fill_unique(self.deduplicate(individuals), self.pop_num)
        for individual in individuals:
            population.population.append(self.calculate_objectives(individual, self.device, his_len, pre_len))
        return population

    def calculate_objectives(self, individual, device=None, his_len=None, pre_len=None):
        genome = self.individual_key(individual)
        if genome in self._fitness_cache:
            self.cache_hits += 1
            individual.objectives, individual.model_parameter = self._fitness_cache[genome]
            individual.objectives = list(individual.objectives)
            return individual
        adjacency = individual.index_to_matrix() + np.eye(self.matrix_size)
        evaluator = self.fitness_evaluator
        if evaluator is None:
            from tester_cnn import fitness_cal
            evaluator = fitness_cal
        fit, model = evaluator(adjmat=adjacency, device=device or self.device, his_len=his_len, pre_len=pre_len)
        if not np.isfinite(fit):
            raise ValueError("Candidate graph produced a nonfinite validation fitness")
        state = None if model is None else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        individual.model_parameter = state
        sparse = int(individual.get_sparse()) if individual.undirected else float(individual.get_sparse())
        individual.objectives = [float(fit), sparse]
        self._fitness_cache[genome] = (list(individual.objectives), state)
        self.fitness_evaluations += 1
        return individual

    def create_children(self, population, his_len=None, pre_len=None):
        if not population:
            return []
        if self.unique_population:
            children, blocked = [], set(self._fitness_cache)
            for _ in range(8 * len(population)):
                if len(children) >= len(population):
                    break
                a, b = self.__tournament(population), self.__tournament(population)
                for child in self.__crossover(a, b):
                    child = self.__mutate(child)
                    key = self.individual_key(child)
                    if key in blocked:
                        self.duplicate_rejections += 1
                    elif len(children) < len(population):
                        children.append(child)
                        blocked.add(key)
            children = self.fill_unique(children, len(population), self._fitness_cache)
            return [self.calculate_objectives(child, self.device, his_len, pre_len) for child in children]
        children = []
        # Equal genotypes are permitted: requiring distinct parents hangs once
        # evolution has converged. Mutation can still introduce new offspring.
        while len(children) < len(population):
            parent1, parent2 = self.__tournament(population), self.__tournament(population)
            for child in self.__crossover(parent1, parent2):
                children.append(self.__mutate(child))
                if len(children) == len(population):
                    break
        return [self.calculate_objectives(child, self.device, his_len, pre_len) for child in children]

    def __crossover(self, individual1, individual2):
        child1, child2 = self._individual(individual1.compression), self._individual(individual2.compression)
        if self.lens > 1 and self.rng.random() < self.cross_pro:
            point = int(self.rng.integers(1, self.lens))
            child1.compression[point:] = individual2.compression[point:]
            child2.compression[point:] = individual1.compression[point:]
        return child1, child2

    def __mutate(self, child):
        optional = np.flatnonzero(~self.mandatory_mask)
        if len(optional) and self.rng.random() < self.mutation_pro:
            count = max(1, int(self.mutation_pro * len(optional)))
            selected = self.rng.choice(optional, count, replace=False)
            child.compression[selected] = 1 - child.compression[selected]
        child.compression[self.mandatory_mask] = 1
        return child

    def __tournament(self, population):
        if not population:
            raise ValueError("Cannot select a parent from an empty population")
        participants = self.py_rng.sample(population, min(self.tourna_size, len(population)))
        return min(participants)

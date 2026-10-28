#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

class Individual:
    def __init__(self):
        self.rank = 0
        self.crowding_distance = None
        self.dominated_solutions = []
        self.domination_count = 0
        self.adj = None
        self.model_parameter = None
        self.compression = None
        self.index = None
        self.matrix_size = 0
        self.objectives = None
        self.specialty = False
        self.undirected = False

    def __lt__(self, other):
        # rank 小的更好
        if getattr(self, "rank", None) != getattr(other, "rank", None):
            return self.rank < other.rank

        # rank 相同时 crowding_distance 大的更好
        return float(self.crowding_distance or 0) > float(other.crowding_distance or 0)

    def index_to_matrix(self):
        matrix = np.zeros((self.matrix_size, self.matrix_size))
        if self.specialty:
            return self.adj
        #print('index: ', self.index)
        for i in range(len(self.index)):
            matrix[self.index[i][0]][self.index[i][1]] = self.compression[i]
        if self.undirected:
            matrix = np.maximum(matrix, matrix.T)
        self.adj = matrix
        return matrix

    def matrix_to_index(self):
        self.compression = np.zeros(len(self.index))
        for i in range(len(self.index)):
            self.compression[i] = self.adj[self.index[i][0]][self.index[i][1]]
        return self.compression

    def get_sparse(self):
        if self.specialty:
            sparse = (np.count_nonzero(np.triu(self.adj, k=1)) if self.undirected else np.sum(self.adj))
        else:
            sparse = np.sum(self.compression)
        return sparse

    def set_objectives(self, objectives):
        self.objectives = objectives
        return objectives


    def dominates(self, other):
        own, theirs = np.asarray(self.objectives), np.asarray(other.objectives)
        return bool((own <= theirs).all() and (own < theirs).any())

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if self.compression is None or other.compression is None:
            return self is other
        return (self.matrix_size == other.matrix_size and self.undirected == other.undirected
                and self.specialty == other.specialty and np.array_equal(self.index, other.index)
                and np.array_equal(self.compression, other.compression)
                and (not self.specialty or np.array_equal(self.adj, other.adj)))

    def __hash__(self):
        if self.compression is None:
            return id(self)
        return hash((self.matrix_size, self.undirected, self.specialty, np.asarray(self.index, dtype=np.int64).tobytes(),
                     np.asarray(self.compression, dtype=np.uint8).tobytes(),
                     np.asarray(self.adj, dtype=np.float64).tobytes() if self.specialty else None))

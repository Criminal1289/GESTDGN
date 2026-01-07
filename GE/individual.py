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

    def __lt__(self, other):
        # rank 小的更好
        if getattr(self, "rank", None) != getattr(other, "rank", None):
            return self.rank < other.rank

        # rank 相同时 crowding_distance 大的更好
        return self.crowding_distance > other.crowding_distance

    def index_to_matrix(self):
        matrix = np.zeros((self.matrix_size, self.matrix_size))
        if self.specialty:
            return self.adj
        #print('index: ', self.index)
        for i in range(len(self.index)):
            matrix[self.index[i][0]][self.index[i][1]] = self.compression[i]
        self.adj = matrix
        return matrix

    def matrix_to_index(self):
        self.compression = np.zeros(len(self.index))
        for i in range(len(self.index)):
            self.compression[i] = self.adj[self.index[i][0]][self.index[i][1]]
        return self.compression

    def get_sparse(self):
        if self.specialty:
            sparse = np.sum(self.adj)
        else:
            sparse = np.sum(self.compression)
        return sparse

    def set_objectives(self, objectives):
        self.objectives = objectives
        return objectives


    def dominates(self, other):
        if (np.array(self.objectives) < np.array(other.objectives)).all():
            return True
        else:
            return False

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return np.array_equal(self.compression, other.compression)

    def __hash__(self):
        return hash(str(self.compression.tolist()))

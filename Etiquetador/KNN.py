__authors__ = ["1635892","1633686","1634264"]
__group__ = '179'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################


        # Make numpy array   
        X = np.array(train_data)
            
        # Ensure all values are float
        X = X.astype(float)
            
        # Group the data
        P, M, N = X.shape
        X = X.reshape(P, N*M)
        
        self.train_data = X
        
        

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        # Reshape test_data
        test_data = np.array(test_data)
        P, M, N = test_data.shape
        test_data = test_data.reshape(P, N*M)

        # Calculate distance between test_data and train_data
        distances = cdist(test_data, self.train_data)
        
        # Get the indices of the K nearest neighbors for each test sample
        indices = np.argsort(distances, axis=1)[:, :k]

        # Get the labels of the K nearest neighbors
        self.neighbors = self.labels[indices]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        biggest_classes = []
        
        # iterem per cada element
        for n in self.neighbors:
            dict_n = {}
            biggest = n[0]
            
            # creem un diccionari a on guardarèm el nombre de repeticions de cada
            for i in n:
                if i in dict_n.keys():
                    dict_n.update({i : dict_n[i] + 1})
                else:
                    dict_n.update({i: 1})
            
            # Mirem quin es el nombre màxim de repeticions:
            # NOTA: no podem fer aixó dins de l'altre bucle sense tindre en compte que en cas d'empat s'agafa la mès propera si ho fem 
            #       així sabem que el diccionari estarà ordenar per el que te el primer nodre mès proper (si no s'enten preguntar a Jan).
            for i in dict_n.keys():
                if dict_n[i] > dict_n[biggest]:
                    biggest = i
            
            # Gaurdem el resultat 
            biggest_classes.append(biggest)

        return np.array(biggest_classes)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()

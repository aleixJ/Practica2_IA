__authors__ = ["1635892","1633686","1634264"]
__group__ = 'nose'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        # Ensure X is a numpy array
        if type(X) is list:
            X = np.array(X)
            
        # Ensure all values are float
        X = X.astype(float)
        
        # If X has more than 2 dimensions, reshape it to 2D
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])
            
        # Assign X to self.X
        self.X = X


    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids based on the initialization method defined in self.options['km_init']
        - 'first': selects the first K points as centroids
        - 'random': selects K random points as centroids
        - 'custom': selects K points as centroids based on a costum method:
                    select the last K points as centroids
        """
        
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.centroids = []
        if self.options['km_init'].lower() == 'first':
            for elem in self.X:
                if not any(np.array_equal(elem, centroid) for centroid in self.centroids): # Check if elem is already in centroids
                    self.centroids.append(elem)
                if len(self.centroids) == self.K:
                    break

        elif self.options['km_init'].lower() == 'random':
            num_elements = self.X.shape[0]
            # Choose K random elements from X
            while len(self.centroids) < self.K:
                elem = self.X[np.random.randint(num_elements)]
                if not any(np.array_equal(elem, centroid) for centroid in self.centroids):
                    self.centroids.append(elem)

        elif self.options['km_init'].lower() == 'custom':
            for elem in self.X[::-1]:
                if not any(np.array_equal(elem, centroid) for centroid in self.centroids):
                    self.centroids.append(elem)
                if len(self.centroids) == self.K:
                    break

        self.centroids = np.array(self.centroids)
        self.old_centroids = np.array(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        # 'self.labels' (numpy array): 1xD numpy array where position i is the label of the centroid closest to the i-th point.

        # 'linalg.norm' calculates the euclidean distance between two points
        # 'argmin' returns the index of the minimum value along an axis. In this case, it returns the index of the closest centroid to each point in 'X

        self.labels = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            distances = np.linalg.norm(self.X[i] - self.centroids, axis=1)
            self.labels[i] = np.argmin(distances)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid.
        Assigns self.centroids to the new centroids and self.old_centroids to the old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        # 'self.centroids' (numpy array): KxD numpy array where position i is the coordinates of the i-th centroid.
        
        self.old_centroids = self.centroids
        self.centroids = np.zeros((self.K, self.X.shape[1]))
        for i in range(self.K):
            self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids.
        Returns True if there is no difference, False otherwise
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        return np.all(self.centroids == self.old_centroids)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid

    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist (numpy array): PxK numpy array where position (i, j) is the distance between the
        i-th point of the first set and the j-th point of the second set
    """

    # 'linalg.norm' calculates the euclidean distance between two points

    ### Option 1 ###
    # dist = np.linalg.norm(X[:, None] - C, axis=2)
    # return dist

    ### Option 2 ###
    dist = np.zeros((X.shape[0], C.shape[0]))
    for i in range(X.shape[0]):
        for j in range(C.shape[0]):
            dist[i, j] = np.linalg.norm(X[i] - C[j])
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)

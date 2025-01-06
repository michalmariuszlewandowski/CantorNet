# imports
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction


def relu(x) -> np.ndarray:
    return np.maximum(x, 0)


def get_codes(vec: np.ndarray) -> np.ndarray:
    """
    Binarizes the input vector by setting strictly positive elements to 1 and non-positive elements to 0.
    """
    return np.atleast_1d((vec > 0).astype(int))


def weights_recursion_based_representation(n_recursions: int) -> list:
    """
    Generates a list of weights for a neural network (Repr. A) with a specific number of recursions.

    This function constructs a list of weights and biases for a neural network with `n_recursions`
    number of layers. The weights and biases are defined as numpy arrays. The resulting list
    contains tuples, where the first element is a weight matrix and the second element is a bias
    vector.

    Parameters
    ----------
    n_recursions : int
        The number of recursions (layers) in the neural network for which the weights will be generated.

    Returns
    -------
    list
        A list of tuples containing the weights and biases for each layer in the neural network.
        Each tuple contains a weight matrix as the first element and a bias vector as the second element.

    """
    W_1 = np.array([[3, 0], [-3, 0], [0, 1]])
    b_1 = np.array([-2, 1, 0])
    W_2 = np.array([[1, 1, 0], [0, 0, 1]])
    # the last layer
    W_top = array_to_fractions(np.array([-1 / 2, 1]))
    b_top = array_to_fractions(np.array([-1 / 2]))

    weights = []
    for _ in range(n_recursions):
        weights.append([W_1, b_1])
        weights.append(W_2)

    weights.append([W_top, b_top])
    return weights


# weights for representation B and C

def weights_odd_elem_case(weight_array) -> np.ndarray:
    """
    For the disjunctive-normal-form-like construction. Handles the odd element case.
    """
    weight_array = np.c_[weight_array, np.zeros(weight_array.shape[0])]
    helper = np.zeros((2, weight_array.shape[1]))
    helper[0, -1] = 1
    helper[1, -1] = -1
    weight_array = np.r_[weight_array, helper]
    return weight_array


def signs_odd_elem_case(signs_array) -> np.ndarray:
    """
    For the disjunctive-normal-form-like construction. Handles the odd element case.
    """
    weight_array = np.r_[signs_array, np.zeros(signs_array.shape[1])[np.newaxis, :]]
    helper = np.zeros((2, weight_array.shape[0])).T
    helper[-1, -1] = -1
    helper[-1, -2] = 1
    weight_array = np.c_[weight_array, helper]
    return weight_array


class PrepareWeights:
    """
    A class for preparing weights of representations B and C.
    """

    def __init__(self, representation: str):
        """
        Initializes the weights for disjuntive normal form representation. Remark that we can represent the min function as
        min(a,b) = a + min(0, b-a) = a - max(0, a-b)  # referred to as representation B
        Or equivalently
        min(a,b) = 1/2 ( a-b - |a-b|) = ...  # referred to as representation C
        """
        if representation == 'B':
            self.A = np.array([[0, 1], [0, -1], [-1, 1]])
            self.S = np.array([1, -1, -1])
        elif representation == 'C':
            self.A = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]])
            self.S = array_to_fractions(np.array([1, -1, -1, -1]) / 2)
        else:
            raise ValueError(f'Unknown representation {representation}')

    def min_representation_by_relus(self, vec: np.ndarray) -> tuple:
        """
        Prepare ReLU architecture to describe the minimum function.
        """
        n = vec.T.shape[0]  # vec.shape[1] if vec.ndim == 2 else vec.shape[0]
        n += (n + 1) % 2  # add 1 if n is even else 0
        S_1 = np.kron(np.eye((n - 1) // 2), self.S)
        W_1 = np.kron(np.eye((n - 1) // 2), self.A)
        if vec.T.shape[0] % 2:
            W_1 = weights_odd_elem_case(np.squeeze(W_1))  # expand and attach 1 and -1
            S_1 = signs_odd_elem_case(S_1)

        return S_1, W_1, relu(W_1 @ vec.T)

    def output_weights(self, arr: np.ndarray) -> list:
        """
        Generates output weights for representations B and C.
        """
        arr = np.squeeze(arr)
        weights = []
        S_m_holder = None
        while arr.shape[0] > 1:  # nb of (repeated) hyperplanes
            S_m, W_m, pre_code = self.min_representation_by_relus(arr.T)
            arr = S_m @ pre_code
            # W_m is the first weight matrix by which we matmul the input
            weights.append(W_m if S_m_holder is None else W_m @ S_m_holder)
            S_m_holder = S_m  # previous value of S_m
        weights.append(self.S)
        return weights


# for creating hyperplanes of CantorNet

def get_y_coordinates(nb_recursion: int):
    """
    Get y coordinates of the ragged shape, alternating between 1 and 1/2.

    :param nb_recursion: Number of times the ragged surface will oscillate between 1 and 1/2
    :return: NumPy array with y coordinates
    """
    y_coord = np.repeat([1, 1 / 2], 2).tolist()
    y_coord = np.ravel([y_coord] * np.power(2, nb_recursion))
    y_coord = np.append(y_coord[1:], 1)
    return array_to_fractions(y_coord)


def match_coordinates(x_coord, y_coord) -> tuple:
    """
    Match x_coord with y_coord to prepare points for hyperplane calculations.

    :param x_coord: NumPy array with x coordinates
    :param y_coord: NumPy array with y coordinates
    :return: Generator yielding pairs of points
    """
    x_coord = np.ravel(x_coord)
    joint_coord = np.c_[x_coord, y_coord]
    for coord_set1, coord_set2 in zip(joint_coord[:-1], joint_coord[1:]):
        yield coord_set1, coord_set2


def cantor(line_segment=[(Fraction(0), Fraction(1))]):
    """
    Highly optimized Cantor set generator using Fraction for exact rational arithmetic.
    Attempts to minimize Fraction creation and maximizes in-loop efficiency.
    """
    one_third = Fraction(1, 3)
    # if rescaling to avoid numerical instabilities for high levels of recursions, then multiply base by 3 each recursion
    while True:
        yield line_segment
        next_segment = []
        for start, end in line_segment:
            segment_length_third = (end - start) * one_third
            next_segment.extend([
                (start, start + segment_length_third),
                (end - segment_length_third, end)
            ])
        line_segment = next_segment

def get_coord_batches(n_recursions):
    """
    Get pairs of points for determining hyperplane equations.

    :param n_recursions: Number of recursions for Cantor set
    :return: Generator yielding pairs of points
    """
    get_x_coordinates = {}
    g = cantor()

    for i in range(n_recursions + 1):
        get_x_coordinates[str(i)] = next(g)

    del get_x_coordinates['0']

    y_coord = get_y_coordinates(n_recursions - 1)
    coord_batches = match_coordinates(get_x_coordinates[str(n_recursions)], y_coord)
    return coord_batches


def get_hyperplane_equation(coord_batch) -> tuple:
    """
    Returns coefficients of a general equation of a hyperplane.

    :param coord_batch: NumPy array containing coordinates for two points
    :return: Tuple with coefficients of the hyperplane equation
    """
    x1, y1, x2, y2 = coord_batch
    assert x1 != x2, "Degenerate points, can't fit a hyperplane."
    A = Fraction(y1 - y2)
    B = Fraction(x2 - x1)
    C = Fraction(x1 * y2 - x2 * y1)
    return [A, B], C


def get_ABC_coeff(n_recursions: int) -> tuple:
    """
    Get A, B, C coefficients of general hyperplane equations.

    :param n_recursions: Number of recursions for Cantor set
    :return: Tuple with lists of coefficients A, B and C
    """
    coord_batches = get_coord_batches(n_recursions)

    AB_coeff, C_coeff = [], []
    for coord_batch in coord_batches:
        AB, C = get_hyperplane_equation(np.ravel(coord_batch))
        AB_coeff.append(AB)  #
        C_coeff.append(C)

    return AB_coeff, np.ravel(C_coeff)


def weights_dnf_based_representation(n_recursions: int, points: np.ndarray, representation: str = 'B') -> np.ndarray:
    """
    Calculate weights based on the specified representation.
    """
    # n_recursions = n_recursions.astype('int')
    coeff_ab, coeff_c = get_ABC_coeff(n_recursions)
    # matrix multiply points with hyperplanes coefficients, later recover the shapes
    y = np.matmul(coeff_ab, points.T)  # y=hyperplane_values
    y += coeff_c if points.ndim == 1 else coeff_c[:, np.newaxis]

    if n_recursions == 1:
        cantor_shape = y
    else:
        dents = [np.max([y[4 * k - 2], y[4 * k - 1], y[4 * k]], axis=0) for k in range(1, y.shape[0] // 4 + 1)]
        try:
            cantor_shape = np.vstack((y[0], y[1], y[-1], np.asarray(dents)))
        except ValueError:
            cantor_shape = np.hstack((y[0], y[1], y[-1], np.asarray(dents)))

    prepare_weights = PrepareWeights(representation=representation)
    weights = prepare_weights.output_weights(relu(cantor_shape))
    weights.insert(0, relu(cantor_shape))
    return weights


def array_to_fractions(arr):
    return np.array(list(map(Fraction.from_float, arr)), dtype=object)


if __name__ == '__main__':
    points = np.array([1 / 3, 1 / 3])
    n_recursions = 2
    weights_recursion_based = weights_recursion_based_representation(n_recursions)
    weights_disjunctive_normal_form_based = weights_dnf_based_representation(points=points, n_recursions=n_recursions)

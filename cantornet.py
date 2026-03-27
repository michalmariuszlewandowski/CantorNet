"""Compact math utilities for constructing CantorNet decision boundaries."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterator, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FractionArray: TypeAlias = NDArray[np.object_]
Representation: TypeAlias = Literal["B", "C"]

__all__ = [
    "DNFWeightBuilder",
    "PrepareWeights",
    "activation_pattern",
    "array_to_fractions",
    "boundary_hyperplane_coefficients",
    "cantor",
    "cantor_segments",
    "boundary_segments",
    "boundary_y_coordinates",
    "decision_boundary_vertices",
    "get_ABC_coeff",
    "get_codes",
    "get_coord_batches",
    "get_hyperplane_equation",
    "get_y_coordinates",
    "match_coordinates",
    "pair_coordinates",
    "relu",
    "dnf_representation_weights",
    "recursive_representation_weights",
    "weights_dnf_based_representation",
    "weights_recursion_based_representation",
]


def relu(x: ArrayLike) -> np.ndarray:
    """Apply the ReLU non-linearity element-wise."""

    return np.maximum(np.asarray(x, dtype=object), 0)


def get_codes(vec: ArrayLike) -> NDArray[np.int_]:
    """Binarize activations into a linear-region code."""

    return np.atleast_1d((np.asarray(vec, dtype=object) > 0).astype(int))


def activation_pattern(vec: ArrayLike) -> NDArray[np.int_]:
    """Return the binary activation pattern induced by an activation vector."""

    return get_codes(vec)


def _validate_recursions(n_recursions: int) -> None:
    if n_recursions < 1:
        raise ValueError("n_recursions must be >= 1.")


def _to_fraction(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value), 1)
    if isinstance(value, (float, np.floating)):
        return Fraction(float(value)).limit_denominator()
    return Fraction(value)


def array_to_fractions(arr: ArrayLike) -> FractionArray:
    """Convert an array-like input to a NumPy object array of Fractions."""

    values = np.asarray(arr, dtype=object)
    fractions = np.empty(values.shape, dtype=object)
    for index, value in np.ndenumerate(values):
        fractions[index] = _to_fraction(value)
    return fractions


def recursive_representation_weights(n_recursions: int) -> list:
    """
    Generate the compact recursion-based CantorNet construction (representation A).

    The returned list alternates between the affine map `(W_1, b_1)` and the projection
    matrix `W_2` for each recursion, followed by the final affine readout `(W_top, b_top)`.
    """

    _validate_recursions(n_recursions)

    W_1 = np.array([[3, 0], [-3, 0], [0, 1]], dtype=int)
    b_1 = np.array([-2, 1, 0], dtype=int)
    W_2 = np.array([[1, 1, 0], [0, 0, 1]], dtype=int)
    W_top = array_to_fractions(np.array([-1 / 2, 1]))
    b_top = array_to_fractions(np.array([-1 / 2]))

    weights = []
    for _ in range(n_recursions):
        weights.append([W_1.copy(), b_1.copy()])
        weights.append(W_2.copy())

    weights.append([W_top.copy(), b_top.copy()])
    return weights


def weights_recursion_based_representation(n_recursions: int) -> list:
    """Backward-compatible alias for :func:`recursive_representation_weights`."""

    return recursive_representation_weights(n_recursions)


def _pad_odd_weight_matrix(weight_array: np.ndarray) -> np.ndarray:
    """Expand a min-by-ReLU construction to handle an odd number of inputs."""

    weight_array = np.c_[weight_array, np.zeros(weight_array.shape[0], dtype=object)]
    helper = np.zeros((2, weight_array.shape[1]), dtype=object)
    helper[0, -1] = 1
    helper[1, -1] = -1
    return np.r_[weight_array, helper]


def _pad_odd_sign_matrix(signs_array: np.ndarray) -> np.ndarray:
    """Expand the sign matrix to match an odd number of inputs."""

    sign_array = np.r_[signs_array, np.zeros(signs_array.shape[1], dtype=object)[np.newaxis, :]]
    helper = np.zeros((2, sign_array.shape[0]), dtype=object).T
    helper[-1, -1] = -1
    helper[-1, -2] = 1
    return np.c_[sign_array, helper]


class DNFWeightBuilder:
    """Prepare the auxiliary ReLU constructions used by representations B and C."""

    def __init__(self, representation: Representation):
        """
        Initialize the min-by-ReLU construction.

        Representation B uses `min(a, b) = a - max(0, a - b)`.
        Representation C uses `min(a, b) = 1/2 * (a + b - |a - b|)`.
        """

        if representation == "B":
            self.A = np.array([[0, 1], [0, -1], [-1, 1]], dtype=int)
            self.S = np.array([1, -1, -1], dtype=int)
        elif representation == "C":
            self.A = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=int)
            self.S = array_to_fractions(np.array([1, -1, -1, -1]) / 2)
        else:
            raise ValueError(f"Unknown representation {representation!r}. Expected 'B' or 'C'.")

    def build_min_stage(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare one reduction stage for a min-by-ReLU tree."""

        n_inputs = vec.shape[-1] if vec.ndim == 2 else vec.shape[0]
        padded_inputs = n_inputs + (n_inputs + 1) % 2
        eye = np.eye((padded_inputs - 1) // 2, dtype=object)
        S_1 = np.kron(eye, self.S)
        W_1 = np.kron(eye, self.A)
        if n_inputs % 2:
            W_1 = _pad_odd_weight_matrix(np.squeeze(W_1))
            S_1 = _pad_odd_sign_matrix(S_1)

        pre_activation = W_1 @ (vec.T if vec.ndim == 2 else vec)
        return S_1, W_1, relu(pre_activation)

    def build_weights(self, arr: np.ndarray) -> list[np.ndarray]:
        """Iteratively reduce hyperplane responses to the DNF-like construction."""

        arr = np.squeeze(arr)
        weights = []
        S_m_holder = None
        while arr.shape[0] > 1:
            S_m, W_m, pre_code = self.build_min_stage(arr.T)
            arr = S_m @ pre_code
            weights.append(W_m if S_m_holder is None else W_m @ S_m_holder)
            S_m_holder = S_m
        weights.append(self.S)
        return weights


PrepareWeights = DNFWeightBuilder


def cantor(
    line_segment: tuple[tuple[Fraction, Fraction], ...] | None = None,
) -> Iterator[tuple[tuple[Fraction, Fraction], ...]]:
    """Yield successive finite Cantor-set approximations as exact intervals."""

    segments = line_segment or ((Fraction(0), Fraction(1)),)
    one_third = Fraction(1, 3)

    while True:
        yield segments
        next_segments = []
        for start, end in segments:
            segment_length_third = (end - start) * one_third
            next_segments.extend(
                [
                    (start, start + segment_length_third),
                    (end - segment_length_third, end),
                ]
            )
        segments = tuple(next_segments)


def cantor_segments(n_recursions: int) -> tuple[tuple[Fraction, Fraction], ...]:
    """Return the Cantor-set intervals after a fixed number of recursions."""

    _validate_recursions(n_recursions)

    generator = cantor()
    next(generator)  # level 0
    segments = ()
    for _ in range(n_recursions):
        segments = next(generator)
    return segments


def boundary_y_coordinates(nb_recursion: int) -> FractionArray:
    """
    Get exact y coordinates of the CantorNet boundary, alternating between 1 and 1/2.

    `nb_recursion` is one less than the overall CantorNet recursion depth.
    """

    if nb_recursion < 0:
        raise ValueError("nb_recursion must be >= 0.")

    base_pattern = [Fraction(1), Fraction(1), Fraction(1, 2), Fraction(1, 2)]
    y_coord = (base_pattern * (2**nb_recursion))[1:] + [Fraction(1)]
    return np.array(y_coord, dtype=object)


def get_y_coordinates(nb_recursion: int) -> FractionArray:
    """Backward-compatible alias for :func:`boundary_y_coordinates`."""

    return boundary_y_coordinates(nb_recursion)


def decision_boundary_vertices(n_recursions: int) -> FractionArray:
    """Return the piecewise-linear CantorNet boundary vertices as exact coordinates."""

    segments = cantor_segments(n_recursions)
    x_coord = np.array([endpoint for segment in segments for endpoint in segment], dtype=object)
    y_coord = boundary_y_coordinates(n_recursions - 1)
    return np.column_stack((x_coord, y_coord))


def pair_coordinates(x_coord: ArrayLike, y_coord: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Pair consecutive boundary vertices to prepare hyperplane calculations."""

    x_coord = np.ravel(np.asarray(x_coord, dtype=object))
    y_coord = np.ravel(np.asarray(y_coord, dtype=object))
    if x_coord.shape[0] != y_coord.shape[0]:
        raise ValueError("x_coord and y_coord must contain the same number of entries.")

    joint_coord = np.c_[x_coord, y_coord]
    return tuple((coord_set1, coord_set2) for coord_set1, coord_set2 in zip(joint_coord[:-1], joint_coord[1:]))


def match_coordinates(x_coord: ArrayLike, y_coord: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for :func:`pair_coordinates`."""

    return pair_coordinates(x_coord, y_coord)


def boundary_segments(n_recursions: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Get consecutive vertex pairs used to fit the CantorNet boundary hyperplanes."""

    vertices = decision_boundary_vertices(n_recursions)
    return tuple((coord_set1, coord_set2) for coord_set1, coord_set2 in zip(vertices[:-1], vertices[1:]))


def get_coord_batches(n_recursions: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Backward-compatible alias for :func:`boundary_segments`."""

    return boundary_segments(n_recursions)


def get_hyperplane_equation(coord_batch: ArrayLike) -> tuple[np.ndarray, Fraction]:
    """Return the coefficients `(A, B, C)` of a 2D line in general form `Ax + By + C = 0`."""

    x1, y1, x2, y2 = array_to_fractions(np.ravel(coord_batch))
    if x1 == x2:
        raise ValueError("Degenerate points cannot define a hyperplane.")

    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return np.array([A, B], dtype=object), C


def boundary_hyperplane_coefficients(n_recursions: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the exact boundary hyperplane coefficients for the CantorNet manifold."""

    coord_batches = boundary_segments(n_recursions)
    AB_coeff, C_coeff = [], []
    for coord_batch in coord_batches:
        AB, C = get_hyperplane_equation(np.ravel(coord_batch))
        AB_coeff.append(AB)
        C_coeff.append(C)

    return np.asarray(AB_coeff, dtype=object), np.asarray(C_coeff, dtype=object)


def get_ABC_coeff(n_recursions: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for :func:`boundary_hyperplane_coefficients`."""

    return boundary_hyperplane_coefficients(n_recursions)


def _normalize_points(points: ArrayLike) -> np.ndarray:
    points_array = array_to_fractions(points)
    if points_array.ndim == 1 and points_array.shape[0] == 2:
        return points_array
    if points_array.ndim == 2 and points_array.shape[1] == 2:
        return points_array
    if points_array.ndim == 2 and points_array.shape[0] == 2:
        return points_array.T
    raise ValueError("points must have shape (2,), (n_points, 2), or (2, n_points).")


def dnf_representation_weights(
    n_recursions: int,
    points: ArrayLike,
    representation: Representation = "B",
) -> list[np.ndarray]:
    """Calculate the DNF-like CantorNet construction for a point or a batch of points."""

    _validate_recursions(n_recursions)

    points_array = _normalize_points(points)
    coeff_ab, coeff_c = boundary_hyperplane_coefficients(n_recursions)
    hyperplane_values = np.matmul(coeff_ab, points_array.T if points_array.ndim == 2 else points_array)
    hyperplane_values += coeff_c[:, np.newaxis] if points_array.ndim == 2 else coeff_c

    if n_recursions == 1:
        cantor_shape = hyperplane_values
    else:
        dents = [
            np.max(
                np.asarray(
                    [
                        hyperplane_values[4 * k - 2],
                        hyperplane_values[4 * k - 1],
                        hyperplane_values[4 * k],
                    ],
                    dtype=object,
                ),
                axis=0,
            )
            for k in range(1, hyperplane_values.shape[0] // 4 + 1)
        ]

        if hyperplane_values.ndim == 1:
            cantor_shape = np.array(
                [hyperplane_values[0], hyperplane_values[1], hyperplane_values[-1], *dents],
                dtype=object,
            )
        else:
            cantor_shape = np.vstack(
                (hyperplane_values[0], hyperplane_values[1], hyperplane_values[-1], np.asarray(dents, dtype=object))
            )

    prepare_weights = DNFWeightBuilder(representation=representation)
    weights = prepare_weights.build_weights(relu(cantor_shape))
    weights.insert(0, relu(cantor_shape))
    return weights


def weights_dnf_based_representation(
    n_recursions: int,
    points: ArrayLike,
    representation: Representation = "B",
) -> list[np.ndarray]:
    """Backward-compatible alias for :func:`dnf_representation_weights`."""

    return dnf_representation_weights(n_recursions, points, representation)

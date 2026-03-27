import unittest
from fractions import Fraction

import numpy as np

import cantornet as cn


def _stage_signature(stage):
    if isinstance(stage, (list, tuple)) and len(stage) == 2:
        left, right = stage
        return np.asarray(left, dtype=object).tolist(), np.asarray(right, dtype=object).tolist()
    return np.asarray(stage, dtype=object).tolist()


class CantorNetTestCase(unittest.TestCase):
    def test_array_to_fractions_recovers_simple_rationals(self) -> None:
        values = cn.array_to_fractions(np.array([1 / 3, 0.5, -2]))
        self.assertEqual(list(values), [Fraction(1, 3), Fraction(1, 2), Fraction(-2, 1)])

    def test_cantor_segments_depth_two(self) -> None:
        segments = cn.cantor_segments(2)
        self.assertEqual(
            segments,
            (
                (Fraction(0), Fraction(1, 9)),
                (Fraction(2, 9), Fraction(1, 3)),
                (Fraction(2, 3), Fraction(7, 9)),
                (Fraction(8, 9), Fraction(1)),
            ),
        )

    def test_decision_boundary_vertices_depth_two(self) -> None:
        vertices = cn.decision_boundary_vertices(2)
        self.assertEqual(vertices.shape, (8, 2))
        self.assertEqual(tuple(vertices[0]), (Fraction(0), Fraction(1)))
        self.assertEqual(tuple(vertices[-1]), (Fraction(1), Fraction(1)))

    def test_recursive_representation_stage_count(self) -> None:
        stages = cn.recursive_representation_weights(2)
        self.assertEqual(len(stages), 5)
        top_weights, top_bias = stages[-1]
        self.assertEqual(list(top_weights), [Fraction(-1, 2), Fraction(1)])
        self.assertEqual(list(top_bias), [Fraction(-1, 2)])

    def test_dnf_representation_runs_for_b_and_c(self) -> None:
        point = np.array([Fraction(1, 3), Fraction(1, 3)], dtype=object)

        stages_b = cn.dnf_representation_weights(2, point, representation="B")
        stages_c = cn.dnf_representation_weights(2, point, representation="C")

        self.assertEqual(np.asarray(stages_b[0]).shape, (4,))
        self.assertEqual(np.asarray(stages_c[0]).shape, (4,))
        self.assertEqual(np.asarray(stages_b[-1]).shape, (3,))
        self.assertEqual(np.asarray(stages_c[-1]).shape, (4,))

    def test_legacy_aliases_match_new_api(self) -> None:
        point = np.array([Fraction(1, 3), Fraction(1, 3)], dtype=object)

        self.assertEqual(
            [_stage_signature(stage) for stage in cn.weights_recursion_based_representation(2)],
            [_stage_signature(stage) for stage in cn.recursive_representation_weights(2)],
        )
        self.assertEqual(
            tuple(arr.tolist() for arr in cn.get_ABC_coeff(2)),
            tuple(arr.tolist() for arr in cn.boundary_hyperplane_coefficients(2)),
        )
        self.assertEqual(
            [_stage_signature(stage) for stage in cn.weights_dnf_based_representation(2, point, "B")],
            [_stage_signature(stage) for stage in cn.dnf_representation_weights(2, point, "B")],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
from fractions import Fraction

import numpy as np

from cantornet import (
    boundary_hyperplane_coefficients,
    decision_boundary_vertices,
    dnf_representation_weights,
    recursive_representation_weights,
)


def _parse_fraction(value: str) -> Fraction:
    return Fraction(value)


def _format_stage_shapes(stages: list) -> str:
    parts = []
    for stage in stages:
        if isinstance(stage, (list, tuple)) and len(stage) == 2:
            weights, bias = stage
            parts.append(f"W{np.asarray(weights).shape}+b{np.asarray(bias).shape}")
        else:
            parts.append(f"W{np.asarray(stage).shape}")
    return ", ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect compact CantorNet constructions.")
    parser.add_argument("--depth", type=int, default=2, help="Number of Cantor recursions.")
    parser.add_argument(
        "--point",
        nargs=2,
        default=("1/3", "1/3"),
        metavar=("X", "Y"),
        help="Point used to instantiate the DNF-like construction.",
    )
    parser.add_argument(
        "--representation",
        choices=("B", "C"),
        default="B",
        help="Alternative ReLU construction to inspect.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    point = np.array([_parse_fraction(args.point[0]), _parse_fraction(args.point[1])], dtype=object)

    vertices = decision_boundary_vertices(args.depth)
    hyperplane_coefficients, _ = boundary_hyperplane_coefficients(args.depth)
    recursive_stages = recursive_representation_weights(args.depth)
    dnf_stages = dnf_representation_weights(args.depth, point, representation=args.representation)

    print(f"CantorNet depth: {args.depth}")
    print(f"Boundary vertices: {vertices.shape[0]}")
    print(f"Boundary segments: {hyperplane_coefficients.shape[0]}")
    print(f"Recursive stages: {_format_stage_shapes(recursive_stages)}")
    print(f"Representation {args.representation} stages: {_format_stage_shapes(dnf_stages)}")
    print(f"Probe point: ({point[0]}, {point[1]})")


if __name__ == "__main__":
    main()

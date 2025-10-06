from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    ArgumentTypeError,
    FileType,
    Namespace,
)
from pathlib import Path

from synth_xfer._util.eval import AbstractDomain


def _int_tuple(s: str) -> tuple[int, int]:
    try:
        items = s.split(",")
        if len(items) != 2:
            raise ValueError
        return (int(items[0]), int(items[1]))
    except Exception:
        raise ArgumentTypeError(f"Invalid tuple format: '{s}'. Expected format: int,int")


def _int_triple(s: str) -> tuple[int, int, int]:
    try:
        items = s.split(",")
        if len(items) != 3:
            raise ValueError
        return (int(items[0]), int(items[1]), int(items[2]))
    except Exception:
        raise ArgumentTypeError(f"Invalid tuple format: '{s}'. Expected format: int,int,int")


def build_parser(prog: str) -> Namespace:
    p = ArgumentParser(prog=prog, formatter_class=ArgumentDefaultsHelpFormatter)

    if prog == "synth_transfer":
        p.add_argument("transfer_functions", type=Path, help="path to transfer function")
        p.add_argument("-random_file", type=FileType("r"), help="file for preset operation picks")
        p.add_argument(
            "-domain",
            type=str,
            choices=[str(x) for x in AbstractDomain],
            required=True,
            help="Abstract Domain to evaluate",
        )

    p.add_argument(
        "-outputs_folder",
        type=Path,
        help="Output folder for logs",
        default=Path("outputs"),
    )
    p.add_argument("-random_seed", type=int, help="seed for synthesis")
    p.add_argument(
        "-program_length",
        type=int,
        help="length of synthed program",
        default=28,
    )
    p.add_argument(
        "-total_rounds",
        type=int,
        help="number of rounds the synthesizer should run",
        default=1500,
    )
    p.add_argument(
        "-num_programs",
        type=int,
        help="number of programs that run every round",
        default=100,
    )
    p.add_argument(
        "-inv_temp",
        type=int,
        help="Inverse temperature for MCMC. The larger the value is, the lower the probability of accepting a program with a higher cost.",
        default=200,
    )
    p.add_argument(
        "-lbw",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4],
        help="Bitwidths to evaluate exhaustively",
    )
    p.add_argument(
        "-mbw",
        nargs="*",
        type=_int_tuple,
        default=[],
        help="Bitwidths to evaluate sampled lattice elements exhaustively",
    )
    p.add_argument(
        "-hbw",
        nargs="*",
        type=_int_triple,
        default=[],
        help="Bitwidths to sample the lattice and abstract values with",
    )
    p.add_argument(
        "-num_iters",
        type=int,
        help="number of iterations for the synthesizer",
        default=10,
    )
    p.add_argument(
        "-no_weighted_dsl",
        dest="weighted_dsl",
        action="store_false",
        help="Disable learning weights for each DSL operation from previous for future iterations",
    )
    p.set_defaults(weighted_dsl=True)
    p.add_argument("-condition_length", type=int, help="length of synthd abduction", default=10)
    p.add_argument(
        "-num_abd_procs",
        type=int,
        help="number of mcmc processes used for abduction. Must be less than num_programs",
        default=30,
    )
    p.add_argument(
        "-num_unsound_candidates",
        type=int,
        help="number of unsound candidates considered for abduction",
        default=15,
    )
    p.add_argument("-quiet", action="store_true")

    return p.parse_args()

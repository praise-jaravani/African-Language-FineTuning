

import argparse
from dataclasses import dataclass
from enum import StrEnum


from config import LEARNING_RATE

from config import LEARNING_RATE


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEFT Method Comparison on MasakhaNews")
    parser.add_argument(
        "--method",
        required=True,
        choices=["lora", "ia3", "prompt"],
        help="PEFT method to use",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help="Rank parameter for LoRA (default: 8)",
    )
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=10,
        help="Number of virtual tokens for Prompt Tuning (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum training epochs (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    args : argparse.Namespace = parser.parse_args()

    return args



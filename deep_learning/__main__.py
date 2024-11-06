from argparse import ArgumentParser, Namespace, ArgumentTypeError
from train import MODEL_ARCHS
from typing import TypeVar
from tensorflow.keras.models import Model


arg_choice_return_type = TypeVar(Model)


def parse_args() -> Namespace:
    """Reads and parses the cli arguments

    Returns:
        Namespace: cli-parser arguments
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train", help="Used for training a new model")
    train_parser.add_argument(
        "-bm",
        "--base-model",
        type=_arch_choice,
        choices=MODEL_ARCHS.values(),
    )
    train_parser.set_defaults(func=train_handler)

    parser.set_defaults(func=lambda _: print("invalid option"))
    return parser.parse_args()


def _arch_choice(key: str) -> arg_choice_return_type:
    """Converts the model architecture cli arg to the actual model architecture used by keras

    Args:
        key (str): model name

    Raises:
        ArgumentTypeError: thrown when the model name is invalid

    Returns:
        TypeVar(Model): keras model class for the specified model type
    """
    if key in MODEL_ARCHS:
        return MODEL_ARCHS[key]
    else:
        raise ArgumentTypeError(
            f"Invalid choice: {key}. Choose one of {', '.join(MODEL_ARCHS.keys())}"
        )


def train_handler(args):
    ...
    # model = build_model(args.base_model, (224, 224, 3))


if __name__ == "__main__":
    args = parse_args()
    args.func(args)

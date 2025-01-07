from argparse import ArgumentParser
from train import train
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# ############ GPU Memory Fix ############
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
fix_gpu()
# ########################################

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.output)

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", "-d", type=str, default="data/images/train", help="Path to the dataset")
    train_parser.add_argument("--img_width", "-iw", type=int, default=224, help="Image width")
    train_parser.add_argument("--img_height", "-ih", type=int, default=224, help="Image height")
    train_parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--output", "-o", type=str, default="model.h5", help="Output model file")
    train_parser.set_defaults(func=train_handler)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.func(args)

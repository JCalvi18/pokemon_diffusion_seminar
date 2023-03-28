from argparse import ArgumentParser
from train import train_loop

if __name__ == "__main__":
    parser = ArgumentParser ()
    parser.add_argument ("-t", "--timesteps", dest = "timesteps", type = int,
                         help = "Total number of timesteps"),
    parser.add_argument ("-e", "--epochs", dest = "epochs", type = int,
                         help = "Total number of Epochs"),
    parser.add_argument ("-lr", dest = "lr", type = float,
                         help = "Learning rate"),
    parser.add_argument ("-b", "--batch", dest = "batch", type = int,
                         help = "Nu,ber of batches to use"),

    parser.add_argument ("-tr", "--train", type = bool,
                         action = "store_true", dest = "mode_training", default = False,
                         help = "Use training mode")

    args = parser.parse_args ()

    train_loop(args)
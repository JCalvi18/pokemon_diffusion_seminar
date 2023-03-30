from argparse import ArgumentParser
from train import train_loop

if __name__ == "__main__":
    parser = ArgumentParser ()
    parser.add_argument ("-t", "--timesteps", dest = "timesteps", type = int,
                         help = "Total number of timesteps"),
    parser.add_argument ("-e", "--epochs", dest = "epochs", type = int,
                         help = "Total number of Epochs"),
    parser.add_argument ("-lr", dest = "lr", type = float, default = 1e-3,
                         help = "Learning rate"),
    parser.add_argument ("-b", "--batch", dest = "batch", type = int,
                         help = "Number of batches to use"),

    parser.add_argument ("-dp", "--dataset-path", dest = "dataset_path", type = str, default = './pokemon',
                         help = "Dataset path"),

    parser.add_argument ("-tr", "--train",
                         action = "store_true", dest = "mode_training", default = False,
                         help = "Use training mode")


    args = parser.parse_args ()

    train_loop(args)
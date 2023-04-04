from argparse import ArgumentParser
from train import train_loop, generate

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

    parser.add_argument("-l", "--load", dest="load_path", type=str,
                        help="Path of the experiment on the results folder"),

    parser.add_argument ("-tr", "--train",
                         action = "store_true", dest = "training_mode", default = False,
                         help = "Use training mode")

    parser.add_argument("--log",
                        action="store_true", dest="logger", default=False,
                        help="Log the results")

    parser.add_argument ("-s", "--seed", dest = "seed", type = int, default = 132,
                         help = "Random seed, defaults: ditto :)"),


    args = parser.parse_args ()
    if args.training_mode:
        train_loop(args)
    elif args.load_path:
        generate(args)




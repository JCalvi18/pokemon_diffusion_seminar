# Class to simualate parser so to keep consistency
class Args:
    def __init__(self,
                 timesteps=None,
                 epochs=None,
                 batch=None,
                 dataset_path='./pokemon/',
                 load_path=None,
                 training_mode=False,
                 unet_version=0,
                 seed=132,
                 scale_down=False,
                 ) -> None:
        self.timesteps = timesteps
        self.epochs = epochs
        self.batch = batch
        self.dataset_path = dataset_path
        self.load_path = load_path
        self.training_mode = training_mode
        self.unet_version = unet_version
        self.seed = seed
        self.scale_down = scale_down

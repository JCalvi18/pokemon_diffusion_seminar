import seaborn as sns


def plot_density (image):
    flattened_channel0 = image [:, :, 0].flatten ()
    flattened_channel1 = image [:, :, 1].flatten ()
    flattened_channel2 = image [:, :, 2].flatten ()
    flattened_channel3 = image [:, :, 3].flatten ()
    channels = [flattened_channel0, flattened_channel1, flattened_channel2, flattened_channel3]
    sns.kdeplot (data = channels)
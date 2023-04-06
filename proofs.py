import torch
import pickle
from model import Model
from torchvision import transforms
from dataset import prepare_data, save_values
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np

batch_size = 1
total_timesteps = 1001

torch.manual_seed (132)
timedate_stamp = "{:%B-%d--%H:%M}".format (datetime.now ())
results_folder = Path (f"./results/proof/{timedate_stamp}")
results_folder.mkdir (parents = True, exist_ok = True)

# Generate for the forward process
dataset_path = 'pokemon'
train_dataloader = prepare_data (dataset_path, batch_size)
timesteps = torch.tensor (torch.linspace (100, 900, 9)).long ()
img = next (iter (train_dataloader))

model = Model (None, total_timesteps)
forward = [img]
for t in timesteps:
    forward.append (model.forward_sample (img, t.reshape (1)))

np_transform = transforms.Compose ([
    transforms.Lambda (lambda t: t.permute (0, 2, 3, 1)),  # BCHW to BHWC
    transforms.Lambda (lambda t: (t + 1) / 2),  # In range [0,1]
    transforms.Lambda (lambda t: t * 255.),  # In range [0,255]
    transforms.Lambda (lambda t: t.numpy ().astype (np.uint8)),
])

fig, ax = plt.subplots (nrows = 2, ncols = 5)
fig.suptitle ('Forward process')
array = []
transformed = [np_transform (img) [0] for img in forward]
for i, row in enumerate (ax):
    for j, col in enumerate (row):
        col.set_title (f't={(5 * i + j) * 100}')
        col.axis ('off')
        col.imshow (transformed [i + j])

plt.tight_layout ()
plt.savefig (f'{results_folder}/sample.png', bbox_inches = 'tight')
transformed_dict = {k*100: v for k, v in zip ([0, 2, 3, 4, 5, 6, 7, 8, 9], transformed)}
# np.save (f'{results_folder}/forward', transformed)
with open(f'{results_folder}/forward.pkl','wb') as f:
    pickle.dump(transformed, f)

forward = pickle.load (open ('results/proof/April-05--11:51/forward.pkl', 'rb'))
forward[0]
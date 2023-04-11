# Pokemon Diffusion
Proyect for the seminar "Deep Generative Diffusion Models" - Saarland University 2023

## Enviroment
- Python 3.10
- Pytorch 2.0

## Installation guide for developer
- Install [miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/macos.html) 🐍
- Create a new enronment 
```
    conda create --name diffusion python=3.10
```
- Activate environment
    ```
    conda activate diffusion
    ```
- Install pytorch
    ```
    conda install pytorch torchvision   torchaudio cpuonly -c pytorch
    ```
- Install additionall libraries
  ```
    pip install tqdm matplotlib einops seaborn imageio
    ```

### Further considerations
- [Download](https://github.com/gerritgr/pokemon_diffusion) the project git and place the `pokemon` folder at the root level of the project.

### Docker?
You can use our docker image to train your own model, use our sample `docker-compose.yml` file, and be sure to place the folder of the project at the same level as this file.
```docker-compose
services:
  torch:
    image: tarkusjc/diffusion:torch-pokemon
    shm_size: 7gb
    volumes:
      - ./pokemon_diffusion_seminar:/project:Z
    working_dir: /project
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## For reviewers
### Folder Structure
The file structure of the project is as follows:
```
├── dataset
│   ├── pokemonDataset.py  --> Load the images
│   └── utils.py --> Prepare the trainloader
├── generate.py --> Generate based on trained model
├── main.py --> ENTRYPOINT 
├── model.py --> Forward and backward process
├── model_utils.py --> Cumulative variables
├── network --> Variations of the Unet
│   ├── attention.py
│   ├── __init__.py
│   ├── positional.py
│   ├── resNet.py
│   ├── unetV1.py 
│   ├── unetV2.py
│   ├── unetV3.py
│   └── unetV4.py
├── notebook.ipynb --> File with our proofs
├── notebook_utils.py 
├── plot_utils.py
├── README.md
├── results
└── train.py --> Training loop
```



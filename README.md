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
Check the files on the `results/proof` folder there you'll find all the required files. The folders `v1`, `v2` , `v3` , `v4` , `v1_upsampled` correspond to the resutls obtained using different U-Net. For a general overview of our results you can also check the `notebook.ipynp` notebook with a summary of the proyect.






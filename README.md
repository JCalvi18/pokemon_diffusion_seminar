# Pokemon Diffusion
Proyect for the seminar "Deep Generative Diffusion Models" - Saarland University 2023

## Enviroment
- Python 3.10
- Pytorch 1.13

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
    conda install matplotlib tqdm
    ```

### Further considerations
- [Download](https://github.com/gerritgr/pokemon_diffusion) the project git and place the `pokemon` folder at the root level of the project.

Additionally you can download a docker image of our proyect 
TODO

## Objectives 
- Implements Forward pass (Noiser)
- Implement U-NET
- Define Loss w.r.t. the paper
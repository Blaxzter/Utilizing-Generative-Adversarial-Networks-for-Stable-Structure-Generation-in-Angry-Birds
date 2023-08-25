
# Utilizing Generative Adversarial Networks for Stable Structure Generation in Angry Birds

Welcome to the repository for the paper [Utilizing Generative Adversarial Networks for Stable Structure Generation in Angry Birds](./AAAI_Utilizing-Generative-Adversarial-Networks-for-Stable-Structure-Generation-in-Angry-Birds.pdf).

Included in this repository are:   
1. The pretrained GAN models to generate Science Birds structures.
2. An application to test various model architectures and the decoding algorithm.
3. The AAAI paper and the [Master Thesis](./Frederic_Abraham-Master_Thesis-Stable__Structure_Generation_with_GANs-signed.pdf) on which the paper is based.
4. The generated testing dataset on which the presented results are based, along with the original GAN output and data collected through simulation.

### Examples:

| Low Profile | Most Blocks |
|------------|------------------------------------------------|
| ![Low Profile](./images/created_structures/LowProfile/1_low_profile_page_0.png) | ![Most Blocks](./images/created_structures/NoBlocksDestroyedMostBlocks/0_block_new_page_0.png) |

| Alot of Pigs | Tower |
|------------|------------------------------------------------|
| ![Alot of Pigs](./images/created_structures/PigAmount/23_pig_amount_new_page_0.png) | ![Most Blocks](./images/created_structures/Tower/1_towers_page_0.png) |

### Abstract:  
> This paper investigates the suitability of using Generative Adversarial Networks (GANs) to generate stable structures for the physics-based puzzle game Angry Birds. 
> While previous applications of GANs for level generation have been mostly limited to tile-based representations, this paper explores their suitability for creating stable structures made from multiple smaller blocks.   
> This includes a detailed encoding/decoding process for converting between Angry Birds level descriptions and a suitable grid-based representation, as well as utilizing state-of-the-art GAN architectures and training methods to produce new structure designs.  
> Our results show that GANs can be successfully applied to generate a varied range of complex and stable Angry Birds structures. 

## Installation Instructions:

This project was developed in python 3.8 and is therefore recomended as the installation has been tested with it. 


1. Install [Python 3.8](https://www.python.org/downloads/release/python-380/)
1. Navigate into [src](./src/) and create a virtual environment.
   1.  I usually use [pyenv](https://github.com/pyenv/pyenv) with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
   1. create venv `virtualenv --python C:\Path\To\Python\python.exe venv`
   1. activate venv `.\venv\Scripts\activate`
1. Install requirements `pip -r requirements.txt`
1. Now you should be able to start the Testing Applicataion with `python .\StartApplication.py`


#### Pretrained models 
If you want to load the pretrained models, download them from this [drive folder](https://drive.google.com/drive/folders/1veidxtf0s1Lwqk-Qj7wzvI2z9Rd3jzRf?usp=drive_link) and put them in the model folder: [models](./models/).   

#### Generated dataset
Similar, if you want to load the generated dataset to run the evaluation script or view them in the [Dash visualization](https://dash.plotly.com/) download them [here](https://drive.google.com/drive/folders/1ob5ER3G-tJsDz0ypG5nax6Yq6ao4jkSA?usp=drive_link) and put them into [grid_search](./src/resources/data/eval/grid_search/).

#### Science birds
In order to start the modified science bird build either go to the respective fork: [science-birds](https://github.com/Blaxzter/science-birds) and build it through unity for your system. `# TODO provide args to select correct science birds build.`  
Or download the files from [this drive folder](https://drive.google.com/drive/folders/1CG9PXbvpv-ICWu9aTqlYnBe6eqvWCU6R?usp=drive_link).
Place the download of the science bird build in to [science_birds](./src/resources/science_birds/) folder.

I've only tested the windows build. 

## Training:

The training script is a work in progress.

```
usage: CreateDatasetAndRunTrainer.py [-h]
                                     [-m {WGANGP128128,WGANGP128128_Multilayer,SimpleGAN88212,SimpleGAN100112,SimpleGAN100116}]
                                     [-d DATASET] [-e EPOCH] [-b BATCH_SIZE]
                                     [-x MULTI_LAYER_SIZE] [-a] [-r RUN_NAME]
                                     [-s SAVE_LOCATION]
                                     [-ds DATASET_SAVE_LOCATION]

GAN Training Script

optional arguments:
  -h, --help            show this help message and exit
  -m {WGANGP128128,WGANGP128128_Multilayer,SimpleGAN88212,SimpleGAN100112,SimpleGAN100116}, --model {WGANGP128128,WGANGP128128_Multilayer,SimpleGAN88212,SimpleGAN100112,SimpleGAN100116}
                        Name of the GAN model to use.
  -d DATASET, --dataset DATASET
                        Path to the dataset folder.
  -e EPOCH, --epoch EPOCH
                        Number of epochs for training.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training.
  -x MULTI_LAYER_SIZE, --multi_layer_size MULTI_LAYER_SIZE
                        If a multilayer model is selected the amount of output
                        layers on the last level. 4 = No Air, 5 = With air
  -a, --augmentation    Use data augmentation if set.
  -r RUN_NAME, --run_name RUN_NAME
                        Description/name of the run.
  -s SAVE_LOCATION, --save_location SAVE_LOCATION
                        Location where the model will be saved.
  -ds DATASET_SAVE_LOCATION, --dataset_save_location DATASET_SAVE_LOCATION
                        Location where the created dataset will be saved.

```

## Test Application:

This application has been mainly used to test the various facets of this repo.

![Application](./images/application/FullSizeApplication.png)

You can:
1. Load models
1. Test decoding functions
1. draw a structure
1. Load decoded model into science birds

| Input Drawing | Output_drawing |
|------------|------------------------------------------------|
| ![Input Drawing](./images/application/Smily.png) | ![Output_drawing](./images/application/SmilyDecoded.png) |

The application is more explained in detail in section 4.4 Testing Application or in this video. (TODO)
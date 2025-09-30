# Reframing Generative Models for Physical Systems using Stochastic Interpolants
Anthony Zhou, Alexander Wikner, Amaury Lancelin, Pedram Hassanzadeh, Amir Barati Farimani
## Requirements

To install requirements:
```setup
conda create -n "my_env" 
pip install torch torchvision
conda install lightning -c conda-forge
pip install wandb h5py einops scikit-learn tqdm scipy matplotlib pandas cftime
```

To run Rayleigh-Benard tests ([The Well](https://github.com/PolymathicAI/the_well)):
```
pip install the_well
```

To run SFNO baselines ([Torch Harmonics](https://github.com/NVIDIA/torch-harmonics)):
```
pip install torch-harmonics
```

To evaluate CRPS/SSR ([WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/)):
```
git clone git@github.com:google-research/weatherbench2.git
cd weatherbench2
pip install .
```

## Datasets
There has been substantial work in prior years to make high-quality, physics data publicly available for research, which the authors are grateful for.

Kolmogorov Flow data is generated from [ApeBench](https://github.com/tum-pbs/apebench). Although it is straighforward to re-generate this dataset, the data used in this work is taken from this [paper](https://www.sciencedirect.com/science/article/pii/S0045782525002622) and can be found on [Huggingface](https://huggingface.co/datasets/ayz2/temporal_pdes)

Rayleigh-Benard data is taken from [The Well](https://github.com/PolymathicAI/the_well). More information about the dataset can be found [here](https://polymathic-ai.org/the_well/datasets/rayleigh_benard/). 

Climate data from PlaSim will be released in another publication.

## Training

Workflow for training a model:
```
- Setup environment
- Download a dataset 
- Make a log directory 
- Setup wandb
- Set paths to dataset, normalization stats, logging directory
```

### Autoencoders
To train an autoencoder:
```
python train.py --config=configs/ae/{km_flow/rayleigh_benard/climate}.yaml
```

### Baselines
To train a model:
```
python train.py --config=configs/{km_flow/rayleigh_benard/climate}.yaml --model_name=model_name
```

### Evaluation
To evaluate a model:
```
python val.py --config=configs/{km_flow/rayleigh_benard/climate}.yaml --model_name=model_name --checkpoint=/path/to/model.ckpt
```

To evaluate biases (Note: this requires precomputed climatologies, which are not available at the moment):
```
python eval_bias.py --config=configs/{km_flow/rayleigh_benard/climate}.yaml --model_name=model_name --model_path=/path/to/model.ckpt --save_path=/path/to/save/dir --bias_path=/path/to/bias/dir
```

To evaluate CRPS/SSR:
```
python eval_crps_ssr.py --config=configs/{km_flow/rayleigh_benard/climate}.yaml --model_name=model_name --model_path=/path/to/model.ckpt --save_path=/path/to/save/dir
```
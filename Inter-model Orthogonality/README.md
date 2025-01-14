## Overview
The experimental codes for the Intra-model Orthogonality 


# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n maple python=3.8

# Activate the environment
conda activate maple

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

# Update setuptools package 
pip install setuptools==59.5.0


# Training and Evaluation

The following process is generalized to all domains including remote sensing, and medical.
#### (1) Base-to-Novel class generalization setting
The default training settings are provided in config file at `configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train GDPL on RSICD. 


```bash

# seed=1
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh RSICD 1
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh RSICD 1

# seed=2
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh RSICD 2
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh RSICD 2

# seed=3
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh RSICD 3
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh RSICD 3
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python parse_test_res.py output/base2new/train_base/RSICD/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx
# averaged results for novel classes
python parse_test_res.py output/base2new/test_new/RSICD/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx --test-log
```

The above steps can be repeated for other individual datasets.




#### (2) Cross-Dataset Transfer

We provide cross-dataset config : `configs/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml`.
* Firstly, train GDPL on dataset1  in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/maple/xd_train_maple.sh RSICD 1
# seed=2 
bash scripts/maple/xd_train_maple.sh RSICD 2
# seed=3 
bash scripts/maple/xd_train_maple.sh RSICD 3
```

* Now evaluate RSICD model on all other datasets.

```bash
for SEED in 1 2 3
do
    bash scripts/maple/xd_test_maple.sh dataset1 ${SEED}
    bash scripts/maple/xd_test_maple.sh dataset2  ${SEED}
    bash scripts/maple/xd_test_maple.sh dataset3  ${SEED}
done
```

#### (3) Domain Generalization 
 The steps are similar to the above cross-dataset experiments, however, the trained model is evaluated on the entire datasets including both training and testing.


```bash
for SEED in 1 2 3
do
    bash scripts/maple/xd_test_maple.sh dataset1v2 ${SEED}
    bash scripts/maple/xd_test_maple.sh dataset2v2 ${SEED}
    bash scripts/maple/xd_test_maple.sh dataset3v2 ${SEED}
done
```


You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.
<br>


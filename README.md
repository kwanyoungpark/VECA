# VECA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

![veca_overview](./docs/veca.png)

VECA is currently preparing for the public release of Bayley-4 cognitive tasks, Unity packages, and additional system optimizations. Sorry for the delay.

## Prerequisite

Supported OS: Windows 10 or Linux (Tested on Ubuntu 18.04 and 20.04)

You need 2 python packages: numpy and gdown. You can install them with following command.
```
pip install numpy gdown
```

## Installation & Usage

1. Clone this veca repository.
```
git clone https://github.com/GGOSinon/VECA.git
```

2. If you want to execute the environment on the same machine with your training algorithm, (Local execution)

```
python example.py
```

3. If you want to execute the environment on the different machine (Remote environment execution)

```
python example_envorchestrator.py  # On the machine you want to execute the environment
```
and then execute
```
python example_remoteenv.py   # On the machine you want to execute the training algorithm
```
Make sure that the `port` of `example_envorchestrator.py` is exposed for remote access.

4. For the list of currently supported tasks, please refer to the `example.py` script for more information.

## Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2019-0-01371, Development of brain-inspired AI with human-like intelligence

## Citation

 If you find this work useful in your research, please cite
```
 @article{park2021veca,
  title={VECA: A Toolkit for Building Virtual Environments to Train and Test Human-like Agents},
  author={Park, Kwanyoung and Oh, Hyunseok and Lee, Youngki},
  journal={arXiv preprint arXiv:2105.00762},
  year={2021}
}
```


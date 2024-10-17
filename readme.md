# Principled Task Grouping

This repository contains the official code implementation for the paper "Towards Principled Task Grouping for Multi-Task Learning" ([arXiv:2402.15328](https://arxiv.org/abs/2402.15328)).

## Paper Abstract

This paper presents a novel approach to task grouping in Multitask Learning (MTL), advancing beyond existing methods by addressing key theoretical and practical limitations. Unlike prior studies, our approach offers a more theoretically grounded method that does not rely on restrictive assumptions for constructing transfer gains. We also propose a flexible mathematical programming formulation which can accommodate a wide spectrum of resource constraints, thus enhancing its versatility. Experimental results across diverse domains, including computer vision datasets, combinatorial optimization benchmarks and time series tasks, demonstrate the superiority of our method over extensive baselines, validating its effectiveness and general applicability in MTL.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citing](#citing)
- [License](#license)

## Installation

For each dataset, please refer to the original code repository for installation guidance.

## Usage

Use the following commands to collect required data for task grouping:

### Taskonomy
```bash
cd taskonomy
bash run_collect.sh
```

### CelebA:
```bash
cd celeba/ours_code
python celeba-ours-collect.py
```

### COP
```bash
cd cop
bash script_collect.sh
```

### ETTm1
```bash
cd ettm1
bash scripts/combinations/longterm_ETTh1_collect.sh
```

## Citing

If you find this code useful for your research, please cite the original paper:

```bibtex
@misc{wang2024principledtaskgroupingmultitask,
      title={Towards Principled Task Grouping for Multi-Task Learning}, 
      author={Chenguang Wang and Xuanhao Pan and Tianshu Yu},
      year={2024},
      eprint={2402.15328},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.15328}, 
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

This repository also includes third-party code from the following projects:
- [TAG from Google Research](https://github.com/google-research/google-research/tree/master/tag) - Licensed under Apache 2.0.
- [Taskgrouping by tstandley](https://github.com/tstandley/taskgrouping) - Licensed under the MIT License.
- [Time-Series-Library by thuml](https://github.com/thuml/Time-Series-Library) - Licensed under the MIT License.
- [MTL-COP by Wastedzz](https://github.com/Wastedzz/MTL-COP) - Licensed under the MIT License.

## RelGAN

This repository contains the code to reproduce the core results 
from the paper [RelGAN: Relational Generative Adversarial Networks for Text Generation](https://openreview.net/pdf?id=rJedV3R5tm).

## Dependencies
This project uses Python 3.5.2, with the following lib dependencies:
* [Tensorflow 1.4](https://www.tensorflow.org/)
* [Numpy 1.14.1](http://www.numpy.org/)
* [Matplotlib 2.2.0](https://matplotlib.org)
* [NLTK 3.2.3](https://www.nltk.org)
* [tqdm 4.19.6](https://pypi.python.org/pypi/tqdm)
* [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit-archive)


## Instructions
The `experiments` folders contain scripts for starting the different experiments.
For example, to reproduce the `synthetic data` experiments, you can try:
```
cd oracle/experiments
python3 oracle_relgan.py [job_id] [gpu_id]
```
or `COCO Image Captions`:
```
cd real/experiments
python3 coco_relgan.py [job_id] [gpu_id]
```
or `EMNLP2017 WMT News`:
```
cd real/experiments
python3 emnlp_relgan.py [job_id] [gpu_id]
```
Note to replace [job_id] and [gpu_id] with appropriate numerical values.

## Reference
To cite this work, please use
```
@INPROCEEDINGS{Nie2019ICLR,
  author = {Weili Nie, Nina Narodytska and Ankit Patel},
  title = {RelGAN: Relational Generative Adversarial Networks for Text Generation},
  booktitle = {ICLR},
  year = {2019}
}
```

## Acknowledgement
This code is based on the previous benchmarking platform [Texygen](https://github.com/geek-ai/Texygen). 
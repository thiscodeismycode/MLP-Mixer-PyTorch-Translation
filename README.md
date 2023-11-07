# MLP-Mixer-experiments
Intelligent Information studies Assignment 4<br>
Based on [MLP Mixer](https://arxiv.org/abs/2105.01601)<br>

### Requirements
```shell
pip install -r requirements.txt
```

### Training
To train from scratch:
```shell
python3 main.py
```
To execute hyperparameter search:
```shell
python3 main.py -e yes
```
Execute PyTorch TensorBoard:
```shell
tensorboard --logdir=runs
```

### Citations
```
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision}, 
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgements
[@rishikksh20](https://github.com/rishikksh20/MLP-Mixer-pytorch/tree/master)'s PyTorch implementation
of MLP-Mixer was of great help.

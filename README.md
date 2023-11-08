# MLP-Mixer PyTorch translation
Implementation based on [MLP Mixer](https://arxiv.org/abs/2105.01601)<br>

### Requirements
```
pip install -r requirements.txt
```

### Training
To train from scratch:
```
python3 main.py
```
To execute hyperparameter search:
```
./mlp_mixer.sh
```
Troubleshooting for 'Permission denied' error
```
chmod +x mlp_mixer.sh
```
Execute in the background with `nohup`
```
nohup ./mlp_mixer.sh > output.txt &
```
Execute PyTorch TensorBoard:
```
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
[@rishikksh20](https://github.com/rishikksh20/MLP-Mixer-pytorch/tree/master)'s PyTorch translation of MLP-Mixer was of great help.

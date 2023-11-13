# MLP-Mixer PyTorch translation
Implementation based on [MLP Mixer](https://arxiv.org/abs/2105.01601)<br>
Pure translation without any structural changes.<br>
⚠ Dataloaders and default hyperparameters are set to load and train on _CIFAR10 dataset_.

### Requirements
```
pip install -r requirements.txt
```

### Training
**Train with default options**
```
python3 main.py
```
**Train with customized options**

Replace the capitalized words with your own arguments. All arguments are optional, thus not necessary.
</br> Execute `python3 main.py -h` for details.
```
python3 main.py -b BATCH_SIZE -n NUM_BLOCKS -p PATCH_SIZE -d HIDDEN_DIM -t TOKENS_MLP_DIM -c CHANNELS_MLP_DIM -l LEARNING_RATE
```
**Hyperparameter tuning**
1. Change the given `hyperparameter.yaml` file, or write your own.
2. Execute training with '-y' or '--hype' and give "yes" as its argument input
3. Or execute the given shell file, `mlp_mixer.sh`
```
python3 main.py -y yes

 or

./mlp_mixer.sh
```

**+ Some tips**

Execute training in the background
```
nohup ./mlp_mixer.sh > output.txt &
```
Execute PyTorch TensorBoard:
```
tensorboard --logdir=runs
```

### Citation
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

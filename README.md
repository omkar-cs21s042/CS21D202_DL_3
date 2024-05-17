# Assignment 3

A Sequence2Sequence model was constructed using PyTorch, comprising encoder-decoder layers. 
The Encoder and Decoder class offers the flexibility to customize the model according to desired specifications, including:
- Input Embedding size
- Output Embedding size  
- layers
- cell type
- Dropout

## wandb
This repository consists of a single scipt. In order to run the script, wandb key has to be added in the script file.
```python
wandb.login('key = '')
```
Update this in the script with your key.

## Dataset
The dataset has to be downloaded and the path must be given as command line argument to the scripts. More details on command line arguments is given in next section. Prefer using fully qualified path over relative path.

## Usage

```bash
python trainA.py [OPTIONS]
```

## Options

- `--wandb_project [PROJECT]`: Specifies the WandB project name. Default is "myproject".
- `--wandb_entity [ENTITY]`: Specifies the WandB entity name. Default is "myname".
- `--inputEmbedding [INPUT_EMBEDDING]`: Specifies the dimension of input embedding layer. Default is 32.
- `--outputEmbedding [OUTPUT_EMBEDDING]`:  Specifies the dimension of output embedding layer. Default is 256.
- `--cellType [CELL_TYPE]`: Specifies the type of cell. Default is "rnn".
- `--beamWidth [BEAM_WIDTH]`: Specifies the width of beam. Default is 3.
- `--epochs [EPOCHS]`: Specifies the number of epochs for training. Default is 10.
- `--batchSize [BATCH_SIZE]`: Specifies the batch size for training. Default is 64.
- `--learningRate [LEARNING_RATE]`: Specifies the learning rate for training. Default is 0.002.
- `--dropout [DROPOUT]`: Specifies the dropout rate to be applied. Default is 0.1.
- `--attention [ATTENTION]`: Specifies whether attention has to be applied or not. Default is True.
- `--optimizer [OPTIMIZER]`: Specifies the type of optimizer. Default is adam.
- `--trainPath [TRAINING_DATA_PATH]`: Specifies the path to the training data. Default is "/content/dataset/train".
- `--valPath [VALIDATION_DATA_PATH]`: Specifies the path to the validation data. Default is "/content/dataset/val".
- `--testPath [TESTING_DATA_PATH]`: Specifies the path to the testing data. Default is "/content/dataset/test".


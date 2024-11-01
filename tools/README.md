# Tools

## Download and preprocess Crello dataset

> [!NOTE]
> The preprocessed Crello dataset is already included in our starter kit.
> You can use this script to create the Crello dataset with your own configuration.

### Setup
To use [huggingface datasets library](https://huggingface.co/docs/datasets/index), we recommend setting up the following Python environment:

Requirement
```
python==3.9
torch==1.13.1
torchvision==0.14.1
torchaudio==0.13.1
torch_geometric==2.3.1
datasets==2.14.4
```

### Run script
```
python ./make_crello_datset ./download/datasets --max_seq_length 25
```
This script downloads the Crello dataset from Hugging Face, performs data preprocessing, and saves the processed data to ./download/datasets/crello-bbox-max{MAX_SEQ_LENGTH}/processed.
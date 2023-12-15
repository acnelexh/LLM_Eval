# bc-lstm evaluation

## Installation

Download the DailyDialog.zip from https://github.com/declare-lab/conv-emotion/tree/master/bc-LSTM-pytorch and unzip it in the root directory of this repository.

## Usage

### Set up

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
python3 train_DailyDialog.py --epochs 1 
```

After training, the best model will be saved in the dailydialog/best_model.pt along side the hyperparameters and model configuration in dailydialog/best_model_config.pkl
Dont need to touch this part really, just need a model to test the evalutation script.

### Evaluation  

```bash
python3 eval.py
```
TODO dataloader.py, under LLMDataset, needs a few more touch.
You can compare the desire output with __getiten__() in DailyDialogueDataset.
import sys; sys.path.append("../../")

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from sklearn.metrics import f1_score
from modules.utils import load_yaml, save_yaml
from modules.evaluation import pretty_classification_report, plot_confusion_matrix
from models.ft_transformer.modules import FTTransformer, MLP, ResNet


# Wandb
import wandb
run = wandb.init(project="FTTransformer")


# Paths
TRAIN_CONFIG_PATH = '../../config/train_config.yaml'
DATA_PATH = '../../data/01_split/'
KST = timezone(timedelta(hours=9))
PREDICT_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
SAVE_DIR = f'checkpoints/{PREDICT_TIMESTAMP}'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# Training configuration
task_type = 'multiclass'
lr = 0.001
weight_decay = 0.0
device = torch.device('cuda:1')
n_epochs = 1000
batch_size = 128




def main():
    ###########################
    # Read data
    ###########################

    config = load_yaml(TRAIN_CONFIG_PATH)
    LABEL_ENCODING = config['LABEL_ENCODING']
    encoding_to_label = {v: k for k, v in LABEL_ENCODING.items()}
    labels = [key for key in LABEL_ENCODING.keys()]

    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(DATA_PATH, 'valid.csv'))
    train_X, train_y = train_df.loc[:,train_df.columns!='leaktype'], train_df['leaktype']
    valid_X, valid_y = valid_df.loc[:,train_df.columns!='leaktype'], valid_df['leaktype']


    train_y = train_y.replace(LABEL_ENCODING)
    valid_y = valid_y.replace(LABEL_ENCODING)

    # Same testing and validation
    test_X, test_y = valid_X, valid_y


    ###########################
    # Model setup
    ###########################

    task_type = 'multiclass'
    assert task_type in ['binclass', 'multiclass', 'regression']

    n_classes = int(max(train_y)) + 1 if task_type == 'multiclass' else None

    X = {}
    y = {}
    X['train'],  X['val'], X['test'] = train_X, valid_X, test_X
    y['train'], y['val'],  y['test'] = train_y, valid_y, test_y

    # not the best way to preprocess features, but enough for the demonstration
    preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
    X = {
        k: torch.tensor(preprocess.fit_transform(v), device=device)
        for k, v in X.items()
    }
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

    # !!! CRUCIAL for neural networks when solving regression problems !!!
    if task_type == 'regression':
        y_mean = y['train'].mean().item()
        y_std = y['train'].std().item()
        y = {k: (v - y_mean) / y_std for k, v in y.items()}
    else:
        y_std = y_mean = None

    if task_type != 'multiclass':
        y = {k: v.float() for k, v in y.items()}

    ### OURS convert all to float for compatibility
    X = {k: v.float() for k, v in X.items()}
    # y = {k: v.float() for k, v in y.items()} # not y, because this is an index

    # Imbalanced learning weights
    class_weights = torch.Tensor(compute_class_weight(class_weight='balanced', classes=np.arange(n_classes), y=train_y)).to(device)


    d_out = n_classes or 1
    num_features = X['train'].shape[1] # change based on your application

    model_config = {'n_num_features':num_features,
                    'cat_cardinalities':None,
                    'last_layer_query_idx':[-1],
                    'd_out':d_out}

    model = FTTransformer.make_default(**model_config)

    # Save configuration
    save_yaml(os.path.join(SAVE_DIR, 'model_config.yaml'), model_config)


    model.to(device)
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)


    # Wandb
    def train_log(target, prediction, epoch=0):
        f1_macro = sklearn.metrics.f1_score(target, prediction, average='macro')
        f1_weighted = sklearn.metrics.f1_score(target, prediction, average='weighted')
        accuracy = sklearn.metrics.accuracy_score(target, prediction)
        wandb.log({"epoch": epoch, "f1_macro": f1_macro, "f1_weighted":f1_weighted, 'accuracy':accuracy})


    # Model apply
    def apply_model(x_num, model=model, x_cat=None):
        if isinstance(model, FTTransformer):
            return model(x_num, x_cat)
        elif isinstance(model, (MLP, ResNet)):
            assert x_cat is None
            return model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(model)}.'
                ' Then you have to implement this branch first.'
            )


    # Only log Wandb validation
    @torch.no_grad()
    def evaluate(part, batch_size_eval=32, log_wandb=True, epoch=0): 
        model.eval()
        prediction = []
        for batch in zero.iter_batches(X[part], batch_size_eval):
            prediction.append(apply_model(batch.float()).cpu())
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        target = y[part].cpu().numpy()

        if task_type == 'binclass':
            prediction = np.round(scipy.special.expit(prediction))
            score = sklearn.metrics.accuracy_score(target, prediction)
        elif task_type == 'multiclass':
            prediction = prediction.argmax(1)
            score = sklearn.metrics.f1_score(target, prediction, average='macro') # NOTE: changed to macro F1
            # score = sklearn.metrics.accuracy_score(target, prediction)

            # Wandb Logging
            if part == 'val' and log_wandb:
                train_log(target, prediction, epoch=epoch)
        else:
            assert task_type == 'regression'
            score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
        return score

    train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)

    # Create a progress tracker for early stopping
    # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
    progress = zero.ProgressTracker(patience=100)

    print(f'Test score before training: {evaluate("test", log_wandb=False):.4f}')

    ###########################
    # Main training loop
    ###########################

    report_frequency = len(X['train']) // batch_size // 5
    for epoch in range(1, n_epochs + 1):
        for iteration, batch_idx in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x_batch = X['train'][batch_idx]
            y_batch = y['train'][batch_idx]
            loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch) # NOTE: move to double
            loss.backward()
            optimizer.step()
            if iteration % report_frequency == 0:
                print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

        val_score = evaluate('val', epoch=epoch)
        test_score = evaluate('test', epoch=epoch)
        print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
        progress.update((-1 if task_type == 'regression' else 1) * val_score)
        if progress.success:
            print(' <<< BEST VALIDATION EPOCH', end='')
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pt')) # best checkpoint
        print()
        if progress.fail:
            break
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last.pt')) # last checkpoint


if __name__ == '__main__':
    main()


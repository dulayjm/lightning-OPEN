from __future__ import absolute_import
from model import Model
from callback import MetricCallback
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from torchvision import models
import optuna
import argparse


def objective(trial):
    metrics_callback = MetricCallback()

    trainer = pl.Trainer(
        max_epochs=25,
        num_sanity_val_steps=3,
        gpus=[1] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
    )
    hparams = {
        'lr': trial.suggest_float("lr", 0.0006, 0.0008), 
    }

    # init_model()
    model_ft = Model(hparams, trial)
    ct=0
    for child in model_ft.model1.children():
        ct += 1
        if ct < 8: # change to lower numbers for training more layers
            for param in child.parameters():
                param.requires_grad = False
    ct=0
    for child in model_ft.model2.children():
        ct += 1
        if ct < 8:
            for param in child.parameters():
                param.requires_grad = False


    # model = Model(hparams, trial)
    trainer.fit(model_ft)


    return metrics_callback.metrics[-1]


if __name__ == "__main__":
    pruner = optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=5, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("All trials:")
    print(study.trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
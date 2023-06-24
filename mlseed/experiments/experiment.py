import mlflow.pytorch
from torch import optim
import pytorch_lightning as pl
import math


class Experiment(pl.LightningModule):

    def __init__(self, model, experiment_params, datamodule, best_model_uri, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.datamodule = datamodule
        self.best_model_uri = best_model_uri
        self.experiment_params = experiment_params

        self.best_g_loss = math.inf

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.experiment_params['lr'])
        return optimizer

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result = self.model(batch)
        loss = self.model.loss(*result)
        self.log_dict({key: val.item() for key, val in loss.items()}, on_step=True, on_epoch=True)
        return loss['loss']

    def validation_step(self, batch, batch_idx):
        result = self.model(batch[0])
        loss = self.model.loss(*result)

        self.log_dict({'val_d_loss': loss['d_loss'], 'val_g_loss': loss['g_loss']})

        if loss['g_loss'] < self.best_g_loss:
            mlflow.pytorch.log_model(self.model, 'best_model')
            mlflow.log_metric('best_model_step', self.current_epoch)
            self.best_g_loss = loss['g_loss']

    def test_step(self, batch, batch_idx):
        best_model = mlflow.pytorch.load_model(self.best_model_uri)
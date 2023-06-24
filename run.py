import argparse
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from mlseed.datasets.sp500 import SP500Dataset
from mlseed.experiments import Experiment, experiments
from mlseed.utils import get_config
from mlseed.models import models


def run(config, enable_progress_bar=True, callbacks=None):

    seed_everything(config['experiment_params']['manual_seed'])
    datamodule = SP500Dataset(batch_size=config['data_params']['batch_size'])
    model = models[config['model_params']['name']](**config['model_params'])

    # init experiment
    active_run = mlflow.active_run()
    best_model_uri = f'runs:/{active_run.info.run_id}/best_model'
    experiment_class = config['experiment_params'].get('experiment_class', None)
    cls = experiments.get(experiment_class, Experiment)
    experiment = cls(model, config.get('experiment_params', None), datamodule, best_model_uri)

    # train
    mlflow.log_dict(config, 'config.yaml')
    if callbacks is None:
        callbacks = []
    num_sanity_val_steps = config['trainer_params'].get('num_sanity_val_steps', 0)
    config['trainer_params'] = {k: v for k, v in config['trainer_params'].items() if k != 'num_sanity_val_steps'}
    trainer = Trainer(callbacks=callbacks, enable_progress_bar=enable_progress_bar, num_sanity_val_steps=num_sanity_val_steps, **config['trainer_params'])

    mlflow.pytorch.autolog(log_models=False)

    # if the param name is longer than 250 chars, there is going to be an error
    for name in 'model_params', 'experiment_params', 'data_params', 'trainer_params':
        if name in config:
            for k, v in config[name].items():
                if not isinstance(v, dict):
                    mlflow.log_param(k, v)

    trainer.fit(experiment, datamodule=datamodule)
    result = trainer.test(experiment, datamodule=datamodule)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--config', dest="filename", metavar='FILE')
    parser.add_argument('--experiment_id', type=str, default="0")
    parser.add_argument('--run_id', type=str, default=None)
    args = parser.parse_args()

    assert not (args.mode == 'test' and args.filename is not None), "--config is invalid in `test` mode, it's retrieved from mlflow."
    assert not (args.mode == 'train' and args.run_id is not None), "--run_id is invalid in `train` mode, it's generated automatically."

    mlflow.set_experiment(args.experiment_id)

    with mlflow.start_run(run_id=args.run_id) as active_run:
        filename = args.filename
        config = get_config(filename)
        run(config)

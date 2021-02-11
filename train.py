from pytorch_lightning import seed_everything, Trainer

from core.callbacks import ProgressBar
from core.data.data_module import LymphoDataModule
from core.models.mil_model import MILModel
from core.models.simple_cnn import SimpleCNN


def run_training(clf, trainer, data_module):
    assert trainer.reload_dataloaders_every_epoch
    print('Setting trainer.max_epochs = 1.')
    trainer.max_epochs = 1
    data_module.inference_dataset_reference = data_module.train_dataset
    inference_metrics = trainer.test(clf, datamodule=data_module)[0]
    data_module.train_dataset_reference = data_module.train_dataset
    trainer.fit(clf, datamodule=data_module)
    training_metrics = clf.training_metrics
    return inference_metrics, training_metrics


if __name__ == '__main__':
    seed_everything(31)

    data_module = LymphoDataModule('/Users/dilip.thiagarajan/data/pcam/', 128, 0, 48)
    data_module.setup()

    model = SimpleCNN()
    clf = MILModel(model)
    trainer = Trainer(reload_dataloaders_every_epoch=True, callbacks=[ProgressBar()])

    all_training_metrics = []
    for epoch in range(5):
        inference_metrics, training_metrics = run_training(clf, trainer, data_module)
        all_training_metrics.append(training_metrics)
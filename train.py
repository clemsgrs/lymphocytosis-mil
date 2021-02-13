from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks import ProgressBar
from data.data_module import LymphoDataModule
from models.mil_model import MILModel
from models.simple_cnn import SimpleCNN

def run_training(clf, trainer, data_module):
    assert trainer.reload_dataloaders_every_epoch
    print('Setting trainer.max_epochs = 1.')
    trainer.max_epochs = 1
    data_module.inference_dataset_reference = data_module.train_dataset
    inference_metrics = trainer.test(clf, datamodule=data_module)[0]
    data_module.train_dataset_reference = data_module.train_dataset
    data_module.validation_dataset_reference = data_module.val_dataset
    trainer.fit(clf, datamodule=data_module)
    training_metrics = clf.training_metrics
    validation_metrics = clf.validation_metrics
    return inference_metrics, training_metrics, validation_metrics


if __name__ == '__main__':
    seed_everything(31)

    data_module = LymphoDataModule(
        data_dir='data/3md3070-dlmi/',
        batch_size=32, 
        num_workers=0)
    data_module.setup()
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/exp1',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        mode='auto'
    )

    model = SimpleCNN()
    clf = MILModel(model)
    trainer = Trainer(
        reload_dataloaders_every_epoch=True, 
        check_val_every_n_epoch=1,
        gpus=1,
        callbacks=[ProgressBar(), checkpoint_callback])

    all_training_metrics = []
    all_validation_metrics = []
    for epoch in range(5):
        inference_metrics, training_metrics, validation_metrics = run_training(clf, trainer, data_module)
        all_training_metrics.append(training_metrics)
        all_validation_metrics.append(validation_metrics)
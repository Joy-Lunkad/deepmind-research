from jaxline import base_config
from ml_collections import config_dict

def get_config():
  """Return config object for training."""
  config = base_config.get_base_config()

  # Experiment config.
  train_batch_size = 1024  # Global batch size.
  images_per_epoch = 1281167
  num_epochs = 90
  steps_per_epoch = images_per_epoch / train_batch_size
  config.training_steps = ((images_per_epoch * num_epochs) // train_batch_size)
  config.random_seed = 0
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              lr=0.1,
              num_epochs=num_epochs,
              label_smoothing=0.1,
              model='ResNet',
              image_size=224,
              use_ema=False,
              ema_decay=0.9999,  # Quatros nuevos amigos
              ema_start=0,
              which_ema='tf1_ema',
              augment_name=None,  # 'mixup_cutmix',
              augment_before_mix=True,
              eval_preproc='crop_resize',
              train_batch_size=train_batch_size,
              eval_batch_size=50,
              eval_subset='test',
              num_classes=1000,
              which_dataset='imagenet',
              fake_data=False,
              which_loss='softmax_cross_entropy',  # For now, must be softmax
              transpose=True,  # Use the double-transpose trick?
              bfloat16=False,
              lr_schedule=dict(
                  name='WarmupCosineDecay',
                  kwargs=dict(
                      num_steps=config.training_steps,
                      start_val=0,
                      min_val=0,
                      warmup_steps=5 * steps_per_epoch),
              ),
              lr_scale_by_bs=True,
              optimizer=dict(
                  name='SGD',
                  kwargs={
                      'momentum': 0.9,
                      'nesterov': True,
                      'weight_decay': 1e-4,
                  },
              ),
              model_kwargs=dict(
                  width=4,
                  which_norm='BatchNorm',
                  norm_kwargs=dict(
                      create_scale=True,
                      create_offset=True,
                      decay_rate=0.9,
                  ),  # cross_replica_axis='i'),
                  variant='ResNet50',
                  activation='relu',
                  drop_rate=0.0,
              ),
          ),))

  # Training loop config: log and checkpoint every minute
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 60
  config.eval_specific_checkpoint_dir = ''

  return config

import torch

import rva.utils
import rva.data_loader

from rva.trainer import Trainer
from rva.config import get_config

import os 
os.chdir("/usr/users/vhassle/curiosity/recurrent-visual-attention")

def main(config):
    rva.utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    # instantiate data loaders
    if config.is_train:
        dloader = rva.data_loader.get_train_valid_loader(
            config.data_dir,
            config.batch_size,
            config.random_seed,
            config.valid_size,
            config.shuffle,
            config.show_sample,
            **kwargs,
        )
    else:
        dloader = rva.data_loader.get_test_loader(
            config.data_dir, config.batch_size, **kwargs,
        )

    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        rva.utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)

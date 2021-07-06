from vision_model import LitClassifier
from dataset import VisionDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch

from argparse import ArgumentParser


def cli_main(args):
    pl.seed_everything(1234)

    # ------------
    # data
    # ------------
    if args.model_type == 'vision':
        dm = VisionDataModule(**vars(args))
        model = LitClassifier(args)
    else:
        raise NotImplementedError("model not implemented yet")

    dm.prepare_data()
    # ------------
    # training
    # ------------
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer.from_argparse_args(args)

    if args.test_only:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])
        # model = LitClassifier.load_from_checkpoint(args.ckpt)
        dm.setup('test')
        result = trainer.test(model, datamodule=dm)
    else:
        dm.setup('fit')
        trainer.fit(model, dm)
        trainer.test(model)


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model_type', default='vision', type=str, choices=['vision', 'nlp'])
    temp_args, _ = parser.parse_known_args()
    if temp_args.model == 'vision':
        parser = LitClassifier.add_model_specific_args(parser)
        parser = VisionDataModule.add_argparse_args(parser)
    elif temp_args.model == 'nlp':
        raise NotImplementedError("nlp not achieve")
    else:
        raise NotImplementedError("pl temp not complete")

    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()

    print(args)
    cli_main(args)

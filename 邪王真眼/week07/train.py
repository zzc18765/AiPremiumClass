import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plugins as plugins

from configs.config import load_train_config
from trainer.trainer import Trainer


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/week07_douban_classification.py")
    trainer = Trainer(cfg)
    trainer.add_plugins([
        plugins.ModelShapeCheckPlugin,
        plugins.InitTrainerPlugin,
        plugins.LoggerPlugin,
        plugins.LRSchedulerPlugin,
        plugins.TrainingMetricsPlugin,
        plugins.ValEvaluationPlugin,
    ])
    trainer.train()

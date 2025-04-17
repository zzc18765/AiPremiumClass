import os
import logging

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class LoggerPlugin(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "setup"
    }

    def __init__(self):
        self.logger = None

    def setup(self, ctx: TrainContext):
        log_path = os.path.join(ctx.workspace["results_folder"], "log.txt")

        self.logger = logging.getLogger(f"trainer_logger_{ctx.workspace.get('uuid', '')}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

        def log_fn(message: str, print_only: bool = False):
            if print_only:
                print(message, end='', flush=True)
                return
            else:
                print(message)

            self.logger.info(message)

        ctx.workspace["logger"] = log_fn

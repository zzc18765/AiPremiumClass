from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext
from torch.optim.lr_scheduler import LambdaLR


class LRSchedulerPlugin(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "initialize_scheduler",
        PluginType.EPOCH_END: "step"
    }

    def initialize_scheduler(self, ctx: TrainContext):
        self._build_scheduler(ctx)

        ctx.workspace["scheduler"] = self.scheduler

    def _build_scheduler(self, ctx: TrainContext):
        optimizer = ctx.optimizer

        lr_lambda = lambda epoch: (1 - epoch / ctx.cfg.get("epochs")) ** 0.9
        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self, _: TrainContext):
        self.scheduler.step()

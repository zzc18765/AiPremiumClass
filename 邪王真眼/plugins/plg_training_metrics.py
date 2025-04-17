from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class TrainingMetricsPlugin(PluginBase):
    plugin_hooks = {
        PluginType.BATCH_END: "cal_acc",
        PluginType.EPOCH_END: "log_metrics",
    }

    def __init__(self):
        self.train_loss = []
        self.train_acc = []

        self.correct_data = 0
        self.totle_data = 0

    
    def cal_acc(self, ctx: TrainContext):
        labels = ctx.labels
        outputs = ctx.outputs
        bs = ctx.cfg['batch_size']

        self.correct_data += (outputs.argmax(dim=1) == labels).sum().item()
        self.totle_data += bs


    def log_metrics(self, ctx: TrainContext):
        avg_loss = ctx.avg_loss

        self.train_loss.append(avg_loss)

        
        acc = self.correct_data / self.totle_data
        self.correct_data = self.totle_data = 0
        
        msg = {
            "mode": "train",
            "epoch": ctx.epoch + 1,
            "lr": round(ctx.optimizer.param_groups[0]['lr'], 6),
            "loss": round(avg_loss, 6),
            "acc": round(acc, 6),
        }

        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))

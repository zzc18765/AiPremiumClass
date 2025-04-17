import torch

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class ValEvaluationPlugin(PluginBase):
    plugin_hooks = {
        PluginType.EPOCH_END: "evaluate"
    }

    def __init__(self):
        self.history = []
    
    @staticmethod
    def _batch_correct(logits, targets):
        preds = logits.argmax(dim=1)
        return (preds == targets).sum()
    
    def evaluate(self, ctx: TrainContext):
        if (ctx.epoch + 1) % 1 != 0:
            ctx.workspace['val_acc'] = None
            return
        
        model, val_loader, criterion, device = (
            ctx.model, ctx.val_loader, ctx.criterion, ctx.device
        )

        if model is None or val_loader is None:
            raise RuntimeError("Validation plugin requires model and val_loader in context.")

        model.eval()
        total_loss = 0.0
        tot_correct = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)['out'].squeeze(1)

                loss = criterion(logits, targets.long())
                total_loss += loss.item()

                tot_correct += self._batch_correct(logits, targets).item()
        

        n = len(val_loader.dataset)
        val_loss = total_loss / len(val_loader)
        acc = tot_correct / n


        self.history.append({
            "epoch": ctx.epoch,
            "val_loss": val_loss,
            "val_acc": acc,
        })

        msg = {
            "mode":   "val",
            "epoch":  ctx.epoch + 1,
            "lr":     round(ctx.optimizer.param_groups[0]['lr'], 6),
            "loss":   round(val_loss, 6),
            "acc":    round(acc, 6),
        }
        
        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))

        ctx.workspace['val_acc'] = acc

        model.train()

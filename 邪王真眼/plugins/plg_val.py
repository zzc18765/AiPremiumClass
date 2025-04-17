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
    def _batch_metrics(logits, targets):
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long()
        targets = targets.long()

        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()
        tn = ((preds == 0) & (targets == 0)).sum()
        return tp, fp, fn, tn
    
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
        tot_tp = tot_fp = tot_fn = tot_tn = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)['out'].squeeze(1)

                loss = criterion(logits, targets.long())
                total_loss += loss.item()

                tp, fp, fn, tn = self._batch_metrics(logits, targets)
                tot_tp += tp.item()
                tot_fp += fp.item()
                tot_fn += fn.item()
                tot_tn += tn.item()
        

        n = len(val_loader.dataset)
        val_loss = total_loss / len(val_loader)
        acc = (tot_tp + tot_tn) / n
        prec = tot_tp / (tot_tp + tot_fp + 1e-8)
        rec  = tot_tp / (tot_tp + tot_fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)


        self.history.append({
            "epoch": ctx.epoch,
            "val_loss": val_loss,
            "val_acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        msg = {
            "mode":   "val",
            "epoch":  ctx.epoch + 1,
            "lr":     round(ctx.optimizer.param_groups[0]['lr'], 6),
            "loss":   round(val_loss, 4),
            "acc":    round(acc, 4),
        }
        
        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))

        ctx.workspace['val_acc'] = acc
        ctx.workspace['val_f1']  = f1

        model.train()

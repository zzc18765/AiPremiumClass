import torch

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class ModelShapeCheckPlugin(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "check_model_shape"
    }

    def check_model_shape(self, ctx: TrainContext):
        try:
            x, y = next(iter(ctx.train_loader))
            device = ctx.device
            x = x.to(device)
            y = y.to(device)
            ctx.model.eval()

            with torch.no_grad():
                out = ctx.model(x)['out']

            if out.shape[2:] != y.shape[1:]:
                raise ValueError(f"Output shape {out.shape[2:]} != label shape {y.shape[1:]}")

        except Exception as e:
            raise RuntimeError(f"Model test run failed: {type(e).__name__}: {e}")

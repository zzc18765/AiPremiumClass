import os
import uuid

from datetime import datetime
from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class InitTrainerPlugin(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "initialize"
    }

    def initialize(self, ctx: TrainContext):
        short_uuid = str(uuid.uuid4())[:6]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if ctx.cfg["result_path"] is None:
            results_folder = os.path.join("./results", f"{timestamp}_{short_uuid}")
        else:
            results_folder = ctx.cfg["result_path"]
        os.makedirs(results_folder, exist_ok=True)

        if not self.check_key(ctx.workspace, "uuid"):
            ctx.workspace["uuid"] = short_uuid
        ctx.workspace["start_time"] = timestamp
        ctx.workspace["results_folder"] = results_folder

import inspect

from typing import Any, List, Union, Type
from enum import Enum

from .plugins_base import PluginBase
from .plugins_manager import PluginManager


class PluginsItem:
    def __init__(self, plugin_points: List[Type[Enum]]):
        self.plugin_manager = PluginManager()
        self.plugin_points = plugin_points
    
    def build_context(self, context_cls):
        context_kwargs = {}
        for k, v in self.__dict__.items():
            if k in ("context", "plugin_manager", "plugin_points"):
                continue
            if inspect.ismethod(v) or inspect.isfunction(v):
                continue
            context_kwargs[k] = v

        self.context = context_cls(**context_kwargs)
    
    def add_plugins(self, plugins: Union[Type[PluginBase], List[Type[PluginBase]]]):
        if not isinstance(plugins, list):
            plugins = [plugins]

        for plugin_cls in plugins:
            if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, PluginBase):
                raise TypeError(f"Expected PluginBase subclass, got: {plugin_cls}")
            
            self.plugin_manager.register(None, plugin_cls)

    def run_plugins(self, plugin_point: Enum, context: Any):
        self.plugin_manager.execute(plugin_point.name, context)

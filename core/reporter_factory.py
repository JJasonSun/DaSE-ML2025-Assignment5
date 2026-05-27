import importlib
from typing import Type

from reporters.base_reporter import BaseReporter


def load_reporter(reporter_spec: str) -> Type[BaseReporter]:
    if ":" not in reporter_spec:
        raise ValueError("Reporter must use 'module.path:ClassName' format")

    module_path, class_name = reporter_spec.split(":", 1)
    module = importlib.import_module(module_path)
    reporter_class = getattr(module, class_name)

    if not issubclass(reporter_class, BaseReporter):
        raise TypeError(f"{reporter_spec} must inherit from BaseReporter")

    return reporter_class

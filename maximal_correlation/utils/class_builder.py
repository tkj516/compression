import dacite
from typing import Any, Callable, List, Mapping, Optional


class ClassBuilder:
    def __init__(
        self,
        register: Mapping[str, Callable],
        data_class: Optional[Any] = None,
    ):
        self.register = register
        self.data_class = data_class

    def is_registered(self, config: List[Any]) -> bool:
        if config[0] not in self.register:
            return False
        return True

    def build_class(self, config: List[Any], **kwargs):
        if not self.is_registered(config):
            raise ValueError(f"{config[0]} has not been registered!")
        return self.register[config[0]]({**config[1], **kwargs})

    def build_dataclass(self, config: List[Any]):
        if not self.is_registered(config):
            raise ValueError(f"{config[0]} has not been registered!")
        return dacite.from_dict(self.data_class, config[1]) if self.data_class is not None else None

    def build(self, config: List[Any]):
        return self.build_class(config), self.build_dataclass(config)

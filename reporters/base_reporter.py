from abc import ABC, abstractmethod
from typing import Dict


class BaseReporter(ABC):
    @abstractmethod
    def generate(self, data: Dict, output_path: str) -> str:
        """Generate a report from structured evaluation data."""

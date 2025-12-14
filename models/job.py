from dataclasses import dataclass
from typing import List
from .JobOperation import JobOperation

@dataclass
class Job:
    """
    Represents a job consisting of ordered operations.
    """
    id: int
    operations: List[JobOperation]

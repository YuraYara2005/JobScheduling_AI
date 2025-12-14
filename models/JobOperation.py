from dataclasses import dataclass

@dataclass
class JobOperation:
    """
    Represents a single step in a job.
    """
    job_id: int
    machine_id: int
    duration: int
    sequence_order: int

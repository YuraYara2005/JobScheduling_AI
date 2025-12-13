from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ==========================================
# PART 1: Data Structures (The "Nouns")
# ==========================================

@dataclass
class JobOperation:
    """
    Represents a single step in a manufacturing process.
    Example: "Job 1 needs Machine 3 for 50 minutes."
    """
    job_id: int
    machine_id: int
    duration: int
    sequence_order: int  # 0 = First step, 1 = Second step, etc.


@dataclass
class Job:
    """
    Represents a full product that needs to be made.
    Contains a list of operations that must be done in order.
    """
    id: int
    operations: List[JobOperation]


# ==========================================
# PART 2: The Schedule & Constraint Engine
# ==========================================

class Schedule:
    """
    Represents the Solution.
    It holds the plan and strictly enforces the rules (The Constraint Engine).
    """

    def __init__(self, num_machines: int):
        # Correct storage: (start, end, JobOperation)
        self.assignments: Dict[int, List[Tuple[int, int, JobOperation]]] = {
            i: [] for i in range(num_machines)
        }
        self.makespan = 0

    def add_assignment(self, machine_id: int, start_time: int, job_op: JobOperation):
        end_time = start_time + job_op.duration

        # Store the operation object, not job_id
        self.assignments[machine_id].append((start_time, end_time, job_op))

        # Update makespan
        if end_time > self.makespan:
            self.makespan = end_time

    def remove_last_assignment(self, machine_id: int):
        if self.assignments[machine_id]:
            self.assignments[machine_id].pop()
            self._recalculate_makespan()

    def _recalculate_makespan(self):
        max_time = 0
        for tasks in self.assignments.values():
            for _, end, _ in tasks:
                if end > max_time:
                    max_time = end
        self.makespan = max_time

    def is_machine_free(self, machine_id: int, start_time: int, duration: int) -> bool:
        requested_end = start_time + duration
        for (existing_start, existing_end, _) in self.assignments[machine_id]:
            if start_time < existing_end and requested_end > existing_start:
                return False
        return True

    def get_earliest_start_time_for_job(self, job_id: int, sequence_order: int) -> int:
        if sequence_order == 0:
            return 0

        latest_end_time = 0
        for tasks in self.assignments.values():
            for _, end, op in tasks:
                if op.job_id == job_id and end > latest_end_time:
                    latest_end_time = end
        return latest_end_time

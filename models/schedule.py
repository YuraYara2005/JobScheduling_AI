from typing import Dict, List, Tuple
from .JobOperation import JobOperation

class Schedule:
    """
    Represents the solution and enforces scheduling constraints.
    """

    def __init__(self, num_machines: int):
        self.assignments: Dict[int, List[Tuple[int, int, JobOperation]]] = {
            i: [] for i in range(num_machines)
        }
        self.makespan = 0

    def add_assignment(self, machine_id: int, start_time: int, job_op: JobOperation):
        end_time = start_time + job_op.duration
        self.assignments[machine_id].append((start_time, end_time, job_op))
        self.makespan = max(self.makespan, end_time)

    def remove_last_assignment(self, machine_id: int):
        if self.assignments[machine_id]:
            self.assignments[machine_id].pop()
            self._recalculate_makespan()

    def _recalculate_makespan(self):
        self.makespan = max(
            (end for tasks in self.assignments.values() for _, end, _ in tasks),
            default=0
        )

    def is_machine_free(self, machine_id: int, start_time: int, duration: int) -> bool:
        requested_end = start_time + duration
        for s, e, _ in self.assignments[machine_id]:
            if start_time < e and requested_end > s:
                return False
        return True

    def get_earliest_start_time_for_job(self, job_id: int, sequence_order: int) -> int:
        if sequence_order == 0:
            return 0
        return max(
            (end for tasks in self.assignments.values()
             for _, end, op in tasks if op.job_id == job_id),
            default=0
        )

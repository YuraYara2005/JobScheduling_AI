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
        # The main storage: { machine_id: [ (start, end, job_id), ... ] }
        self.assignments: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(num_machines)}
        self.makespan = 0  # The total time the whole project takes

    def add_assignment(self, machine_id: int, start_time: int, job_op: JobOperation):
        """
        Adds a job to the schedule and updates the total Makespan.
        NOTE: This does NOT check validity. Use is_valid_move() first.
        """
        end_time = start_time + job_op.duration

        # Add to the specific machine's timeline
        self.assignments[machine_id].append((start_time, end_time, job_op.job_id))

        # Update the total project time (Makespan)
        if end_time > self.makespan:
            self.makespan = end_time

    def remove_last_assignment(self, machine_id: int, job_op: JobOperation):
        """
        Needed for Backtracking!
        Allows the algorithm to 'undo' a move if it leads to a dead end.
        """
        if self.assignments[machine_id]:
            # Remove the last tuple from the list
            start, end, j_id = self.assignments[machine_id].pop()

            # Recalculate makespan (this is expensive but necessary for accuracy)
            self._recalculate_makespan()

    def _recalculate_makespan(self):
        """Helper to fix the makespan variable after removing a job."""
        max_time = 0
        for m_id, tasks in self.assignments.items():
            for start, end, j_id in tasks:
                if end > max_time:
                    max_time = end
        self.makespan = max_time

    # ==========================================
    # PART 3: The Constraint Engine Logic
    # ==========================================

    def is_machine_free(self, machine_id: int, start_time: int, duration: int) -> bool:
        """
        Constraint 1: Resource Capacity
        Checks if the machine is actually free during the requested time window.
        """
        requested_end = start_time + duration

        for (existing_start, existing_end, _) in self.assignments[machine_id]:
            # Logic: If New starts before Old ends, AND New ends after Old starts -> OVERLAP
            if start_time < existing_end and requested_end > existing_start:
                return False  # Conflict found!
        return True

    def get_earliest_start_time_for_job(self, job_id: int, sequence_order: int) -> int:
        """
        Constraint 2: Sequential Precedence
        A job cannot start Step 2 until Step 1 is finished.
        Returns the timestamp when the PREVIOUS step finished.
        """
        if sequence_order == 0:
            return 0  # First step can start at time 0

        target_prev_sequence = sequence_order - 1

        # Scan all machines to find where the previous step of this job is located
        # (Note: In a huge database this is slow, but for 6x6 it is instant)
        for m_id, tasks in self.assignments.items():
            for start, end, j_id in tasks:
                # We can't strictly check sequence_order here easily without storing it in the tuple.
                # However, since we schedule in order, the last instance of job_id found
                # is guaranteed to be the previous step if the algo is correct.
                if j_id == job_id:
                    # For the purpose of the Backtracking/Cultural algos,
                    # we just need to ensure we don't start before the previous operation of this job ends.
                    # Since we haven't stored 'sequence' in the tuple, we rely on the
                    # fact that 'end' represents the finish of a previous task.
                    # A safer way (if we strictly needed it) is to pass the Job object or store sequence.
                    pass

        # IMPROVED LOGIC FOR ACCURACY:
        # Instead of scanning, we ask: "When did this job last finish something?"
        # We can implement a helper for this or assume the algorithm tracks it.
        # FOR SIMPLICITY in this project:
        # The Algorithm (Backtracking) usually tracks "job_end_times".
        # But let's add a helper here to be safe.

        latest_end_time = 0
        found = False
        for m_id, tasks in self.assignments.items():
            for start, end, j_id in tasks:
                if j_id == job_id:
                    if end > latest_end_time:
                        latest_end_time = end
        return latest_end_time
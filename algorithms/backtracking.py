import copy
from typing import List, Dict

# Import your models
from models import Job, Schedule, JobOperation


class BacktrackingSolver:
    def __init__(self, jobs: List[Job], num_machines: int):
        self.jobs = jobs
        self.num_machines = num_machines

        # Optimization: We store the best solution found so far
        self.best_schedule = None
        self.min_makespan = float('inf')

        # Helper: Track which operation is 'next' for each job
        # Key: job_id, Value: index of the next operation in job.operations list
        self.next_op_indices = {job.id: 0 for job in jobs}

    def solve(self):
        print("Starting Backtracking Search...")

        # Create an empty schedule using your model
        initial_schedule = Schedule(self.num_machines)

        # Calculate total operations to perform (for base case check)
        total_ops = sum(len(j.operations) for j in self.jobs)

        # Start recursion
        self._backtrack(initial_schedule, ops_scheduled_count=0, total_ops=total_ops)

        return self.best_schedule

    def _backtrack(self, current_schedule: Schedule, ops_scheduled_count: int, total_ops: int):
        """
        Recursive function to build the schedule.
        """

        # 1. PRUNING: If current partial schedule is already worse than our best found full schedule, STOP.
        if current_schedule.makespan >= self.min_makespan:
            return

        # 2. BASE CASE: All operations are scheduled
        if ops_scheduled_count == total_ops:
            if current_schedule.makespan < self.min_makespan:
                print(f"New Best Solution Found! Makespan: {current_schedule.makespan}")
                self.min_makespan = current_schedule.makespan
                # Deepcopy is crucial to save the state of this object
                self.best_schedule = copy.deepcopy(current_schedule)
            return

        # 3. IDENTIFY CANDIDATES (The "Branching" Step)
        # We can only schedule the *next* operation for each job (respecting sequence)
        candidates = []
        for job in self.jobs:
            op_index = self.next_op_indices[job.id]
            if op_index < len(job.operations):
                candidates.append(job.operations[op_index])

        # 4. RECURSION LOOP
        for op in candidates:
            # --- A. PREPARE ---
            machine_id = op.machine_id

            # Find the Earliest Valid Start Time
            # Part 1: When is the Job ready? (Previous step finished?)
            job_ready_time = current_schedule.get_earliest_start_time_for_job(op.job_id, op.sequence_order)

            # Part 2: When is the Machine free?
            # We search forward from job_ready_time until we find a gap
            start_time = job_ready_time
            while not current_schedule.is_machine_free(machine_id, start_time, op.duration):
                start_time += 1  # Increment time until we fit

            # --- B. DO (Make the move) ---
            current_schedule.add_assignment(machine_id, start_time, op)
            self.next_op_indices[op.job_id] += 1  # Advance this job's pointer

            # --- C. RECURSE ---
            self._backtrack(current_schedule, ops_scheduled_count + 1, total_ops)

            # --- D. UNDO (Backtrack) ---
            self.next_op_indices[op.job_id] -= 1  # Reset pointer
            current_schedule.remove_last_assignment(machine_id, op)


def run_test():
    print("=== Setting up Test Scenario ===")

    # --- 1. Create Data (The Problem) ---
    # Scenario: 2 Jobs, 2 Machines
    # We manually create the objects instead of using data_loader.py

    # JOB 0: Machine 0 (3 mins) -> Machine 1 (2 mins)
    job0_ops = [
        JobOperation(job_id=0, machine_id=0, duration=3, sequence_order=0),
        JobOperation(job_id=0, machine_id=1, duration=2, sequence_order=1)
    ]
    job0 = Job(id=0, operations=job0_ops)

    # JOB 1: Machine 0 (2 mins) -> Machine 1 (4 mins)
    job1_ops = [
        JobOperation(job_id=1, machine_id=0, duration=2, sequence_order=0),
        JobOperation(job_id=1, machine_id=1, duration=4, sequence_order=1)
    ]
    job1 = Job(id=1, operations=job1_ops)

    all_jobs = [job0, job1]
    num_machines = 2

    # --- 2. Run Backtracking (The Solution) ---
    print("Running Solver...")
    solver = BacktrackingSolver(all_jobs, num_machines)
    best_schedule = solver.solve()

    # --- 3. Validate Results ---
    print("\n=== Final Schedule Results ===")

    if best_schedule is None:
        print("❌ Error: Solver returned None!")
        return

    print(f"Total Makespan: {best_schedule.makespan} (Target: 8)")

    # Print what happened on each machine
    for machine_id, tasks in best_schedule.assignments.items():
        print(f"\nMachine {machine_id}:")
        # Sort tasks by start time just for reading clarity
        sorted_tasks = sorted(tasks, key=lambda x: x[0])
        for start, end, job_id in sorted_tasks:
            print(f"  [Time {start}-{end}] Job {job_id}")

    # Final Check
    if best_schedule.makespan == 8:
        print("\n✅ SUCCESS: Optimal schedule found!")
    else:
        print(f"\n⚠️ WARNING: Found makespan {best_schedule.makespan}, but optimal is 8.")


if __name__ == "__main__":
    run_test()
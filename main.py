import time
import sys
from typing import List, Tuple, Any

from models import Job, JobOperation, Schedule

MAX_BACKTRACKING_JOBS = 5


# monkey patch Schedule.remove_last_assignment ---
_original_remove = Schedule.remove_last_assignment
def _patched_remove_last_assignment(self, machine_id, *args):
    return _original_remove(self, machine_id)
Schedule.remove_last_assignment = _patched_remove_last_assignment

# Try to import solvers
try:
    from backtracking import BacktrackingSolver
except Exception as e:
    BacktrackingSolver = None
    print("Warning: backtracking import failed:", e)

try:
    from cultural_algo import CulturalAlgorithmSolver  # your modified CA
except Exception as e:
    CulturalAlgorithmSolver = None
    print("Warning: cultural_algo import failed:", e)

# Optional visualization function
try:
    from visiualization import draw_gantt_chart
except Exception:
    draw_gantt_chart = None

def read_int(prompt: str, min_val: int = None, default: int = None) -> int:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        try:
            v = int(s)
            if min_val is not None and v < min_val:
                print(f"Please enter an integer >= {min_val}.")
                continue
            return v
        except ValueError:
            print("Please enter a valid integer.")

def read_job_input() -> Tuple[List[Job], int]:
    num_jobs = read_int("Number of jobs: ", min_val=1)
    num_machines = read_int("Number of machines: ", min_val=1)

    jobs: List[Job] = []
    for j in range(num_jobs):
        print(f"\nJob {j}:")
        ops_count = read_int(f"  Number of operations for job {j}: ", min_val=1)
        ops = []
        for op_i in range(ops_count):
            while True:
                raw = input(f"    Operation {op_i} -> enter 'machine,duration' (e.g. 2,5): ").strip()
                try:
                    parts = raw.split(',')
                    if len(parts) != 2:
                        raise ValueError
                    m = int(parts[0].strip())
                    d = int(parts[1].strip())
                    if m < 0 or m >= num_machines:
                        print(f"    machine must be between 0 and {num_machines-1}")
                        continue
                    if d <= 0:
                        print("    duration must be positive")
                        continue
                    ops.append(JobOperation(job_id=j, machine_id=m, duration=d, sequence_order=op_i))
                    break
                except ValueError:
                    print("Invalid format. Try again.")
        jobs.append(Job(id=j, operations=ops))
    return jobs, num_machines

def print_schedule_details(label: str, schedule_like: Any, elapsed: float):
    if schedule_like is None:
        print(f"{label}: no solution produced.")
        return

    # get schedule object and makespan
    if hasattr(schedule_like, "schedule_obj"):
        sch = schedule_like.schedule_obj
        makespan = getattr(schedule_like, "fitness", getattr(sch, "makespan", None))
    else:
        sch = schedule_like
        makespan = getattr(sch, "makespan", None)

    print(f"\n--- {label} summary ---")
    print(f"Makespan: {makespan}")
    print(f"Elapsed time: {elapsed:.4f} s")

    # Print assignments per machine
    assignments = getattr(sch, "assignments", None)
    if assignments:
        for m, tasks in sorted(assignments.items()):
            normalized = []
            for t in sorted(tasks, key=lambda x: x[0]):
                normalized.append((t[0], t[1], t[2]))
            print(f" Machine {m}: {normalized}")

# ---------------------- Runner wrappers ------------------------
def run_backtracking(jobs: List[Job], num_machines: int):
    if BacktrackingSolver is None:
        print("Backtracking solver not available")
        return None, None
    solver = BacktrackingSolver(jobs, num_machines)
    t0 = time.time()
    best_schedule = solver.solve()
    t1 = time.time()
    elapsed = t1 - t0
    # wrap for uniform printing
    class _Wrap:
        def __init__(self, schedule):
            self.schedule_obj = schedule
            self.fitness = getattr(schedule, "makespan", None)
    wrapped = _Wrap(best_schedule) if best_schedule else None
    return wrapped, elapsed

def run_cultural(jobs: List[Job], num_machines: int, pop: int = 40, gen: int = 120):
    if CulturalAlgorithmSolver is None:
        print("Cultural algorithm solver not available")
        return None, None
    solver = CulturalAlgorithmSolver(jobs, num_machines, pop_size=pop, generations=gen)
    t0 = time.time()
    # your CA returns only the best Individual, wrap history as empty list
    best_ind = solver.solve(verbose=True)
    t1 = time.time()
    elapsed = t1 - t0
    return best_ind, elapsed, []

# ---------------------- Main ------------------------
def main():
    print("=== Job Scheduling Runner ===")
    jobs, num_machines = read_job_input()
    print(f"\nLoaded {len(jobs)} jobs, {num_machines} machines.")

    print("\nChoose solver to run:")
    print(" 1) Backtracking")
    print(" 2) Cultural Algorithm")
    print(" 3) Both (Backtracking then Cultural)")
    choice = read_int("Select (1/2/3): ", min_val=1, default=3)

    if choice == 1:
        if len(jobs) > MAX_BACKTRACKING_JOBS:
            print(
                f"\nBacktracking cannot be run for more than {MAX_BACKTRACKING_JOBS} jobs "
                "because it becomes too slow.\n"
                "Please reduce the number of jobs or choose the Cultural Algorithm."
            )
            return

        wrapped, elapsed = run_backtracking(jobs, num_machines)
        print_schedule_details("Backtracking", wrapped, elapsed)
    elif choice == 2:
        pop = read_int("Population size (default 40): ", min_val=1, default=40)
        gen = read_int("Generations (default 120): ", min_val=1, default=120)
        best_ind, elapsed, _ = run_cultural(jobs, num_machines, pop=pop, gen=gen)
        print_schedule_details("Cultural", best_ind, elapsed)
    else:
        if len(jobs) > MAX_BACKTRACKING_JOBS:
            print(
                f"\nSkipping Backtracking: more than {MAX_BACKTRACKING_JOBS} jobs "
                "would take too long."
            )
        else:
            wrapped, bt_elapsed = run_backtracking(jobs, num_machines)
            print_schedule_details("Backtracking", wrapped, bt_elapsed)

        pop = read_int("Population size (default 40): ", min_val=1, default=40)
        gen = read_int("Generations (default 120): ", min_val=1, default=120)
        best_ind, ca_elapsed, _ = run_cultural(jobs, num_machines, pop=pop, gen=gen)
        print_schedule_details("Cultural", best_ind, ca_elapsed)

    print("\nDone.")

if __name__ == "__main__":
    main()


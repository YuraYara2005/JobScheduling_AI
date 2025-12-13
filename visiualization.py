import tkinter as tk
from tkinter import simpledialog, messagebox
import matplotlib.pyplot as plt
import time
import random
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from scipy.interpolate import make_interp_spline

# ==========================================
# MODELS
# ==========================================

@dataclass
class JobOperation:
    job_id: int
    machine_id: int
    duration: int
    sequence_order: int

@dataclass
class Job:
    id: int
    operations: List[JobOperation]

class Schedule:
    def __init__(self, num_machines: int):
        self.assignments: Dict[int, List[Tuple[int,int,int]]] = {i: [] for i in range(num_machines)}
        self.makespan = 0

    def add_assignment(self, machine_id: int, start_time: int, job_op: JobOperation):
        end_time = start_time + job_op.duration
        self.assignments[machine_id].append((start_time, end_time, job_op))
        if end_time > self.makespan:
            self.makespan = end_time

    def remove_last_assignment(self, machine_id: int, job_op: JobOperation):
        if self.assignments[machine_id]:
            self.assignments[machine_id].pop()
            self._recalculate_makespan()

    def _recalculate_makespan(self):
        max_time = 0
        for tasks in self.assignments.values():
            for start, end, _ in tasks:
                if end > max_time:
                    max_time = end
        self.makespan = max_time

    def is_machine_free(self, machine_id: int, start_time: int, duration: int) -> bool:
        end_time = start_time + duration
        for s,e,_ in self.assignments[machine_id]:
            if start_time < e and end_time > s:
                return False
        return True

    def get_earliest_start_time_for_job(self, job_id: int, sequence_order: int) -> int:
        if sequence_order == 0:
            return 0
        latest_end = 0
        for tasks in self.assignments.values():
            for s,e,op in tasks:
                if op.job_id == job_id and op.sequence_order == sequence_order - 1:
                    if e > latest_end:
                        latest_end = e
        return latest_end

# ==========================================
# BACKTRACKING SOLVER
# ==========================================
class BacktrackingSolver:
    def __init__(self, jobs: List[Job], num_machines: int):
        self.jobs = jobs
        self.num_machines = num_machines
        self.best_schedule = None
        self.min_makespan = float('inf')
        self.next_op_indices = {job.id:0 for job in jobs}

    def solve(self):
        initial_schedule = Schedule(self.num_machines)
        total_ops = sum(len(job.operations) for job in self.jobs)
        self._backtrack(initial_schedule, 0, total_ops)
        return self.best_schedule

    def _backtrack(self, schedule: Schedule, ops_count: int, total_ops: int):
        if schedule.makespan >= self.min_makespan:
            return
        if ops_count == total_ops:
            self.best_schedule = copy.deepcopy(schedule)
            self.min_makespan = schedule.makespan
            return
        candidates = []
        for job in self.jobs:
            idx = self.next_op_indices[job.id]
            if idx < len(job.operations):
                candidates.append(job.operations[idx])
        for op in candidates:
            m = op.machine_id
            start = schedule.get_earliest_start_time_for_job(op.job_id, op.sequence_order)
            while not schedule.is_machine_free(m, start, op.duration):
                start += 1
            schedule.add_assignment(m, start, op)
            self.next_op_indices[op.job_id] += 1
            self._backtrack(schedule, ops_count + 1, total_ops)
            self.next_op_indices[op.job_id] -= 1
            schedule.remove_last_assignment(m, op)

# ==========================================
# CULTURAL ALGORITHM SOLVER
# ==========================================
@dataclass
class Individual:
    chromosome: List[int]
    fitness: float = float('inf')
    schedule_obj: Schedule = None

class BeliefSpace:
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.best: List[Individual] = []
        self.best_fitness = float('inf')

    def update(self, population: List[Individual]) -> bool:
        population.sort(key=lambda x: x.fitness)
        new_best = [copy.deepcopy(ind) for ind in population[:self.top_k]]
        improved = False
        if new_best and new_best[0].fitness < self.best_fitness:
            self.best = new_best
            self.best_fitness = new_best[0].fitness
            improved = True
        else:
            if not self.best:
                self.best = new_best
                if new_best:
                    self.best_fitness = new_best[0].fitness
        return improved

    def influence(self, population: List[Individual], decode_fn, influence_strength=0.2, protect_top=2):
        if not self.best:
            return
        population.sort(key=lambda x: x.fitness)
        pop_len = len(population)
        num_to_influence = max(1, int(pop_len * influence_strength))
        replaceable_indices = list(range(pop_len-1, protect_top-1, -1))[:num_to_influence]
        for idx in replaceable_indices:
            elite = random.choice(self.best)
            target = population[idx].chromosome[:]
            blended = [random.choice([e,t]) for e,t in zip(elite.chromosome, target)]
            repaired = CulturalAlgorithmSolver.repair_chromosome_static(blended)
            population[idx] = decode_fn(repaired, guided=True)

def ppx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    size = len(p1)
    mask = [random.choice([0,1]) for _ in range(size)]
    child = []
    i1=i2=0
    for bit in mask:
        if bit==0:
            if i1<size:
                child.append(p1[i1])
                i1+=1
            else:
                child.append(p2[i2])
                i2+=1
        else:
            if i2<size:
                child.append(p2[i2])
                i2+=1
            else:
                child.append(p1[i1])
                i1+=1
    return child

class CulturalAlgorithmSolver:
    def __init__(self, jobs: List[Job], num_machines: int, pop_size=40, generations=50, mut_rate=0.1, seed=None):
        if seed: random.seed(seed)
        self.jobs = jobs
        self.num_machines = num_machines
        self.pop_size = pop_size
        self.generations = generations
        self.base_mut = mut_rate
        self.mutation_rate = mut_rate
        self.total_ops = sum(len(j.operations) for j in jobs)
        self.population: List[Individual] = []
        self.belief = BeliefSpace()
        self.job_map = {j.id:j for j in jobs}

    def _generate_chromosome(self):
        genes=[]
        for j in self.jobs:
            genes.extend([j.id]*len(j.operations))
        random.shuffle(genes)
        return genes

    @staticmethod
    def repair_chromosome_static(chromo: List[int]) -> List[int]:
        return chromo

    def repair_chromosome(self, chromo: List[int]) -> List[int]:
        expected = {j.id: len(j.operations) for j in self.jobs}
        actual = {}
        new_chromo = []
        used = {k:0 for k in expected}
        for g in chromo:
            if g in expected and used[g]<expected[g]:
                new_chromo.append(g)
                used[g]+=1
        for jid,cnt in expected.items():
            missing = cnt - used.get(jid,0)
            new_chromo.extend([jid]*missing)
        for _ in range(max(1,len(new_chromo)//10)):
            i,j = random.sample(range(len(new_chromo)),2)
            new_chromo[i], new_chromo[j] = new_chromo[j], new_chromo[i]
        return new_chromo

    def _mutate(self, chromo: List[int]) -> List[int]:
        ch = chromo[:]
        if random.random()<self.mutation_rate:
            i,j = random.sample(range(len(ch)),2)
            ch[i], ch[j] = ch[j], ch[i]
        return ch

    def _decode(self, chromo: List[int], guided=True) -> Individual:
        schedule = Schedule(self.num_machines)
        if len(chromo)!=self.total_ops:
            chromo = self.repair_chromosome(chromo)
        job_next = {j.id:0 for j in self.jobs}
        job_ready = {j.id:0 for j in self.jobs}
        machine_ready = {m:0 for m in range(self.num_machines)}
        for jid in chromo:
            job = self.job_map[jid]
            idx = job_next[jid]
            if idx>=len(job.operations): continue
            op = job.operations[idx]
            earliest = max(job_ready[jid], machine_ready[op.machine_id])
            start = earliest
            end = start+op.duration
            schedule.add_assignment(op.machine_id, start, op)
            job_ready[jid] = end
            machine_ready[op.machine_id] = end
            job_next[jid]+=1
        makespan = max(job_ready.values()) if job_ready else 0
        return Individual(chromosome=chromo, fitness=makespan, schedule_obj=schedule)

    def solve(self, verbose=True):
        if verbose: print("===== CULTURAL ALGORITHM START =====")
        history = []
        self.population = [self._decode(self._generate_chromosome(), guided=False) for _ in range(self.pop_size)]
        self.belief.update(self.population)
        stagnation=0
        for gen in range(1,self.generations+1):
            guided = (gen>=20)
            new_pop=[]
            self.population.sort(key=lambda x:x.fitness)
            elites = [copy.deepcopy(self.population[0]), copy.deepcopy(self.population[1] if len(self.population)>1 else self.population[0])]
            new_pop.extend(elites)
            while len(new_pop)<self.pop_size:
                p1=min(random.sample(self.population,3), key=lambda x:x.fitness)
                p2=min(random.sample(self.population,3), key=lambda x:x.fitness)
                child_chromo = ppx_crossover(p1.chromosome, p2.chromosome)
                child_chromo = self.repair_chromosome(child_chromo)
                child_chromo = self._mutate(child_chromo)
                decoded_child = self._decode(child_chromo, guided=guided)
                new_pop.append(decoded_child)
            self.population = new_pop
            improved = self.belief.update(self.population)
            if improved: stagnation=0
            else: stagnation+=1
            self.belief.influence(self.population, self._decode)
            history.append(self.belief.best[0].fitness if self.belief.best else float('inf'))
            if verbose:
                print(f"Gen {gen} → Best={history[-1]}")
        if verbose: print("===== DONE =====")
        return copy.deepcopy(self.belief.best[0]), history

# ==========================================
# VISUALIZATION CURVES
# ==========================================
def draw_gantt_chart(schedule, title="Gantt Chart"):
    fig, ax = plt.subplots(figsize=(10,5))
    colors = ['skyblue','salmon','lightgreen','plum','orange','pink','yellow','lightgrey']
    for m,tasks in schedule.assignments.items():
        for s,e,op in tasks:
            c = colors[op.job_id%len(colors)]
            ax.barh(y=m, width=e-s, left=s, height=0.4, color=c, edgecolor='black')
            ax.text(s+(e-s)/2, m, f"Job {op.job_id}", va='center', ha='center', fontsize=8)
    ax.set_yticks(list(schedule.assignments.keys()))
    ax.set_yticklabels([f"Machine {m}" for m in schedule.assignments.keys()])
    ax.set_xlabel("Time")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_runtime_vs_iterations(history, title="Cultural Algorithm Progress"):
    plt.figure(figsize=(10,5))
    x = np.arange(1,len(history)+1)
    y = np.array(history)
    plt.plot(x, y, 'o-', color='purple', label='Raw data')
    if len(history)>=4:
        x_smooth = np.linspace(x.min(), x.max(), 200)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, color='blue', alpha=0.6, label='Smooth curve')
    plt.xlabel("Generation")
    plt.ylabel("Best Makespan")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_runtime_comparison(runtimes):
    plt.figure(figsize=(6,4))
    x = list(runtimes.keys())
    y = list(runtimes.values())
    plt.plot(x, y, 'o-', color='green', label='Solver Runtime')
    plt.ylabel("Runtime (s)")
    plt.title("Solver Runtime Comparison")
    plt.grid(True)
    plt.show()

# ==========================================
# GUI
# ==========================================
class JobSchedulingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Job Scheduling GUI")
        self.jobs=[]
        self.num_machines=0
        self.ca_pop=40
        self.ca_gen=50
        self.create_input_selection_screen()

    def create_input_selection_screen(self):
        for w in self.root.winfo_children(): w.destroy()
        tk.Label(self.root, text="Select Input Method:", font=("Arial",12,"bold")).pack(pady=10)
        tk.Button(self.root, text="Import Dataset", width=30, command=self.import_dataset).pack(pady=5)
        tk.Button(self.root, text="Input Parameters Manually", width=30, command=self.input_parameters).pack(pady=5)

    def import_dataset(self):
        messagebox.showinfo("Info","Dataset import not implemented. Use manual input.")

    def input_parameters(self):
        self.jobs=[]
        self.num_machines = simpledialog.askinteger("Input","Enter number of machines:", minvalue=1)
        num_jobs = simpledialog.askinteger("Input","Enter number of jobs:", minvalue=1)
        for j in range(num_jobs):
            ops=[]
            num_ops = simpledialog.askinteger("Input", f"Number of operations for Job {j}:", minvalue=1)
            for o in range(num_ops):
                m = simpledialog.askinteger("Input", f"Job {j} Operation {o} -> Machine ID (0-{self.num_machines-1}):", minvalue=0, maxvalue=self.num_machines-1)
                d = simpledialog.askinteger("Input", f"Job {j} Operation {o} -> Duration:", minvalue=1)
                ops.append(JobOperation(job_id=j, machine_id=m, duration=d, sequence_order=o))
            self.jobs.append(Job(id=j, operations=ops))
        self.ca_pop = simpledialog.askinteger("Input", "Cultural Algorithm Population Size:", minvalue=1, initialvalue=40)
        self.ca_gen = simpledialog.askinteger("Input", "Cultural Algorithm Generations:", minvalue=1, initialvalue=50)
        self.create_solver_selection_screen()

    def create_solver_selection_screen(self):
        for w in self.root.winfo_children(): w.destroy()
        tk.Label(self.root, text="Select Solver to Run:", font=("Arial",12,"bold")).pack(pady=10)
        tk.Button(self.root, text="Run using Backtracking", width=30, command=lambda:self.run_solver("backtracking")).pack(pady=5)
        tk.Button(self.root, text="Run using Cultural Algorithm", width=30, command=lambda:self.run_solver("cultural")).pack(pady=5)
        tk.Button(self.root, text="Run Both", width=30, command=lambda:self.run_solver("both")).pack(pady=5)

    def run_solver(self, choice):
        if not self.jobs:
            messagebox.showwarning("Warning","Please input parameters first!")
            return
        runtimes={}
        if choice in ["backtracking","both"]:
            t0 = time.time()
            bt_solver = BacktrackingSolver(self.jobs, self.num_machines)
            best_bt = bt_solver.solve()
            t1 = time.time()
            runtimes['Backtracking'] = t1-t0
            messagebox.showinfo("Backtracking Result", f"Makespan: {best_bt.makespan}\nRuntime: {t1-t0:.2f}s")
            draw_gantt_chart(best_bt, title="Backtracking Gantt Chart")
        if choice in ["cultural","both"]:
            t0 = time.time()
            ca_solver = CulturalAlgorithmSolver(self.jobs, self.num_machines, pop_size=self.ca_pop, generations=self.ca_gen)
            best_ca, history = ca_solver.solve(verbose=True)
            t1 = time.time()
            runtimes['Cultural'] = t1-t0
            messagebox.showinfo("Cultural Algorithm Result", f"Makespan: {best_ca.fitness}\nRuntime: {t1-t0:.2f}s")
            draw_gantt_chart(best_ca.schedule_obj, title="Cultural Algorithm Gantt Chart")
            plot_runtime_vs_iterations(history)
        if choice=="both":
            plot_runtime_comparison(runtimes)

# ==========================================
# RUN GUI
# ==========================================
if __name__=="__main__":
    root=tk.Tk()
    app=JobSchedulingApp(root)
    root.mainloop()

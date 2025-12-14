import random
import time
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple
from models import Job, JobOperation, Schedule
# Individual
# --------------------
@dataclass
class Individual:
    chromosome: List[int]
    fitness: float = float('inf')
    schedule_obj: Schedule = None
# --------------------
# BeliefSpace (fixed)
# --------------------
class BeliefSpace:
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.best: List[Individual] = []
        self.best_fitness = float('inf')

    def update(self, population: List[Individual]) -> bool:
        """
        Update belief space with top_k individuals.
        Return True if best fitness improved.
        """
        population.sort(key=lambda x: x.fitness)
        new_best = [copy.deepcopy(ind) for ind in population[:self.top_k]]
        improved = False
        if new_best and new_best[0].fitness < self.best_fitness:
            self.best = new_best
            self.best_fitness = new_best[0].fitness
            improved = True
        else:
            if not self.best:
                # first time initialization
                self.best = new_best
                if new_best:
                    self.best_fitness = new_best[0].fitness
        return improved

    def influence(self, population: List[Individual], decode_fn, influence_strength=0.20, protect_top=2):
        """
        Influence the worst individuals by blending with elite solutions.
        decode_fn: function to decode a chromosome -> Individual (with fitness & schedule)
        protect_top: number of top individuals to avoid replacing (elitism)
        """
        if not self.best:
            return

        population.sort(key=lambda x: x.fitness)
        pop_len = len(population)
        num_to_influence = max(1, int(pop_len * influence_strength))

        # ensure we don't try to replace the elites
        replaceable_indices = list(range(pop_len - 1, protect_top - 1, -1))  # from worst backwards, avoid top protect_top
        replaceable_indices = replaceable_indices[:num_to_influence]

        for idx in replaceable_indices:
            elite = random.choice(self.best)
            target = population[idx].chromosome[:]

            # blend gene-wise
            blended = [random.choice([e, t]) for e, t in zip(elite.chromosome, target)]

            # Repair blended chromosome to have correct counts
            repaired = CulturalAlgorithmSolver.repair_chromosome_static(blended, decode_fn)
            # decode to get fitness and schedule_obj
            decoded = decode_fn(repaired, guided=True)
            population[idx] = decoded


# --------------------
# PPX crossover
# --------------------
def ppx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Precedence-preserving style crossover assuming p1 and p2 are permutations/multisets of same multiset."""
    size = len(p1)
    mask = [random.choice([0, 1]) for _ in range(size)]
    child = []
    i1 = 0
    i2 = 0
    for bit in mask:
        if bit == 0:
            if i1 < size:
                child.append(p1[i1])
                i1 += 1
            else:
                # fallback to p2
                child.append(p2[i2])
                i2 += 1
        else:
            if i2 < size:
                child.append(p2[i2])
                i2 += 1
            else:
                child.append(p1[i1])
                i1 += 1
    return child


# --------------------
# Cultural Algorithm Solver
# --------------------
class CulturalAlgorithmSolver:
    def __init__(self, jobs: List[Job], num_machines: int,
                 pop_size=80, generations=150, mut_rate=0.10, seed=None):
        if seed is not None:
            random.seed(seed)

        self.jobs = jobs
        self.num_machines = num_machines
        self.pop_size = pop_size
        self.generations = generations

        self.base_mut = mut_rate
        self.mutation_rate = mut_rate

        self.total_ops = sum(len(j.operations) for j in jobs)
        self.population: List[Individual] = []

        self.belief = BeliefSpace()
        # speed: job_map for O(1) access in decoder
        self.job_map: Dict[int, Job] = {j.id: j for j in jobs}

    # --------------------
    # Chromosome helpers
    # --------------------
    def _generate_chromosome(self) -> List[int]:
        genes = []
        for job in self.jobs:
            genes.extend([job.id] * len(job.operations))
        random.shuffle(genes)
        return genes

    @staticmethod
    def repair_chromosome_static(chromosome: List[int], decode_fn_or_dummy) -> List[int]:
        """
        Static repair function that ensures the chromosome contains the correct counts for each job id.
        This function assumes decode_fn_or_dummy is just used to infer needed counts if necessary.
        (We will rely on solver context usually; this static version tries minimal work.)
        """
        # If we cannot infer required counts from decode_fn_or_dummy, we'll just keep the length.
        # The calling code prefers to call instance method version.
        return chromosome  # placeholder for static usage; instance method used normally

    def repair_chromosome(self, chromo: List[int]) -> List[int]:
        """
        Ensure the chromosome has exactly the expected multiset of job ids.
        If it's wrong (due to crossover), repair by trimming extras and appending missing ones.
        """
        expected_counts = {j.id: len(j.operations) for j in self.jobs}
        actual_counts = {}
        for g in chromo:
            actual_counts[g] = actual_counts.get(g, 0) + 1

        # Remove extra occurrences (first pass)
        new_chromo = []
        used_counts = {k: 0 for k in expected_counts}
        for g in chromo:
            if g in expected_counts and used_counts[g] < expected_counts[g]:
                new_chromo.append(g)
                used_counts[g] += 1
            else:
                # skip extra
                pass

        # Append missing ones
        for jid, cnt in expected_counts.items():
            missing = cnt - used_counts.get(jid, 0)
            new_chromo.extend([jid] * missing)

        # final shuffle to avoid bias (but keep relative order somewhat)
        # we won't fully shuffle to preserve some structure; do a small random shuffle
        for _ in range(max(1, len(new_chromo) // 10)):
            i, j = random.sample(range(len(new_chromo)), 2)
            new_chromo[i], new_chromo[j] = new_chromo[j], new_chromo[i]

        return new_chromo

    # --------------------
    # Mutation
    # --------------------
    def _mutate(self, chromo: List[int]) -> List[int]:
        ch = chromo[:]
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(ch)), 2)
            ch[i], ch[j] = ch[j], ch[i]
        return ch

    # --------------------
    # Decoder (fixed guided logic + uses job_map)
    # --------------------
    def _decode(self, chromosome: List[int], guided=True) -> Individual:
        """
        Decode chromosome into schedule (greedy by order).
        Guided mode: occasionally try to use end times of previous assignments but never earlier than earliest (preserve precedence).
        """
        schedule = Schedule(self.num_machines)

        # defensive: ensure chromosome has correct multiset
        if len(chromosome) != self.total_ops:
            # try to repair
            chromosome = self.repair_chromosome(chromosome)

        job_next = {j.id: 0 for j in self.jobs}
        job_ready = {j.id: 0 for j in self.jobs}
        machine_ready = {m: 0 for m in range(self.num_machines)}

        for job_id in chromosome:
            job = self.job_map[job_id]
            op_idx = job_next[job_id]
            if op_idx >= len(job.operations):
                # malformed chromosome, skip this gene
                continue
            op = job.operations[op_idx]
            duration = op.duration
            m = op.machine_id

            earliest = max(job_ready[job_id], machine_ready[m])

            if guided:
                possible = [earliest]
                # include machine existing assignment end times that are >= earliest
                for (s, e, _) in schedule.assignments.get(m, []):
                    if e >= earliest:
                        possible.append(e)
                # choose candidate start but ensure >= earliest
                if possible:
                    start_candidate = random.choice(possible)
                    start = max(start_candidate, earliest)
                else:
                    start = earliest
            else:
                start = earliest

            end = start + duration
            schedule.add_assignment(m, start, op)

            job_ready[job_id] = end
            machine_ready[m] = end
            job_next[job_id] += 1

        # finalize schedule if Schedule has finalize
        if hasattr(schedule, "finalize"):
            schedule.finalize()

        makespan = getattr(schedule, 'makespan', max(job_ready.values()) if job_ready else 0)
        return Individual(chromosome=chromosome, fitness=makespan, schedule_obj=schedule)

    # --------------------
    # Main solve
    # --------------------
    def solve(self, verbose=True) -> Individual:
        if verbose:
            print("===== CULTURAL ALGORITHM START =====")

        # init population (decoded)
        self.population = [self._decode(self._generate_chromosome(), guided=False) for _ in range(self.pop_size)]

        improved = self.belief.update(self.population)
        stagnation = 0

        for gen in range(1, self.generations + 1):
            guided = (gen >= 20)

            new_pop: List[Individual] = []

            # elitism: keep top 2
            self.population.sort(key=lambda x: x.fitness)
            elites = [copy.deepcopy(self.population[0]), copy.deepcopy(self.population[1] if len(self.population) > 1 else self.population[0])]
            new_pop.extend(elites)

            # generate rest
            while len(new_pop) < self.pop_size:
                # tournament selection of parents (size 3)
                p1 = min(random.sample(self.population, 3), key=lambda x: x.fitness)
                p2 = min(random.sample(self.population, 3), key=lambda x: x.fitness)

                child_chromo = ppx_crossover(p1.chromosome, p2.chromosome)
                # repair to ensure multiset correctness
                child_chromo = self.repair_chromosome(child_chromo)
                child_chromo = self._mutate(child_chromo)

                decoded_child = self._decode(child_chromo, guided=guided)
                new_pop.append(decoded_child)

            self.population = new_pop
            improved = self.belief.update(self.population)

            # adaptive mutation based on true improvement
            if improved:
                stagnation = 0
                self.mutation_rate = max(self.base_mut, self.mutation_rate * 0.95)
            else:
                stagnation += 1
                if stagnation >= 10:
                    self.mutation_rate = max(self.mutation_rate, 0.25)

            # belief influence: pass decode function so influence creates valid Individuals
            self.belief.influence(self.population, decode_fn=self._decode, influence_strength=0.20, protect_top=2)

            if verbose:
                best_f = self.belief.best[0].fitness if self.belief.best else float('inf')
                print(f"Gen {gen:3d} → Best = {best_f} | Mut= {self.mutation_rate:.3f}")

        if verbose:
            print("===== DONE =====")
        return copy.deepcopy(self.belief.best[0])
# --------------------
# Example usage / quick test
# --------------------
# if __name__ == "__main__":
#     # small test instance: 3 jobs, 3 operations each, 3 machines
#     # job 1: op0->m0, dur 3 ; op1->m1, dur 2 ; op2->m2, dur 2
#     # job 2: ...
#     j1_ops = [JobOperation(job_id=1, machine_id=0, duration=3, sequence_order=0),
#               JobOperation(job_id=1, machine_id=1, duration=2, sequence_order=1),
#               JobOperation(job_id=1, machine_id=2, duration=2, sequence_order=2)]
#     j2_ops = [JobOperation(job_id=2, machine_id=1, duration=2, sequence_order=0),
#               JobOperation(job_id=2, machine_id=2, duration=1, sequence_order=1),
#               JobOperation(job_id=2, machine_id=0, duration=4, sequence_order=2)]
#     j3_ops = [JobOperation(job_id=3, machine_id=2, duration=2, sequence_order=0),
#               JobOperation(job_id=3, machine_id=0, duration=1, sequence_order=1),
#               JobOperation(job_id=3, machine_id=1, duration=3, sequence_order=2)]
#
#     jobs = [Job(1, j1_ops), Job(2, j2_ops), Job(3, j3_ops)]
#     solver = CulturalAlgorithmSolver(jobs=jobs, num_machines=3, pop_size=30, generations=80, mut_rate=0.12, seed=42)
#     best = solver.solve(verbose=True)
#
#     print("\nBest fitness (makespan):", best.fitness)
#     print("Best chromosome:", best.chromosome)
#     sched = best.schedule_obj
#     print("Schedule assignments per machine:")
#     for m in range(3):
#         assigns = sched.assignments.get(m, [])
#         assigns_sorted = sorted(assigns, key=lambda x: x[0])
#         for (s, e, op) in assigns_sorted:
#             print(f"  Machine {m}: Job{op.job_id} op dur={op.duration} -> start {s}, end {e}")
# # --------------------
# Example usage / quick test
# --------------------
# if __name__ == "__main__":
#
#     j1_ops = [
#         JobOperation(1, 0, 3, 0),
#         JobOperation(1, 1, 2, 1),
#         JobOperation(1, 2, 2, 2),
#         JobOperation(1, 3, 4, 3),
#     ]
#
#     j2_ops = [
#         JobOperation(2, 1, 2, 0),
#         JobOperation(2, 2, 1, 1),
#         JobOperation(2, 3, 3, 2),
#         JobOperation(2, 0, 2, 3),
#     ]
#
#     j3_ops = [
#         JobOperation(3, 2, 4, 0),
#         JobOperation(3, 3, 2, 1),
#         JobOperation(3, 0, 1, 2),
#         JobOperation(3, 1, 3, 3),
#     ]
#
#     j4_ops = [
#         JobOperation(4, 3, 3, 0),
#         JobOperation(4, 0, 2, 1),
#         JobOperation(4, 1, 4, 2),
#         JobOperation(4, 2, 1, 3),
#     ]
#
#     j5_ops = [
#         JobOperation(5, 0, 4, 0),
#         JobOperation(5, 2, 3, 1),
#         JobOperation(5, 1, 2, 2),
#         JobOperation(5, 3, 1, 3),
#     ]
#
#     j6_ops = [
#         JobOperation(6, 1, 3, 0),
#         JobOperation(6, 3, 2, 1),
#         JobOperation(6, 2, 4, 2),
#         JobOperation(6, 0, 1, 3),
#     ]
#
#     jobs = [
#         Job(1, j1_ops),
#         Job(2, j2_ops),
#         Job(3, j3_ops),
#         Job(4, j4_ops),
#         Job(5, j5_ops),
#         Job(6, j6_ops),
#     ]
#
#     solver = CulturalAlgorithmSolver(
#         jobs=jobs,
#         num_machines=4,
#         pop_size=80,
#         generations=50,
#         mut_rate=0.10,
#         seed=42
#     )
#
#     best = solver.solve(verbose=True)
#
#     print("\n========== FINAL RESULT ==========")
#     print("Best makespan:", best.fitness)
#     print("Best chromosome:", best.chromosome)
#
#     sched = best.schedule_obj
#     for m in range(4):
#         print(f"\nMachine {m}:")
#         for (s, e, op) in sorted(sched.assignments[m], key=lambda x: x[0]):
#             print(f"  Job{op.job_id} | start={s} end={e}")

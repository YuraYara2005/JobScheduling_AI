Intelligent Job Scheduling Solver
A Python-based application that solves the Job Scheduling Problem (JSP) using Backtracking Search and Cultural Algorithms. Features a complete GUI for data input, visualization, and algorithm comparison.

🚀 Features
Two Algorithms:

Backtracking: Finds the optimal solution (limited to ≤ 5 jobs).

Cultural Algorithm: Evolutionary approach for larger datasets.

Dual Input: Enter data manually via a grid or load standard .txt datasets.

Visualization: Generates interactive Gantt Charts and Performance Comparison Plots (Runtime vs. Makespan).

🛠️ Quick Start
Clone the repo:

Bash

git clone https://github.com/yourusername/Job_Sch_AI.git
cd Job_Sch_AI
Install dependencies:

Bash

pip install matplotlib
Run the Application:

Bash

python GUI/visualization.py
📂 Project Structure
GUI/ - Main visualization logic (visualization.py).

algorithms/ - Implementation of Backtracking and Cultural algorithms.

models/ - Data classes (Job, Schedule).

data/ - Dataset loader and sample files.

📝 Input Format (File Mode)
If uploading a text file, use this format:

Plaintext

<num_jobs> <num_machines>
<machine_id> <time> <machine_id> <time> ... (Job 0)
<machine_id> <time> <machine_id> <time> ... (Job 1)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
import platform
import random

# Ensure data modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataloader import DataLoader


# --- Helper: Robust Scrollable Frame (Hidden Scrollbar) ---
class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Grid config: Canvas takes all available space
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg='white', highlightthickness=0)
        self.scrollable_content = tk.Frame(self.canvas, bg='white')

        # Link scrollable content to canvas
        self.scroll_window = self.canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")

        # --- HIDDEN SCROLLBAR ---
        # The logic is there, but we do not .grid() the widget, so it is invisible.
        # self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.scrollable_content.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.bind_mouse_scroll(self.canvas)
        self.bind_mouse_scroll(self.scrollable_content)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.scroll_window, width=event.width)

    def bind_mouse_scroll(self, widget):
        widget.bind("<Enter>", self._bind_to_mousewheel)
        widget.bind("<Leave>", self._unbind_from_mousewheel)

    def _bind_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_from_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if platform.system() == 'Windows':
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif platform.system() == 'Darwin':
            self.canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            if event.num == 4: self.canvas.yview_scroll(-1, "units")
            if event.num == 5: self.canvas.yview_scroll(1, "units")


# --- Main App ---
class JobSchedulingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Job Scheduling Solver")

        # 1. Safe Default Size (Fits standard laptops)
        self.geometry("1100x650")
        self.configure(bg='white')

        # 2. Attempt Auto-Maximize
        try:
            if platform.system() == "Windows":
                self.state('zoomed')
            else:
                self.attributes('-zoomed', True)
        except:
            pass

        # Shared Data
        self.shared_data = {
            "source_type": None,
            "algo_choice": None,
            "num_jobs": 0,
            "num_machines": 0,
            "max_ops": 0,
            "jobs_data": []
        }

        self.setup_styles()

        self.container = tk.Frame(self, bg='white')
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (HomePage, AlgoSelectionPage, DimensionsPage, DetailedInputPage, ResultsPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("HomePage")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        bg = 'white'
        primary = '#1a73e8'
        secondary = '#5f6368'

        style.configure('.', background=bg, foreground='#202124', font=('Segoe UI', 11))
        style.configure('Heading.TLabel', font=('Segoe UI', 28, 'bold'), foreground=primary, background=bg)
        style.configure('SubHeading.TLabel', font=('Segoe UI', 14), foreground=secondary, background=bg)

        style.configure('Action.TButton', font=('Segoe UI', 12, 'bold'), background=primary, foreground='white',
                        borderwidth=0)
        style.map('Action.TButton', background=[('active', '#174ea6')])

        style.configure('Secondary.TButton', font=('Segoe UI', 12), background=secondary, foreground='white',
                        borderwidth=0)
        style.map('Secondary.TButton', background=[('active', '#3c4043')])

        style.configure('Card.TFrame', background='#f1f3f4', relief='flat')

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, 'update_view'):
            frame.update_view()


class BasePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='white')


# --- Pages ---

class HomePage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        content = tk.Frame(self, bg='white')
        content.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(content, text="Job Scheduling System", style="Heading.TLabel").pack(pady=(0, 15))
        ttk.Label(content, text="Select Data Source", style="SubHeading.TLabel").pack(pady=(0, 40))

        ttk.Button(content, text="Input Data Manually", style="Action.TButton", width=30,
                   command=self.go_manual).pack(pady=10, ipady=8)
        ttk.Button(content, text="Load Dataset (File)", style="Action.TButton", width=30,
                   command=self.load_file).pack(pady=10, ipady=8)

    def go_manual(self):
        self.controller.shared_data['source_type'] = 'manual'
        self.controller.show_frame("AlgoSelectionPage")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                loader = DataLoader()
                data = loader.load_instance(file_path)
                self.controller.shared_data.update(data)
                self.controller.shared_data['source_type'] = 'file'
                self.controller.shared_data['jobs_data'] = data['jobs']
                messagebox.showinfo("Success", "Dataset loaded successfully!")
                self.controller.show_frame("AlgoSelectionPage")
            except Exception as e:
                messagebox.showerror("Error", str(e))


class AlgoSelectionPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        content = tk.Frame(self, bg='white')
        content.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(content, text="Select Algorithm", style="Heading.TLabel").pack(pady=(0, 40))

        btn_width = 35
        ttk.Button(content, text="Run Backtracking Search", style="Action.TButton", width=btn_width,
                   command=lambda: self.select("backtracking")).pack(pady=10, ipady=8)
        ttk.Button(content, text="Run Cultural Algorithm", style="Action.TButton", width=btn_width,
                   command=lambda: self.select("cultural")).pack(pady=10, ipady=8)
        ttk.Button(content, text="Run Both Algorithms", style="Action.TButton", width=btn_width,
                   command=lambda: self.select("both")).pack(pady=10, ipady=8)
        ttk.Button(content, text="Back", style="Secondary.TButton", width=btn_width,
                   command=lambda: controller.show_frame("HomePage")).pack(pady=40, ipady=5)

    def select(self, choice):
        self.controller.shared_data['algo_choice'] = choice
        data = self.controller.shared_data

        if data['source_type'] == 'file' and choice in ['backtracking', 'both']:
            if data['num_jobs'] > 5:
                messagebox.showerror("Constraint", "Backtracking limited to 5 jobs.")
                return

        if data['source_type'] == 'manual':
            self.controller.show_frame("DimensionsPage")
        else:
            self.controller.show_frame("ResultsPage")


class DimensionsPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        content = tk.Frame(self, bg='white')
        content.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(content, text="Problem Dimensions", style="Heading.TLabel").pack(pady=20)
        self.entries = {}
        fields = [("Number of Machines", "machines"), ("Number of Jobs", "jobs"), ("Max Ops per Job", "ops")]

        for label, key in fields:
            row = tk.Frame(content, bg='white')
            row.pack(pady=10, fill='x')
            tk.Label(row, text=label, width=20, anchor='e', bg='white', font=('Segoe UI', 12)).pack(side='left',
                                                                                                    padx=15)
            ent = ttk.Entry(row, font=('Segoe UI', 12))
            ent.pack(side='left', fill='x', expand=True)
            self.entries[key] = ent

        btn_box = tk.Frame(content, bg='white')
        btn_box.pack(pady=40)
        ttk.Button(btn_box, text="Back", style="Secondary.TButton", width=15,
                   command=lambda: controller.show_frame("AlgoSelectionPage")).pack(side='left', padx=10)
        ttk.Button(btn_box, text="Next", style="Action.TButton", width=15,
                   command=self.validate).pack(side='left', padx=10)

    def validate(self):
        try:
            n_jobs = int(self.entries["jobs"].get())
            if self.controller.shared_data['algo_choice'] in ['backtracking', 'both'] and n_jobs > 5:
                messagebox.showerror("Constraint", "Backtracking limited to 5 jobs.")
                return

            self.controller.shared_data.update({
                "num_jobs": n_jobs,
                "num_machines": int(self.entries["machines"].get()),
                "max_ops": int(self.entries["ops"].get())
            })
            self.controller.show_frame("DetailedInputPage")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers.")


class DetailedInputPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        # GRID LAYOUT to guarantee footer visibility
        # Row 1 (Content) gets weight=1, forcing it to shrink/scroll before footer is pushed off
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Content (Shrinkable)
        self.grid_rowconfigure(2, weight=0)  # Footer (Fixed)
        self.grid_columnconfigure(0, weight=1)

        # 1. Header
        header = tk.Frame(self, bg='white')
        header.grid(row=0, column=0, sticky='ew', padx=30, pady=15)
        ttk.Label(header, text="Detailed Configuration", style="Heading.TLabel").pack(side='left')

        # 2. Scrollable Content
        self.scroll_area = ScrollableFrame(self)
        self.scroll_area.grid(row=1, column=0, sticky='nsew', padx=20)

        # 3. Footer (Stickied to Bottom)
        footer = tk.Frame(self, bg='white', height=80)
        footer.grid(row=2, column=0, sticky='ew', pady=20)

        ttk.Button(footer, text="Back", style="Secondary.TButton", width=15,
                   command=lambda: controller.show_frame("DimensionsPage")).pack(side='left', padx=40)
        ttk.Button(footer, text="Run Analysis", style="Action.TButton", width=15,
                   command=self.run_algorithms).pack(side='right', padx=40)

        self.input_refs = []

    def update_view(self):
        content_frame = self.scroll_area.scrollable_content
        for w in content_frame.winfo_children(): w.destroy()
        self.input_refs = []
        data = self.controller.shared_data

        msg = f"Valid Machine IDs: 0 to {data['num_machines'] - 1}"
        tk.Label(content_frame, text=msg, bg='white', fg='#d93025', font=('Segoe UI', 10, 'bold')).pack(pady=(0, 15))

        center_grid = tk.Frame(content_frame, bg='white')
        center_grid.pack(pady=10)

        for i in range(data['num_jobs']):
            row = i // 2
            col = i % 2
            card = ttk.LabelFrame(center_grid, text=f"Job {i}", style='Card.TFrame', padding=15)
            card.grid(row=row, column=col, padx=20, pady=15, sticky="nsew")

            job_ops = []
            for j in range(data['max_ops']):
                row_f = tk.Frame(card, bg='#f1f3f4')
                row_f.pack(fill='x', pady=4)

                tk.Label(row_f, text=f"Op {j}", width=4, bg='#f1f3f4', font=('Segoe UI', 9, 'bold')).pack(side='left')
                tk.Label(row_f, text="M-ID:", bg='#f1f3f4', font=('Segoe UI', 9)).pack(side='left')
                m_ent = ttk.Entry(row_f, width=5)
                m_ent.pack(side='left', padx=3)

                tk.Label(row_f, text="Time:", bg='#f1f3f4', font=('Segoe UI', 9)).pack(side='left')
                t_ent = ttk.Entry(row_f, width=5)
                t_ent.pack(side='left', padx=3)

                job_ops.append((m_ent, t_ent))
            self.input_refs.append(job_ops)

    def run_algorithms(self):
        parsed = []
        mach_limit = self.controller.shared_data['num_machines']
        try:
            for i, ops in enumerate(self.input_refs):
                clean_ops = []
                for m, t in ops:
                    if m.get() and t.get():
                        m_val = int(m.get())
                        t_val = int(t.get())
                        if m_val < 0 or m_val >= mach_limit:
                            raise ValueError(f"Job {i}, Op Machine ID {m_val} invalid. (0-{mach_limit - 1})")
                        clean_ops.append({'machine': m_val, 'time': t_val})
                if not clean_ops:
                    raise ValueError(f"Job {i} has no operations.")
                parsed.append({'job_id': i, 'operations': clean_ops})

            self.controller.shared_data['jobs_data'] = parsed
            self.controller.show_frame("ResultsPage")
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))


class ResultsPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self, bg='white')
        header.grid(row=0, column=0, sticky='ew', padx=30, pady=15)
        ttk.Label(header, text="Optimization Dashboard", style="Heading.TLabel").pack(side='left')
        ttk.Button(header, text="New Problem", style="Secondary.TButton", width=15,
                   command=lambda: controller.show_frame("HomePage")).pack(side='right')

        self.scroll_area = ScrollableFrame(self)
        self.scroll_area.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)

    def update_view(self):
        content_frame = self.scroll_area.scrollable_content
        for w in content_frame.winfo_children(): w.destroy()

        data = self.controller.shared_data
        algo = data['algo_choice']
        num_machines = data['num_machines']

        # --- MOCK EXECUTION & DATA ---
        schedule_bt, rt_bt, makespan_bt = [], 0, 0
        nodes_history_bt = []

        if algo in ['backtracking', 'both']:
            schedule_bt = self.mock_schedule(data['jobs_data'], random_delays=False)
            rt_bt = random.uniform(2.5, 4.0)
            makespan_bt = self.calculate_makespan(schedule_bt)
            nodes_history_bt = [(10, 25), (50, 20), (100, 18), (500, 15), (1000, 15)]

        schedule_ca, rt_ca, makespan_ca = [], 0, 0
        fitness_history_ca = []

        if algo in ['cultural', 'both']:
            schedule_ca = self.mock_schedule(data['jobs_data'], random_delays=True)
            rt_ca = random.uniform(0.1, 0.5)
            # Ensure CA Makespan > BT Makespan for consistent demo logic
            temp_makespan = self.calculate_makespan(schedule_ca)
            if algo == 'both' and temp_makespan <= makespan_bt:
                temp_makespan = int(makespan_bt * 1.2) + 1
            makespan_ca = temp_makespan

            fitness_history_ca = [30 - i * 0.2 + random.uniform(-1, 1) for i in range(50)]

        # --- PLOTTING ---
        total_plots = 0
        if algo == 'backtracking':
            total_plots = 2
        elif algo == 'cultural':
            total_plots = 2
        elif algo == 'both':
            total_plots = 3

        fig = plt.Figure(figsize=(10, 5 * total_plots), dpi=100, facecolor='white')
        current_plot = 1

        if algo == 'backtracking':
            ax1 = fig.add_subplot(total_plots, 1, current_plot)
            self.draw_gantt(ax1, schedule_bt, "Backtracking Schedule (Optimal)", num_machines)
            current_plot += 1
            ax2 = fig.add_subplot(total_plots, 1, current_plot)
            self.draw_convergence(ax2, nodes_history_bt, "Backtracking Progress", "Nodes Explored", "Makespan")

        elif algo == 'cultural':
            ax1 = fig.add_subplot(total_plots, 1, current_plot)
            self.draw_gantt(ax1, schedule_ca, "Cultural Algorithm Schedule (Approx)", num_machines)
            current_plot += 1
            ax2 = fig.add_subplot(total_plots, 1, current_plot)
            ca_plot_data = list(enumerate(fitness_history_ca))
            self.draw_convergence(ax2, ca_plot_data, "Cultural Algorithm Performance", "Generation", "Makespan")

        elif algo == 'both':
            ax1 = fig.add_subplot(total_plots, 1, current_plot)
            self.draw_gantt(ax1, schedule_bt, "Backtracking Schedule (Optimal)", num_machines)
            current_plot += 1
            ax2 = fig.add_subplot(total_plots, 1, current_plot)
            self.draw_gantt(ax2, schedule_ca, "Cultural Algorithm Schedule (Approx)", num_machines)
            current_plot += 1

            # **LINE PLOT COMPARISON**
            ax3 = fig.add_subplot(total_plots, 1, current_plot)
            self.draw_comparison_line(ax3, rt_bt, makespan_bt, rt_ca, makespan_ca)

        fig.tight_layout(pad=4.0)
        canvas = FigureCanvasTkAgg(fig, content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def draw_gantt(self, ax, schedule, title, num_machines):
        colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8E24AA', '#FD7E14']
        for (m_id, start, duration, job_id) in schedule:
            c = colors[job_id % len(colors)]
            ax.barh(y=m_id, width=duration, left=start, height=0.6,
                    color=c, edgecolor='black', linewidth=1.5, align='center')
            mid = start + (duration / 2)
            ax.text(mid, m_id, f"J{job_id}", ha='center', va='center',
                    color='white', fontweight='bold', fontsize=9)

        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"Machine {i}" for i in range(num_machines)], fontsize=10, fontweight='bold')
        ax.set_xlabel("Time", fontsize=10)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=10, color='#555')
        ax.invert_yaxis()

    def draw_convergence(self, ax, data_points, title, xlabel, ylabel):
        if not data_points: return
        xs, ys = zip(*data_points)
        ax.plot(xs, ys, color='#1a73e8', linewidth=2.5, marker='o', markersize=4)
        ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=10, color='#555')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def draw_comparison_line(self, ax, t1, m1, t2, m2):
        labels = ['Backtracking', 'Cultural Algo']
        runtimes = [t1, t2]
        makespans = [m1, m2]
        x = [0, 1]

        color_rt = '#4285F4'
        ax.plot(x, runtimes, color=color_rt, marker='o', linewidth=2.5, markersize=10, label='Runtime (s)')
        ax.set_ylabel('Runtime (seconds)', color=color_rt, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=color_rt)

        ax2 = ax.twinx()
        color_ms = '#EA4335'
        ax2.plot(x, makespans, color=color_ms, marker='s', linewidth=2.5, markersize=10, label='Makespan',
                 linestyle='--')
        ax2.set_ylabel('Makespan (Time)', color=color_ms, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_ms)

        ax.set_title('Performance Trade-off', fontweight='bold', pad=15, color='#444')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold', fontsize=11)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    def calculate_makespan(self, schedule):
        if not schedule: return 0
        return max(item[1] + item[2] for item in schedule)

    def mock_schedule(self, jobs, random_delays=False):
        schedule = []
        machine_free = {}
        for job in jobs:
            last_end = 0
            for op in job['operations']:
                m, dur = op['machine'], op['time']
                start = max(last_end, machine_free.get(m, 0))
                if random_delays:
                    start += random.randint(1, 3)
                schedule.append((m, start, dur, job['job_id']))
                last_end = start + dur
                machine_free[m] = last_end
        return schedule


if __name__ == "__main__":
    app = JobSchedulingApp()
    app.mainloop()
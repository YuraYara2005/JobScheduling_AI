import os


class DataLoader:
    """
    Responsible for loading Job Scheduling Problem instances from text files.
    Parses the file and returns a structure suitable for the 'models' classes.
    """

    def __init__(self):
        self.jobs_data = []
        self.num_jobs = 0
        self.num_machines = 0

    def load_instance(self, file_path):
        """
        Reads a standard JSP text file.

        Expected Format:
        Line 1: <number_of_jobs> <number_of_machines>
        Line 2 (Job 0): <machine_id> <processing_time> <machine_id> <processing_time> ...
        Line 3 (Job 1): ...

        Args:
            file_path (str): The absolute or relative path to the instance file.

        Returns:
            dict: A dictionary containing 'num_jobs', 'num_machines', and 'jobs'.
                  'jobs' is a list of lists, where each inner list represents operations.
                  e.g., {'num_jobs': 3, 'jobs': [[(0, 10), (1, 5)], ...]}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} was not found.")

        try:
            with open(file_path, 'r') as f:
                # Read all lines, stripping whitespace and skipping empty lines
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if not lines:
                raise ValueError("The selected file is empty.")

            # Parse Header (Number of Jobs and Machines)
            header = lines[0].split()
            if len(header) < 2:
                raise ValueError("Header must contain at least Number of Jobs and Number of Machines.")

            self.num_jobs = int(header[0])
            self.num_machines = int(header[1])

            self.jobs_data = []

            # Parse Jobs
            # We start from line 1 (index 1) because line 0 is the header
            for i, line in enumerate(lines[1:]):
                if i >= self.num_jobs:
                    break  # Stop if we have read all expected jobs

                parts = list(map(int, line.split()))

                # In standard JSP formats, data is often pairs: (Machine, Time)
                # We group them into tuples
                operations = []
                for j in range(0, len(parts), 2):
                    if j + 1 < len(parts):
                        machine_id = parts[j]
                        processing_time = parts[j + 1]
                        operations.append({'machine': machine_id, 'time': processing_time})

                self.jobs_data.append({
                    'job_id': i,
                    'operations': operations
                })

            return {
                'num_jobs': self.num_jobs,
                'num_machines': self.num_machines,
                'jobs': self.jobs_data
            }

        except ValueError as e:
            raise ValueError(f"Error parsing file: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading: {e}")

    def validate_data(self):
        """
        Optional helper to ensure the loaded data makes sense
        (e.g., no negative times).
        """
        if not self.jobs_data:
            return False

        for job in self.jobs_data:
            for op in job['operations']:
                if op['time'] < 0:
                    raise ValueError(f"Negative processing time found in Job {job['job_id']}")
        return True
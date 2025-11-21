import os
import signal
import subprocess
import concurrent.futures
import threading

from tqdm import tqdm


class UpdatableCommand:
    def __init__(self, command=None, config_dir='config/HybridCBM', **kwargs):
        if command is not None:
            self.config = command.config.copy()
            self.options = command.options.copy()
            self.config_dir = command.config_dir
        else:
            self.config = {}
            self.options = {}
            self.config_dir = config_dir
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in ['dataset', 'n_shots', 'script']:
                self.config[k] = v
            self.options[k] = v

    def config_to_str(self):
        if 'probe' in self.config['script']:
            return f"config/LProbe/{self.config['dataset']}.py"
        return f"{self.config_dir}/{self.config['dataset']}/{self.config['dataset']}_{self.config['n_shots']}shot.py"

    def options_to_str(self):
        if 'probe' in self.config['script']:
            self.options[
                'exp_root'] = f'exp/LProbe/{self.config["dataset"]}/{self.options["clip_model"].replace("/", "_")}'
        kwargs = ' '.join([f'--cfg-options {k}={v}' for k, v in self.options.items()])
        return kwargs

    def __call__(self):
        kwargs = self.options_to_str()
        config = self.config_to_str()
        return f"python {self.config['script']} --config {config} {kwargs}"


class MultiRunner:
    def __init__(self, commands, devices=None, max_workers=None, n_workers_per_device=None,
                 tail_command="", batch_run=False):
        self.commands = commands
        self.batch_run = batch_run
        if devices is not None:
            self.devices = devices
        else:
            self.devices = ['cpu']
        if max_workers is None and n_workers_per_device is None:
            raise ValueError("Either max_workers or n_workers_per_device must be provided")
        if n_workers_per_device is None:
            self.n_workers_per_device = max_workers // len(self.devices)
        else:
            self.n_workers_per_device = n_workers_per_device

        self.generator = self.generate_commands(tail_command)
        self.progress_bar = tqdm(total=len(self.commands), desc='Running Processes')
        print(f"Commands: {len(self.commands)}")

    def generate_commands(self, tail_command):
        while self.commands:
            commands = []
            for d in self.devices:
                for _ in range(self.n_workers_per_device if self.batch_run else 1):
                    if self.commands:
                        cmd = self.commands.pop(0) + f" --device {d}"
                        if tail_command:
                            cmd += f" {tail_command}"
                        commands.append(cmd)
            yield commands

    def run(self):
        if self.batch_run:
            for commands in self.generator:
                self._run(commands)
        else:
            commands = []
            for c in self.generator:
                commands.extend(c)
            self._run(commands, max_workers=self.n_workers_per_device * len(self.devices))
        self.progress_bar.close()

    def _run(self, commands, max_workers=None):
        # Create progress bar
        processes = []
        if max_workers is None:
            max_workers = len(commands)

        def run_command(command):
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            processes.append(process)
            print(f'Starting Process {process.pid}，Executing command：{command}')
            out, err = process.communicate()
            if process.returncode == 0:
                print(f'Process {process.pid} execute successfully')
            else:
                print(f'Process {process.pid} execute failed')
                print(f'Process {process.pid} Standard error output：')
                print(err.decode())
                print(f'Process {process.pid} Executed Failed command：{command}')
            self.progress_bar.update(1)

        # Signal handler for SIGINT (Ctrl+C)
        def signal_handler(sig, frame):
            print("\nReceived Ctrl+C, terminating all child processes...")
            for process in processes:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Send SIGTERM to the process group
                    print(f'Terminated Process {process.pid}')
                except ProcessLookupError:
                    # Process might have already terminated
                    continue
            self.progress_bar.close()
            exit(0)

        # Register the signal handler
        signal.signal(signal.SIGINT, signal_handler)
        # Use ThreadPoolExecutor to manage subprocesses.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_command, command) for command in commands}


class AdvancedMultiRunner:
    def __init__(self, commands, devices=None, max_workers=None, n_workers_per_device=None, tail_command=""):
        self.commands = commands
        self.tail_command = tail_command
        if devices is not None:
            self.devices = devices
        else:
            self.devices = ['cpu']
        if max_workers is None and n_workers_per_device is None:
            raise ValueError("Either max_workers or n_workers_per_device must be provided")
        if n_workers_per_device is None:
            self.n_workers_per_device = max_workers // len(self.devices)
        else:
            self.n_workers_per_device = n_workers_per_device

        self.commands_queue = self.commands.copy()
        self.tasks_per_device = {device: 0 for device in self.devices}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.progress_bar = tqdm(total=len(self.commands), desc='Running Processes')
        self.processes = []
        print(f"Using AdvancedMultiRunner [Commands: {len(self.commands)}]")

    def run(self):
        # Signal handler for SIGINT (Ctrl+C)
        def signal_handler(sig, frame):
            print("\nReceived Ctrl+C, terminating all child processes...")
            for process in self.processes:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Send SIGTERM to the process group
                    print(f'Terminated Process {process.pid}')
                except ProcessLookupError:
                    # Process might have already terminated
                    continue
            self.progress_bar.close()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        max_workers = self.n_workers_per_device * len(self.devices)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.worker) for _ in range(max_workers)]
            concurrent.futures.wait(futures)
        self.progress_bar.close()

    def worker(self):
        while True:
            with self.condition:
                while True:
                    if not self.commands_queue:
                        return  # No more commands
                    # Find devices with available capacity
                    available_devices = [
                        device for device in self.devices if self.tasks_per_device[device] < self.n_workers_per_device
                    ]
                    if available_devices:
                        break
                    else:
                        # Wait until a device becomes available
                        self.condition.wait()
                # Select the device with the least tasks running
                min_tasks = min(
                    self.tasks_per_device[device] for device in available_devices
                )
                devices_with_min_tasks = [
                    device for device in available_devices if self.tasks_per_device[device] == min_tasks
                ]
                selected_device = devices_with_min_tasks[0]  # Tie-breaker: pick first
                # Get the command
                command = self.commands_queue.pop(0)
                # Increment task count
                self.tasks_per_device[selected_device] += 1
                # Prepare command
                command += f' --device {selected_device}'
                if self.tail_command:
                    command += f' {self.tail_command}'
            # Run command outside the lock
            self.run_command(command)
            # After command finishes
            with self.condition:
                self.tasks_per_device[selected_device] -= 1
                # Notify other waiting threads
                self.condition.notify_all()
                self.progress_bar.update(1)

    def run_command(self, command):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        self.processes.append(process)
        print(f'Starting Process {process.pid}, Executing command: {command}')
        out, err = process.communicate()
        if process.returncode == 0:
            print(f'Process {process.pid} executed successfully')
        else:
            print(f'Process {process.pid} execution failed')
            print(f'Process {process.pid} Standard error output:')
            print(err.decode())
            print(f'Process {process.pid} Failed command: {command}')
        # Progress bar update is handled in the worker function

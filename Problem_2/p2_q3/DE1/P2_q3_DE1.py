import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
import time
import datetime
import os
import json
import signal
import pickle
import sys
import shutil
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


class TimeoutManager:
    """
    Manages execution time limits and provides graceful termination
    """
    def __init__(self, max_runtime_hours=48):  # Default 2 days (48 hours)
        self.max_runtime_seconds = max_runtime_hours * 3600
        self.start_time = time.time()
        self.checkpoint_interval = 1800  # Save checkpoint every 30 minutes
        self.last_checkpoint_time = self.start_time
        
        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Set a termination flag
        self.terminate_requested = False
        
        print(f"TimeoutManager initialized. Max runtime: {max_runtime_hours} hours")
        print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def check_timeout(self):
        """Check if the maximum runtime has been exceeded"""
        elapsed = time.time() - self.start_time
        return elapsed > self.max_runtime_seconds
    
    def should_checkpoint(self):
        """Check if it's time to create a checkpoint"""
        current_time = time.time()
        if current_time - self.last_checkpoint_time > self.checkpoint_interval:
            self.last_checkpoint_time = current_time
            print(f"\n=== CHECKPOINT TRIGGERED at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"=== Elapsed time: {self.elapsed_time_formatted()}, Interval: {self.checkpoint_interval/60:.1f} minutes ===\n")
            return True
        return False
    
    def time_remaining(self):
        """Get the remaining runtime in seconds"""
        elapsed = time.time() - self.start_time
        return max(0, self.max_runtime_seconds - elapsed)
    
    def elapsed_time(self):
        """Get the elapsed time in seconds"""
        return time.time() - self.start_time
    
    def elapsed_time_formatted(self):
        """Get the elapsed time as a formatted string"""
        elapsed = self.elapsed_time()
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def signal_handler(self, sig, frame):
        """Handle termination signals"""
        print(f"\nReceived termination signal ({sig}). Preparing for graceful shutdown...")
        self.terminate_requested = True
    
    def should_terminate(self):
        """Check if termination has been requested or timeout reached"""
        return self.terminate_requested or self.check_timeout()


class EpsilonConstrainedOptimizer:
    """
    Implements epsilon-constrained design optimization for the NASA/DNV UQ Challenge.
    
    This solves Question 3: Find the control variable X_c that maximizes the performance
    metric under the constraint that the system failure probability is below a threshold ε.
    
    Enhanced with timeout management and checkpointing capabilities.
    """
    def __init__(self, exe_path, data_path, n_epistemic_samples=35, 
                 checkpoint_dir="epsilon_checkpoints", max_runtime_hours=48):
        self.exe_path = exe_path
        self.data_path = Path(data_path)
        self.n_epistemic_samples = n_epistemic_samples
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize timeout manager
        self.timeout_mgr = TimeoutManager(max_runtime_hours)
        
        # Indices for performance and constraint outputs
        self.I1 = [0, 1, 2]  # Performance indices (y1, y2, y3)
        self.I2 = [3, 4, 5]  # Constraint indices (y4, y5, y6)
        
        # Load data
        self.load_data()
        
        # Control variable bounds (0 to 1 for all three)
        self.bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Constants c_i from problem description
        self.c_values = [2750, 2000, 1000]  # c1, c2, c3 as provided in the problem statement
        
        # Store optimization history
        self.iteration_history = []
        self.objective_history = []
        self.constraint_history = []
        
        # Setup function evaluation counting and caching
        self.func_eval_count = 0
        self.eval_cache = {}  # Standard cache
        self.high_fidelity_cache = {}  # Cache for high-fidelity evaluations
        
        # Best solution found so far
        self.best_solution = {
            "X_c": np.array([0.5, 0.5, 0.5]),  # Default initial guess
            "objective": float('-inf'),
            "constraint_value": 1.0,
            "feasible": False,
            "iteration": 0
        }
        
        # Initialize criticality tracking for epistemic samples
        self.epistemic_criticality_obj = np.zeros(n_epistemic_samples)  # For objective function
        self.epistemic_criticality_pof = np.zeros(n_epistemic_samples)  # For constraints
        
        # Restore from checkpoint if available
        self.restore_checkpoint()
        
        print(f"EpsilonConstrainedOptimizer initialized with all aleatory samples and {n_epistemic_samples} epistemic samples")
        
    def load_data(self):
        """
        Load data for uncertainty models from files.
        - Load all aleatory samples (Xa1, Xa2) from .npy files
        - Generate epistemic samples (Xe1, Xe2, Xe3) using uniform distributions with specified bounds
        """
        # Load aleatory samples from .npy files
        
        xa1 = np.load(self.data_path / "Xa1_stratified_samples_100.npy")
        xa2 = np.load(self.data_path / "Xa2_stratified_samples_100.npy")
        
        # Reshape if necessary and combine into a single array
        xa1 = xa1.reshape(-1, 1)  # Ensure column vector
        xa2 = xa2.reshape(-1, 1)  # Ensure column vector
        
        # Store all available samples
        self.Xa_samples = np.hstack((xa1, xa2))
        self.total_aleatory_samples = len(self.Xa_samples)
        
        print(f"Loaded {self.total_aleatory_samples} aleatory samples from .npy files")
        print(f"Using all available aleatory samples")
            
        # Initialize array for epistemic samples
        self.E_samples = np.zeros((self.n_epistemic_samples, 3))
        
        # Define bounds for epistemic variables
        xe1_min, xe1_max = 0.322, 0.323  # Example bounds for Xe1
        xe2_min, xe2_max = 0.588, 0.592  # Example bounds for Xe2
        xe3_min, xe3_max = 0.5258, 0.5659  # Example bounds for Xe3
        
        # Generate uniform samples for all epistemic variables
        self.E_samples[:, 0] = xe1_min + (xe1_max - xe1_min) * np.random.random(self.n_epistemic_samples)
        self.E_samples[:, 1] = xe2_min + (xe2_max - xe2_min) * np.random.random(self.n_epistemic_samples)
        self.E_samples[:, 2] = xe3_min + (xe3_max - xe3_min) * np.random.random(self.n_epistemic_samples)
        
        print(f"Generated Xe1 samples using range [{xe1_min:.4f}, {xe1_max:.4f}]")
        print(f"Generated Xe2 samples using range [{xe2_min:.4f}, {xe2_max:.4f}]")
        print(f"Generated Xe3 samples using range [{xe3_min:.4f}, {xe3_max:.4f}]")
        print(f"Using {self.n_epistemic_samples} epistemic samples")
        

    
    def run_forward_model(self, input_data):
        """
        Run the forward model simulation for given input data
        Args:
            input_data: Array of input samples [Xa1, Xa2, Xe1, Xe2, Xe3, Xc1, Xc2, Xc3, seed]
        """
        # Save input data to temporary file
        np.savetxt("temp_input.txt", input_data, delimiter=',')
        
        # Run simulation
        try:
            command = [self.exe_path, "temp_input.txt"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)  # 5-minute timeout
            
            if result.returncode != 0:
                print(f"Simulation warning: Return code {result.returncode}")
                print(f"Error message: {result.stderr}")
                return None
            
            output_file = Path("Y_out.csv")
            if not output_file.exists():
                print("No output file generated")
                return None
                
            df = pd.read_csv(output_file, header=None)
            sample_indices = df[6].unique()
            num_samples = len(sample_indices)
            df = df.drop(columns=[6])
            
            Y_out = df.to_numpy().reshape(num_samples, 60, 6)
            
            return Y_out
            
        except subprocess.TimeoutExpired:
            print("Simulation timed out after 5 minutes")
            return None
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
    
    def combined_evaluation(self, Xa, Xe, Xc, seeds=None):
        """
        Combined function to evaluate both the objective and constraints in a single call.
        This avoids running the forward model twice for the same inputs.
        
        Args:
            Xa: Aleatory samples (n_samples x 2)
            Xe: Epistemic parameters (3,)
            Xc: Control parameters (3,)
            seeds: Random seeds for simulation
            
        Returns:
            objective_value: Expected value of the performance metric
            pof_i: Individual failure probabilities for each constraint
            pof_sys: System failure probability
        """
        n_samples = Xa.shape[0]
        
        # Generate seeds if not provided
        if seeds is None:
            seeds = np.random.randint(0, 10000, n_samples)
        
        # Prepare input data for forward model
        input_data = np.zeros((n_samples, 9))
        for i in range(n_samples):
            input_data[i, 0:2] = Xa[i]  # Aleatory variables
            input_data[i, 2:5] = Xe     # Epistemic variables
            input_data[i, 5:8] = Xc     # Control variables
            input_data[i, 8] = seeds[i] # Random seed
        
        # Run simulation
        Y_out = self.run_forward_model(input_data)
        if Y_out is None:
            return np.nan, [np.nan] * len(self.I2), np.nan
        
        # ---- Calculate objective function (performance metric) ----
        J_values = np.zeros(n_samples)
        for i in range(n_samples):
            J_sample = 0
            for idx in self.I1:
                # Sum over time steps
                J_sample += np.sum(Y_out[i, :, idx])*(1/60)
            J_values[i] = J_sample
        
        # Expected value (average over all samples)
        objective_value = np.mean(J_values)
        
        # ---- Calculate constraint functions (reliability) ----
        pof_i = []
        g_values = []
        
        for idx, i in enumerate(self.I2):
            g_samples = np.zeros(n_samples)
            for j in range(n_samples):
                # Get the maximum absolute value over time for this output
                max_abs_value = np.max(np.abs(Y_out[j, :, i]))
                # Calculate g_i(X, s) = c_i - max_t |y_i(X, s, t)|
                g_samples[j] = self.c_values[idx] - max_abs_value
                
            # Failure occurs when g_i < 0
            failure_indicator = g_samples < 0
            # Individual failure probability
            pof = np.mean(failure_indicator)
            pof_i.append(pof)
            g_values.append(g_samples)
        
        # System failure occurs when any constraint is violated (min_i g_i < 0)
        # For each sample j, find the minimum g value across all constraints
        min_g_values = np.min(np.array(g_values), axis=0)
        # System failure probability
        pof_sys = np.mean(min_g_values < 0)
        
        return objective_value, pof_i, pof_sys
    
    def save_checkpoint(self, final=False):
        """
        Save current optimization state to checkpoint file
        """
        # Create a versioned checkpoint filename based on timestamp
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_basename = f"checkpoint_{timestamp_str}"
        
        # Prepare checkpoint data
        checkpoint = {
            "timestamp": time.time(),
            "elapsed_time": self.timeout_mgr.elapsed_time(),
            "iteration_history": self.iteration_history,
            "objective_history": self.objective_history,
            "constraint_history": self.constraint_history,
            "eval_cache": self.eval_cache,
            "high_fidelity_cache": self.high_fidelity_cache,
            "best_solution": self.best_solution,
            "epistemic_criticality_obj": self.epistemic_criticality_obj,
            "epistemic_criticality_pof": self.epistemic_criticality_pof,
            "func_eval_count": self.func_eval_count,
            "final": final
        }
        
        # Save to multiple files for redundancy
        # 1. Latest checkpoint (always overwritten)
        latest_checkpoint_file = self.checkpoint_dir / "latest_checkpoint.pkl"
        with open(latest_checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        # 2. Versioned checkpoint (never overwritten)
        versioned_checkpoint_file = self.checkpoint_dir / f"{checkpoint_basename}.pkl"
        with open(versioned_checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save results in a more accessible format
        results = {
            "X_c_optimal": self.best_solution["X_c"].tolist(),
            "best_objective": float(self.best_solution["objective"]),
            "constraint_value": float(self.best_solution["constraint_value"]),
            "feasible": bool(self.best_solution["feasible"]),
            "iteration": int(self.best_solution["iteration"]),
            "func_evaluations": int(self.func_eval_count),
            "elapsed_time": self.timeout_mgr.elapsed_time_formatted(),
            "completed": bool(final),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save latest results as JSON
        with open(self.checkpoint_dir / "latest_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        # Also save versioned results JSON
        with open(self.checkpoint_dir / f"{checkpoint_basename}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save plots if we have data
        if len(self.iteration_history) > 0:
            self.plot_optimization_progress(save_path=self.checkpoint_dir / f"{checkpoint_basename}.png")
            self.plot_optimization_progress(save_path=self.checkpoint_dir / "epsilon_optimization_progress.png")
        
        print(f"\n=== CHECKPOINT SAVED ===")
        print(f"Saved to: {versioned_checkpoint_file}")
        print(f"Iteration: {self.best_solution['iteration']}")
        print(f"Best objective: {self.best_solution['objective']:.6f}")
        print(f"Constraint value: {self.best_solution['constraint_value']:.6f}")
        print(f"Function evaluations: {self.func_eval_count}")
        print(f"======================\n")
    
    def restore_checkpoint(self):
        """
        Restore from latest checkpoint if available
        """
        checkpoint_file = self.checkpoint_dir / "latest_checkpoint.pkl"
        if not checkpoint_file.exists():
            print("No checkpoint found. Starting fresh optimization.")
            return False
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.iteration_history = checkpoint["iteration_history"]
            self.objective_history = checkpoint["objective_history"]
            self.constraint_history = checkpoint["constraint_history"]
            self.eval_cache = checkpoint["eval_cache"]
            
            # Restore high-fidelity cache if available
            if "high_fidelity_cache" in checkpoint:
                self.high_fidelity_cache = checkpoint["high_fidelity_cache"]
            
            # Restore function evaluation count if available
            if "func_eval_count" in checkpoint:
                self.func_eval_count = checkpoint["func_eval_count"]
            
            # Restore epistemic criticality tracking if available
            if "epistemic_criticality_obj" in checkpoint:
                self.epistemic_criticality_obj = checkpoint["epistemic_criticality_obj"]
            if "epistemic_criticality_pof" in checkpoint:
                self.epistemic_criticality_pof = checkpoint["epistemic_criticality_pof"]
            
            self.best_solution = checkpoint["best_solution"]
            
            print(f"Restored checkpoint from iteration {self.best_solution['iteration']}")
            print(f"Best objective value so far: {self.best_solution['objective']:.6f}")
            print(f"Best constraint value so far: {self.best_solution['constraint_value']:.6f}")
            print(f"Solution is {'feasible' if self.best_solution['feasible'] else 'infeasible'}")
            print(f"Elapsed time before: {datetime.timedelta(seconds=checkpoint['elapsed_time'])}")
            
            return True
            
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            print("Starting fresh optimization.")
            return False
    
    def plot_optimization_progress(self, save_path=None):
        """
        Plot the optimization progress over iterations
        """
        if len(self.iteration_history) == 0:
            print("No history to plot yet.")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Performance metric plot
        ax1.plot(self.iteration_history, self.objective_history, 'b-o', label='Objective')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('ε-Constrained Optimization Progress')
        ax1.grid(True)
        ax1.legend()
        
        # Constraint plot
        ax2.plot(self.iteration_history, self.constraint_history, 'r-s', label='System Failure Probability')
        ax2.axhline(y=0.001, color='k', linestyle='--', label='ε Threshold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Probability of Failure')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Progress plot saved to {save_path}")
        else:
            plt.show()
    
    def epsilon_constrained_design(self, epsilon=0.05, hyperparam_set=None, max_iter=20, verbose=True):
        """
        Solve the ε-constrained design optimization problem:
        Find X_c that maximizes min(X_e∈E) J(X_e, X_c) subject to max(X_e∈E) pof_sys(X_e, X_c) ≤ ε
        
        Args:
            epsilon: Maximum allowed system failure probability
            hyperparam_set: Hyperparameters for differential evolution
            max_iter: Maximum number of optimization iterations
            verbose: Whether to print progress
            
        Returns:
            X_c_optimal: Optimal control parameters
            best_obj: Best objective value
            worst_pof: Worst-case system failure probability at the optimum
        """
        if verbose:
            print("Solving ε-Constrained Design Optimization Problem...")
            print(f"Failure probability threshold ε: {epsilon}")
            print(f"Max iterations: {max_iter}")
            print(f"Time limit: {self.timeout_mgr.max_runtime_seconds/3600:.1f} hours")
        
        # Setup iteration counter - IMPORTANT: Make this a simple counter
        # Starting value should be the highest iteration from previous runs
        iteration_counter = max(0, self.best_solution["iteration"])
        
        # Penalty factor for constraint violations
        penalty_factor = 1e6  # Large penalty for constraint violations
        
        # Define the augmented objective function (includes penalty for constraint violations)
        def augmented_objective(X_c):
            # Increment function evaluation counter
            self.func_eval_count += 1
            
            # Stop if time limit reached
            if self.timeout_mgr.should_terminate():
                return np.nan  # Return a neutral value
                
            X_c_tuple = tuple(X_c.tolist())
            
            # Check high-fidelity cache first for accurate results
            if X_c_tuple in self.high_fidelity_cache:
                cached_results = self.high_fidelity_cache[X_c_tuple]
                obj_value = cached_results["objective"]
                worst_pof = cached_results["worst_pof"]
                
                # Use cached results and apply penalty if needed
                if worst_pof > epsilon:
                    return -obj_value - penalty_factor * (worst_pof - epsilon)
                else:
                    return -obj_value  # Negate for maximization
            
            # Then check standard cache
            if X_c_tuple in self.eval_cache:
                cached_results = self.eval_cache[X_c_tuple]
                obj_value = cached_results["objective"]
                worst_pof = cached_results["worst_pof"]
                
                # Use cached results and apply penalty if needed
                if worst_pof > epsilon:
                    return -obj_value - penalty_factor * (worst_pof - epsilon)
                else:
                    return -obj_value  # Negate for maximization
            
            # Using all available samples for every evaluation
            Xa_subset = self.Xa_samples
            E_subset = self.E_samples
            
            # Log evaluation count periodically
            if self.func_eval_count % 10 == 0:
                print(f"Function eval #{self.func_eval_count}: Using all {len(self.Xa_samples)} aleatory and {len(self.E_samples)} epistemic samples")
            
            # Debug output for evaluation
            if verbose and (self.func_eval_count <= 1 or self.func_eval_count % 20 == 0):
                print(f"Evaluating X_c = [{X_c[0]:.4f}, {X_c[1]:.4f}, {X_c[2]:.4f}]")
            
            # For each X_c, find the worst-case performance and reliability
            worst_case_J = float('inf')
            worst_case_pof = -float('inf')
            worst_X_e_obj = None  # Track worst epistemic for objective
            worst_X_e_pof = None  # Track worst epistemic for constraint
            
            # Track criticality of each epistemic sample
            criticality_obj = np.zeros(len(E_subset))
            criticality_pof = np.zeros(len(E_subset))
            
            # Evaluate both metrics for each epistemic sample with combined evaluation
            for i, X_e in enumerate(E_subset):
                # Combined evaluation (one forward model call)
                J, _, pof_sys = self.combined_evaluation(Xa_subset, X_e, X_c)
                
                # Track criticality 
                criticality_obj[i] = J if not np.isnan(J) else float('inf')
                criticality_pof[i] = pof_sys if not np.isnan(pof_sys) else 0
                
                # Update worst-case objective
                if J < worst_case_J and not np.isnan(J):
                    worst_case_J = J
                    worst_X_e_obj = X_e
                
                # Update worst-case constraint
                if pof_sys > worst_case_pof and not np.isnan(pof_sys):
                    worst_case_pof = pof_sys
                    worst_X_e_pof = X_e
            
            # Handle NaN values
            if np.isnan(worst_case_J) or np.isinf(worst_case_J):
                worst_case_J = float('-inf')  # Very bad performance
            
            if np.isnan(worst_case_pof):
                worst_case_pof = 1.0  # Assume worst-case failure
            
            # Update criticality tracking
            self.epistemic_criticality_obj = criticality_obj
            self.epistemic_criticality_pof = criticality_pof
            
            # All evaluations are high-fidelity since we're using all samples
            is_high_fidelity = True
            
            # Store in appropriate cache
            eval_result = {
                "objective": worst_case_J,
                "worst_pof": worst_case_pof,
                "worst_X_e_obj": worst_X_e_obj,
                "worst_X_e_pof": worst_X_e_pof,
                "fidelity": "high" if is_high_fidelity else "low"
            }
            
            # Always update regular cache
            self.eval_cache[X_c_tuple] = eval_result
            
            # Update high-fidelity cache if appropriate
            if is_high_fidelity:
                self.high_fidelity_cache[X_c_tuple] = eval_result
                if verbose and self.func_eval_count % 10 == 0:
                    print(f"X_c: {X_c}, HIGH-FIDELITY evaluation - obj: {worst_case_J:.4f}, pof: {worst_case_pof:.6f}")
            elif verbose and self.func_eval_count % 20 == 0:
                print(f"X_c: {X_c}, obj: {worst_case_J:.4f}, pof: {worst_case_pof:.6f}")
            
            # Check if this solution is better than the current best
            is_solution_feasible = worst_case_pof <= epsilon
            should_update_best = False

            if is_solution_feasible:
                if not self.best_solution["feasible"]:
                    # Any feasible solution is better than any infeasible solution
                    should_update_best = True
                elif worst_case_J > self.best_solution["objective"]:
                    # Better objective among feasible solutions
                    should_update_best = True
            else:
                if not self.best_solution["feasible"] and worst_case_pof < self.best_solution["constraint_value"]:
                    # Better constraint value among infeasible solutions
                    should_update_best = True

            if should_update_best:
                # Update best solution
                self.best_solution["X_c"] = X_c.copy()
                self.best_solution["objective"] = worst_case_J
                self.best_solution["constraint_value"] = worst_case_pof
                self.best_solution["feasible"] = is_solution_feasible
                
                # Don't update iteration counter as it's managed by the callback
                if verbose and self.func_eval_count % 10 == 0:
                    status = "FEASIBLE" if is_solution_feasible else "INFEASIBLE"
                    print(f"New best solution from func eval! Obj: {worst_case_J:.4f}, PoF: {worst_case_pof:.6f} [{status}]")
            
            # Check for checkpoint
            if self.timeout_mgr.should_checkpoint():
                # Save the current iteration info before checkpointing
                # This ensures we capture the latest solution status
                if X_c_tuple not in self.eval_cache and len(self.iteration_history) > 0:
                    # Add this to history if not already there
                    self.iteration_history.append(iteration_counter)
                    self.objective_history.append(worst_case_J)
                    self.constraint_history.append(worst_case_pof)
                    
                self.save_checkpoint()
            
            # Calculate objective with penalty for constraint violation
            if worst_case_pof > epsilon:
                return -worst_case_J - penalty_factor * (worst_case_pof - epsilon)
            else:
                return -worst_case_J  # Negate for maximization

        # Callback function to track optimization progress
        def callback(xk, convergence=None):
            nonlocal iteration_counter  # Access the iteration counter as a non-local variable
            
            # Stop if time limit reached
            if self.timeout_mgr.should_terminate():
                print("\nTime limit reached or termination requested. Stopping optimization.")
                return True  # Return True to stop optimization
            
            # Increment iteration counter
            iteration_counter += 1
            
            # Current X_c and metrics
            current_X_c = xk
            X_c_tuple = tuple(current_X_c.tolist())
            
            # Get cached values or compute if not available
            if X_c_tuple in self.high_fidelity_cache:
                cached_results = self.high_fidelity_cache[X_c_tuple]
                current_obj = cached_results["objective"]
                current_pof = cached_results["worst_pof"]
                source = "high-fidelity cache"
            elif X_c_tuple in self.eval_cache:
                cached_results = self.eval_cache[X_c_tuple]
                current_obj = cached_results["objective"]
                current_pof = cached_results["worst_pof"]
                source = "standard cache"
            else:
                # This should rarely happen as the objective function should have cached these
                # Re-evaluate with higher fidelity for iteration tracking
                print(f"Re-evaluating solution in callback (rare case)...")
                
                # Using all samples for verification in callback
                Xa_subset = self.Xa_samples
                E_subset = self.E_samples
                
                # Evaluate
                worst_case_J = float('inf')
                worst_case_pof = -float('inf')
                
                for X_e in E_subset:
                    # Combined evaluation (one forward model call)
                    J, _, pof_sys = self.combined_evaluation(Xa_subset, X_e, current_X_c)
                    
                    if J < worst_case_J and not np.isnan(J):
                        worst_case_J = J
                    
                    if pof_sys > worst_case_pof and not np.isnan(pof_sys):
                        worst_case_pof = pof_sys
                
                # Handle NaN values
                if np.isnan(worst_case_J) or np.isinf(worst_case_J):
                    worst_case_J = float('-inf')
                
                if np.isnan(worst_case_pof):
                    worst_case_pof = 1.0
                
                current_obj = worst_case_J
                current_pof = worst_case_pof
                source = "new evaluation"
                
                # Cache these results
                self.eval_cache[X_c_tuple] = {
                    "objective": current_obj,
                    "worst_pof": current_pof
                }
            
            # Check if current solution is feasible
            is_feasible = current_pof <= epsilon
            
            # Always store progress at each callback
            self.iteration_history.append(iteration_counter)
            self.objective_history.append(current_obj)
            self.constraint_history.append(current_pof)
            
            # Update best solution if better
            should_update = False
            
            if is_feasible:
                if not self.best_solution["feasible"]:
                    # Any feasible solution is better than any infeasible solution
                    should_update = True
                elif current_obj > self.best_solution["objective"]:
                    # Better objective among feasible solutions
                    should_update = True
            else:
                if not self.best_solution["feasible"] and current_pof < self.best_solution["constraint_value"]:
                    # Better constraint value among infeasible solutions
                    should_update = True
            
            # Debug output to understand the decision-making
            if verbose:
                current_status = "FEASIBLE" if is_feasible else "INFEASIBLE"
                best_status = "FEASIBLE" if self.best_solution["feasible"] else "INFEASIBLE"
                update_status = "UPDATING" if should_update else "KEEPING CURRENT BEST"
                
                print(f"Comparing solutions: Current [{current_status}] (obj={current_obj:.4f}, pof={current_pof:.6e}) vs " + 
                    f"Best [{best_status}] (obj={self.best_solution['objective']:.4f}, pof={self.best_solution['constraint_value']:.6f}) - {update_status}")
            
            if should_update:
                self.best_solution["X_c"] = current_X_c.copy()
                self.best_solution["objective"] = current_obj
                self.best_solution["constraint_value"] = current_pof
                self.best_solution["feasible"] = is_feasible
                self.best_solution["iteration"] = iteration_counter
                
                if verbose:
                    status = "FEASIBLE" if is_feasible else "INFEASIBLE"
                    print(f"New best solution found! Objective: {current_obj:.4f}, PoF: {current_pof:.6e} [{status}]")
            
            # Always print progress for each iteration
            if verbose:
                elapsed = self.timeout_mgr.elapsed_time_formatted()
                status = "FEASIBLE" if is_feasible else "INFEASIBLE"
                print(f"Iteration {iteration_counter}: Obj = {current_obj:.4f}, PoF = {current_pof:.6e} " +
                    f"[{status}] (Elapsed: {elapsed}, Source: {source})")
            
            # Force checkpoint every 5 iterations for additional safety
            force_checkpoint = iteration_counter % 5 == 0 and iteration_counter > 0
            
            # Check if we should create a checkpoint
            if self.timeout_mgr.should_checkpoint() or force_checkpoint:
                if force_checkpoint:
                    print(f"Creating scheduled checkpoint at iteration {iteration_counter}")
                self.save_checkpoint()
            
            # Continue unless timeout
            return False

        # Initial solution setup
        # Try to use best solution from previous runs if available
        if np.any(self.best_solution["X_c"] != np.array([0.5, 0.5, 0.5])):
            x0 = self.best_solution["X_c"]
            print(f"Using previous best solution as initial guide: {x0}")
        else:
            x0 = np.array([0.533, 0.666, 0.5])  # Example values
            print(f"Using default initial position: {x0}")
        
        try:
            # Use provided hyperparameters or defaults
            if hyperparam_set is None:
                # Default settings
                de_params = {
                    'strategy': 'best1bin',
                    'mutation': 0.7,
                    'recombination': 0.9,
                    'popsize': 15,
                    'tol': 0.01,
                    'polish': True,
                    'maxiter': max_iter
                }
            else:
                # Use provided hyperparameter set
                de_params = {k: v for k, v in hyperparam_set.items() if k != 'name'}
                
                # Ensure maxiter is set if not in params
                if 'maxiter' not in de_params:
                    de_params['maxiter'] = max_iter
            
            if verbose:
                print(f"Using optimization parameters: {de_params}")
            
            # Calculate time available for optimization
            time_remaining = self.timeout_mgr.time_remaining()
            
            # Adjust maxiter based on time constraint if needed
            avg_time_per_iter = 300  # Assume 5 minutes per iteration as starting point
            if len(self.iteration_history) > 1:
                # Calculate from previous iterations if available
                elapsed = self.timeout_mgr.elapsed_time()
                avg_time_per_iter = elapsed / len(self.iteration_history)
            
            # Estimate how many iterations we can do in remaining time
            estimated_max_iter = int(time_remaining / avg_time_per_iter) - 1  # Leave 1 iteration buffer
            
            if estimated_max_iter < de_params['maxiter']:
                print(f"Time constraint: Reducing maxiter from {de_params['maxiter']} to {max(1, estimated_max_iter)}")
                de_params['maxiter'] = max(1, estimated_max_iter)
            
            # Run the optimization with proper iteration tracking
            result = differential_evolution(
                augmented_objective, 
                self.bounds, 
                callback=callback,
                x0=x0,  # Use initial guess
                disp=True,  # Enable display of progress
                **de_params
            )
            
            X_c_optimal = result.x
            
            terminated_early = self.timeout_mgr.should_terminate()
            
            # Print optimization result summary
            print("\n" + "="*50)
            print("OPTIMIZATION RESULT SUMMARY")
            print("="*50)
            print(f"Final solution from differential evolution: {X_c_optimal}")
            print(f"Number of iterations completed: {iteration_counter}")
            print(f"Total function evaluations: {self.func_eval_count}")
            print(f"Optimization {'completed successfully' if not terminated_early else 'terminated early'}")
            
            # Always print progress history
            if len(self.iteration_history) > 0:
                print("\nOptimization Progress History:")
                print(f"{'Iteration':<10}{'Objective':<15}{'Failure Prob':<15}{'Status':<10}")
                print("-"*50)
                for i, (iter_num, obj, pof) in enumerate(zip(self.iteration_history, self.objective_history, self.constraint_history)):
                    status = "FEASIBLE" if pof <= epsilon else "INFEASIBLE"
                    print(f"{iter_num:<10}{obj:<15.4f}{pof:<15.6e}{status:<10}")
            
            # No solution verification step
            # We're using all samples during the entire optimization
            
            # Save final checkpoint
            self.save_checkpoint(final=not terminated_early)
            
            # Plot the optimization progress
            self.plot_optimization_progress()
            
            # Return the best solution found
            return self.best_solution["X_c"], self.best_solution["objective"], self.best_solution["constraint_value"]
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Save what we have so far
            self.save_checkpoint(final=False)
            
            # Still return the best solution found, even if optimization failed
            return self.best_solution["X_c"], self.best_solution["objective"], self.best_solution["constraint_value"]



def main():
    # Set up signal handlers for system-wide signals
    def signal_handler(sig, frame):
        print(f"\n[SIGNAL RECEIVED] {sig}. Attempting to save final checkpoint before exiting...")
        if 'optimizer' in locals():
            try:
                optimizer.save_checkpoint(final=False)
                print("[CHECKPOINT SAVED] Emergency checkpoint saved successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to save emergency checkpoint: {e}")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Path to executable and data
    exe_path = os.path.abspath(Path("local_model_linux"))
    data_path = Path("/home/droysar1/data_sgoswam4/Dibakar/NASA_DNV/UM_model_pred/Samples_Xa/")
    
    # Set constraint threshold epsilon
    epsilon = 0.001  # 0.1% maximum allowed failure probability
    
    # Create a results directory
    results_dir = Path("epsilon_results")
    results_dir.mkdir(exist_ok=True)
    
    # Define maximum runtime in hours
    max_runtime_hours = 72  # 72 hours = 3 days
    
    print("NASA/DNV UQ Challenge - ε-Constrained Design Optimization")
    print("=" * 80)
    print(f"Maximum runtime: {max_runtime_hours} hours")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set epistemic sample size
    n_epistemic = 10  # Number of epistemic samples
    
    print(f"Using all available aleatory samples and {n_epistemic} epistemic samples")
    print(f"Constraint threshold ε: {epsilon}")
    
    # Define hyperparameter set
    # Modified parameters to ensure more iterations
    hyperparam_set = {
        'strategy': 'rand1bin',  # Random base vector selection for better exploration
        'mutation': 0.8,         # Slightly increased mutation rate for better exploration
        'recombination': 0.7,    # Reduced recombination to encourage more diverse solutions
        'popsize': 15,           # Increased population size
        'tol': 0.001,            # Reduced tolerance to avoid early convergence
        'polish': True,          # Keep polishing enabled
        'maxiter': 25            # Increased max iterations
    }
    
    print(f"\nUsing hyperparameters:")
    for key, value in hyperparam_set.items():
        print(f"  {key}: {value}")
    
    # Create checkpoint directory
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create the optimizer with checkpointing enabled
    optimizer = EpsilonConstrainedOptimizer(
        exe_path=exe_path, 
        data_path=data_path, 
        n_epistemic_samples=n_epistemic,
        checkpoint_dir=checkpoint_dir,
        max_runtime_hours=max_runtime_hours
    )
    
    # Run the optimization
    print("\nStarting ε-constrained design optimization...")
    X_c_optimal, best_obj, worst_pof = optimizer.epsilon_constrained_design(
        epsilon=epsilon,
        hyperparam_set=hyperparam_set,
        verbose=True
    )
    
    # Check if optimization completed or was interrupted
    completed = not optimizer.timeout_mgr.should_terminate()
    completion_status = "COMPLETED" if completed else "INTERRUPTED"
    
    # Save results
    if X_c_optimal is not None:
        result_info = {
            "X_c_optimal": X_c_optimal.tolist(),
            "best_objective": float(best_obj),
            "worst_case_pof": float(worst_pof),
            "epsilon": epsilon,
            "feasible": float(worst_pof) <= epsilon,
            "elapsed_time": optimizer.timeout_mgr.elapsed_time_formatted(),
            "completed": completed,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to files
        np.save(results_dir / "epsilon_constrained_results.npy", result_info)
        
        # Also save in JSON format for better readability
        with open(results_dir / "epsilon_constrained_results.json", "w") as f:
            json.dump(result_info, f, indent=4)
        
        print(f"\nResults saved to {results_dir / 'epsilon_constrained_results.json'}")
        
        # Report final status
        feasibility = "FEASIBLE" if result_info["feasible"] else "INFEASIBLE"
        print(f"\nOptimization {completion_status}")
        print(f"Final solution is {feasibility}")
        print(f"X_c = {X_c_optimal}")
        print(f"Objective value = {best_obj:.4f}")
        print(f"System failure probability = {worst_pof:.6e} (threshold: {epsilon})")
    
    total_time = time.time() - optimizer.timeout_mgr.start_time
    print(f"\nTotal runtime: {datetime.timedelta(seconds=int(total_time))}")
    print("ε-constrained design optimization process completed.")

if __name__ == "__main__":
    main()
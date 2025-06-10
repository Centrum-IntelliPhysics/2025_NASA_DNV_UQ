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


class PerformanceBasedOptimizer:
    """
    Implements performance-based design optimization for the NASA/DNV UQ Challenge.
    
    This solves Question 1: Find the control variable X_c that maximizes the minimum
    performance metric across all epistemic uncertainties.
    
    Enhanced with timeout management and checkpointing capabilities.
    """
    def __init__(self, exe_path, data_path, n_epistemic_samples=35, 
                 checkpoint_dir="performance_checkpoints", max_runtime_hours=48):
        self.exe_path = exe_path
        self.data_path = Path(data_path)
        self.n_epistemic_samples = n_epistemic_samples
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize timeout manager
        self.timeout_mgr = TimeoutManager(max_runtime_hours)
        
        # Indices for performance outputs
        self.I1 = [0, 1, 2]  # Performance indices (y1, y2, y3)
        
        # Load data
        self.load_data()
        
        # Control variable bounds (0 to 1 for all three)
        self.bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Store optimization history
        self.iteration_history = []
        self.objective_history = []
        
        # Setup function evaluation counting and caching
        self.func_eval_count = 0
        self.eval_cache = {}  # Standard cache
        self.high_fidelity_cache = {}  # Cache for high-fidelity evaluations
        
        # Best solution found so far
        self.best_solution = {
            "X_c": np.array([0.5, 0.5, 0.5]),  # Default initial guess
            "objective": float('-inf'),
            "iteration": 0
        }
        
        # Initialize criticality tracking for epistemic samples
        self.epistemic_criticality = np.zeros(n_epistemic_samples)
        
        # Restore from checkpoint if available
        self.restore_checkpoint()
        
        print(f"PerformanceBasedOptimizer initialized with all aleatory samples and {n_epistemic_samples} epistemic samples")
        
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
    
    def evaluate_performance_metric(self, Xa, Xe, Xc, seeds=None):
        """
        Evaluate the performance metric J(Xe, Xc) for given samples.
        
        Args:
            Xa: Aleatory samples (n_samples x 2)
            Xe: Epistemic parameters (3,)
            Xc: Control parameters (3,)
            seeds: Random seeds for simulation
            
        Returns:
            performance_value: Expected value of the performance metric
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
            return np.nan
        
        # Calculate objective function (performance metric)
        J_values = np.zeros(n_samples)
        for i in range(n_samples):
            J_sample = 0
            for idx in self.I1:
                # Sum over time steps
                J_sample += np.sum(Y_out[i, :, idx])*(1/60)
            J_values[i] = J_sample
        
        # Expected value (average over all samples)
        performance_value = np.mean(J_values)
        
        return performance_value
    
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
            "eval_cache": self.eval_cache,
            "high_fidelity_cache": self.high_fidelity_cache,
            "best_solution": self.best_solution,
            "epistemic_criticality": self.epistemic_criticality,
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
            self.plot_optimization_progress(save_path=self.checkpoint_dir / "performance_optimization_progress.png")
        
        print(f"\n=== CHECKPOINT SAVED ===")
        print(f"Saved to: {versioned_checkpoint_file}")
        print(f"Iteration: {self.best_solution['iteration']}")
        print(f"Best objective: {self.best_solution['objective']:.6f}")
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
            self.eval_cache = checkpoint["eval_cache"]
            
            # Restore high-fidelity cache if available
            if "high_fidelity_cache" in checkpoint:
                self.high_fidelity_cache = checkpoint["high_fidelity_cache"]
            
            # Restore function evaluation count if available
            if "func_eval_count" in checkpoint:
                self.func_eval_count = checkpoint["func_eval_count"]
            
            # Restore epistemic criticality tracking if available
            if "epistemic_criticality" in checkpoint:
                self.epistemic_criticality = checkpoint["epistemic_criticality"]
            
            self.best_solution = checkpoint["best_solution"]
            
            print(f"Restored checkpoint from iteration {self.best_solution['iteration']}")
            print(f"Best objective value so far: {self.best_solution['objective']:.6f}")
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
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Performance metric plot
        ax.plot(self.iteration_history, self.objective_history, 'b-o', label='Objective')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Performance-Based Optimization Progress')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Progress plot saved to {save_path}")
        else:
            plt.show()
    
    def performance_based_design(self, hyperparam_set=None, max_iter=30, verbose=True):
        """
        Solve the performance-based design optimization problem:
        Find X_c that maximizes min(X_e∈E) J(X_e, X_c)
        
        Args:
            hyperparam_set: Hyperparameters for differential evolution
            max_iter: Maximum number of optimization iterations
            verbose: Whether to print progress
            
        Returns:
            X_c_optimal: Optimal control parameters
            best_obj: Best objective value
        """
        if verbose:
            print("Solving Performance-Based Design Optimization Problem...")
            print(f"Max iterations: {max_iter}")
            print(f"Time limit: {self.timeout_mgr.max_runtime_seconds/3600:.1f} hours")
        
        # Setup iteration counter
        iteration_counter = max(0, self.best_solution["iteration"])
        
        # Define the objective function to be maximized
        def objective_function(X_c):
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
                return -obj_value  # Negate for maximization
            
            # Then check standard cache
            if X_c_tuple in self.eval_cache:
                cached_results = self.eval_cache[X_c_tuple]
                obj_value = cached_results["objective"]
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
            
            # For each X_c, find the worst-case performance
            worst_case_J = float('inf')
            worst_X_e = None
            
            # Track criticality of each epistemic sample
            criticality = np.zeros(len(E_subset))
            
            # Evaluate performance for each epistemic sample
            for i, X_e in enumerate(E_subset):
                # Evaluate performance metric
                J = self.evaluate_performance_metric(Xa_subset, X_e, X_c)
                
                # Track criticality
                criticality[i] = J if not np.isnan(J) else float('inf')
                
                # Update worst-case objective
                if J < worst_case_J and not np.isnan(J):
                    worst_case_J = J
                    worst_X_e = X_e
            
            # Handle NaN values
            if np.isnan(worst_case_J) or np.isinf(worst_case_J):
                worst_case_J = float('-inf')  # Very bad performance
            
            # Update criticality tracking
            self.epistemic_criticality = criticality
            
            # All evaluations are high-fidelity since we're using all samples
            is_high_fidelity = True
            
            # Store in appropriate cache
            eval_result = {
                "objective": worst_case_J,
                "worst_X_e": worst_X_e,
                "fidelity": "high" if is_high_fidelity else "low"
            }
            
            # Always update regular cache
            self.eval_cache[X_c_tuple] = eval_result
            
            # Update high-fidelity cache if appropriate
            if is_high_fidelity:
                self.high_fidelity_cache[X_c_tuple] = eval_result
                if verbose and self.func_eval_count % 10 == 0:
                    print(f"X_c: {X_c}, HIGH-FIDELITY evaluation - obj: {worst_case_J:.4f}")
            elif verbose and self.func_eval_count % 20 == 0:
                print(f"X_c: {X_c}, obj: {worst_case_J:.4f}")
            
            # Check if this solution is better than the current best
            if worst_case_J > self.best_solution["objective"]:
                # Update best solution
                self.best_solution["X_c"] = X_c.copy()
                self.best_solution["objective"] = worst_case_J
                
                if verbose and self.func_eval_count % 10 == 0:
                    print(f"New best solution from func eval! Obj: {worst_case_J:.4f}")
            
            # Check for checkpoint
            if self.timeout_mgr.should_checkpoint():
                # Save the current iteration info before checkpointing
                # This ensures we capture the latest solution status in our history
                if X_c_tuple not in self.eval_cache and len(self.iteration_history) > 0:
                    # Add this to history if not already there
                    self.iteration_history.append(iteration_counter)
                    self.objective_history.append(worst_case_J)
                    
                self.save_checkpoint()
            
            # Return the negative of worst-case J for maximization
            return -worst_case_J

        # Callback function to track optimization progress
        def callback(xk, convergence=None):
            nonlocal iteration_counter
            
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
                source = "high-fidelity cache"
            elif X_c_tuple in self.eval_cache:
                cached_results = self.eval_cache[X_c_tuple]
                current_obj = cached_results["objective"]
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
                
                for X_e in E_subset:
                    # Evaluate performance metric
                    J = self.evaluate_performance_metric(Xa_subset, X_e, current_X_c)
                    
                    if J < worst_case_J and not np.isnan(J):
                        worst_case_J = J
                
                # Handle NaN values
                if np.isnan(worst_case_J) or np.isinf(worst_case_J):
                    worst_case_J = float('-inf')
                
                current_obj = worst_case_J
                source = "new evaluation"
                
                # Cache these results
                self.eval_cache[X_c_tuple] = {
                    "objective": current_obj,
                }
            
            # Always store progress at each callback
            self.iteration_history.append(iteration_counter)
            self.objective_history.append(current_obj)
            
            # Update best solution if better
            if current_obj > self.best_solution["objective"]:
                self.best_solution["X_c"] = current_X_c.copy()
                self.best_solution["objective"] = current_obj
                self.best_solution["iteration"] = iteration_counter
                
                if verbose:
                    print(f"New best solution found! Objective: {current_obj:.4f}")
            
            # Always print progress for each iteration
            if verbose:
                elapsed = self.timeout_mgr.elapsed_time_formatted()
                print(f"Iteration {iteration_counter}: Obj = {current_obj:.4f} " +
                    f"(Elapsed: {elapsed}, Source: {source})")
            
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
                    'mutation': 0.8,         # Slightly higher mutation for better exploration
                    'recombination': 0.9,
                    'popsize': 15,
                    'tol': 0.01,
                    'polish': True,          # Enable polishing for better final result
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
                objective_function, 
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
                print(f"{'Iteration':<10}{'Objective':<15}")
                print("-"*25)
                for i, (iter_num, obj) in enumerate(zip(self.iteration_history, self.objective_history)):
                    print(f"{iter_num:<10}{obj:<15.4f}")
            
            # Save final checkpoint
            self.save_checkpoint(final=not terminated_early)
            
            # Plot the optimization progress
            self.plot_optimization_progress()
            
            # Return the best solution found
            return self.best_solution["X_c"], self.best_solution["objective"]["objective"]
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Save what we have so far
            self.save_checkpoint(final=False)
            
            # Still return the best solution found, even if optimization failed
            return self.best_solution["X_c"], self.best_solution


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
    
    # Create a results directory
    results_dir = Path("performance_results")
    results_dir.mkdir(exist_ok=True)
    
    # Define maximum runtime in hours
    max_runtime_hours = 72  # 72 hours = 3 days
    
    print("NASA/DNV UQ Challenge - Performance-Based Design Optimization")
    print("=" * 80)
    print(f"Maximum runtime: {max_runtime_hours} hours")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set epistemic sample size
    n_epistemic = 10  # Number of epistemic samples
    
    print(f"Using all available aleatory samples and {n_epistemic} epistemic samples")
    
    # Define hyperparameter set
    hyperparam_set = {
        'strategy': 'best1bin',    # Use best1bin for exploitation
        'mutation': 0.8,           # Slightly increased mutation rate for better exploration
        'recombination': 0.9,      # High recombination rate
        'popsize': 15,             # Moderate population size
        'tol': 0.001,              # Reduced tolerance to avoid early convergence
        'polish': True,            # Keep polishing enabled
        'maxiter': 30              # Increased max iterations
    }
    
    print(f"\nUsing hyperparameters:")
    for key, value in hyperparam_set.items():
        print(f"  {key}: {value}")
    
    # Create checkpoint directory
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create the optimizer with checkpointing enabled
    optimizer = PerformanceBasedOptimizer(
        exe_path=exe_path, 
        data_path=data_path, 
        n_epistemic_samples=n_epistemic,
        checkpoint_dir=checkpoint_dir,
        max_runtime_hours=max_runtime_hours
    )
    
    # Run the optimization
    print("\nStarting performance-based design optimization...")
    X_c_optimal, best_obj = optimizer.performance_based_design(
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
            "elapsed_time": optimizer.timeout_mgr.elapsed_time_formatted(),
            "completed": completed,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to files
        np.save(results_dir / "performance_based_results.npy", result_info)
        
        # Also save in JSON format for better readability
        with open(results_dir / "performance_based_results.json", "w") as f:
            json.dump(result_info, f, indent=4)
        
        print(f"\nResults saved to {results_dir / 'performance_based_results.json'}")
        
        # Report final status
        print(f"\nOptimization {completion_status}")
        print(f"Final solution: X_c = {X_c_optimal}")
        print(f"Objective value (worst-case performance): {best_obj:.4f}")
    
    total_time = time.time() - optimizer.timeout_mgr.start_time
    print(f"\nTotal runtime: {datetime.timedelta(seconds=int(total_time))}")
    print("Performance-based design optimization process completed.")

if __name__ == "__main__":
    main()
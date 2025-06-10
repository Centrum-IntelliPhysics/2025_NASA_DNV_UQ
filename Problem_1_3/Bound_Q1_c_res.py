import numpy as np
from pathlib import Path
import subprocess
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import json

def run_forward_model(input_data, exe_path):
    """
    Run the forward model simulation for given input data
    Args:
        input_data: Array of input samples [Xa1, Xa2, Xe1, Xe2, Xe3, Xc1, Xc2, Xc3, seed]
        exe_path: Path to the model executable
    """
    # Save input data to temporary file
    np.savetxt("temp_input.txt", input_data, delimiter=',')
    
    # Run simulation
    try:
        command = [exe_path, "temp_input.txt"]
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
        
        # Reshape to have dimensions (num_samples, timesteps, components)
        Y_out = df.to_numpy().reshape(num_samples, 60, 6)
        
        return Y_out
        
    except subprocess.TimeoutExpired:
        print("Simulation timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"Simulation error: {e}")
        return None

def monte_carlo_simulation(Xa_samples, Xe, Xc, n_samples, exe_path):
    """
    Perform Monte Carlo simulation for fixed Xe and Xc using the provided aleatory samples
    
    Args:
        Xa_samples: Array of aleatory samples to sample from
        Xe: Fixed epistemic variable values
        Xc: Fixed control variable values
        n_samples: Number of Monte Carlo samples
        exe_path: Path to the executable
        
    Returns:
        Y_out: Output array from simulation
        runtime: Simulation runtime in seconds
    """
    start_time = time.time()
    
    # Sample from the aleatory variables with replacement
    sample_indices = np.random.choice(len(Xa_samples), size=n_samples, replace=True)
    Xa_subset = Xa_samples[sample_indices]
    
    # Create input matrix with fixed Xe and Xc
    Xe_repeated = np.tile(Xe, (n_samples, 1))
    Xc_repeated = np.tile(Xc, (n_samples, 1))
    seeds = np.random.randint(1, 10000, size=(n_samples, 1))
    
    # Combine all inputs
    X_input = np.column_stack((Xa_subset, Xe_repeated, Xc_repeated, seeds))
    
    # Run forward model
    Y_out = run_forward_model(X_input, exe_path)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    return Y_out, runtime

def calculate_prob_bounds(Y_all_xe, component, thresholds):
    """
    Calculate max pu and pl across all epistemic samples for a given component
    
    Args:
        Y_all_xe: List of Y outputs for different epistemic samples
        component: Component index
        thresholds: Array of threshold values
        
    Returns:
        max_pu_values: Maximum upper probability for each threshold
        max_pl_values: Maximum lower probability for each threshold
    """
    max_pu_values = []
    max_pl_values = []
    
    for threshold in thresholds:
        # Calculate pu (probability of max > threshold)
        pu_values = []
        for Y in Y_all_xe:
            max_values = np.max(Y[:, :, component], axis=1)  # Max over time for each sample
            pu = np.mean(max_values > threshold)
            pu_values.append(pu)
        max_pu_values.append(max(pu_values))
        
        # Calculate pl (probability of min < threshold)
        pl_values = []
        for Y in Y_all_xe:
            min_values = np.min(Y[:, :, component], axis=1)  # Min over time for each sample
            pl = np.mean(min_values < threshold)
            pl_values.append(pl)
        max_pl_values.append(max(pl_values))
    
    return max_pu_values, max_pl_values

def find_prediction_intervals(Xa_samples, Xe_samples, Xc, alphas, n_mc_samples, exe_path):
    """
    Find prediction intervals for all components given aleatory and epistemic samples
    
    Args:
        Xa_samples: Array of aleatory samples
        Xe_samples: Array of epistemic samples
        Xc: Fixed control parameters
        alphas: Confidence levels (e.g., [0.999, 0.95])
        n_mc_samples: Number of Monte Carlo samples per simulation
        exe_path: Path to the executable
        
    Returns:
        intervals: Dictionary of intervals for each component and alpha
        pu_values_all: Dictionary of pu values for each component
        pl_values_all: Dictionary of pl values for each component
    """
    print(f"Finding prediction intervals for Xc: {Xc}")
    print(f"Using {len(Xa_samples)} aleatory samples")
    print(f"Using {len(Xe_samples)} epistemic samples")
    
    mc_times = []
    intervals = {}
    pu_values_all = {}  # Store all pu values for each component
    pl_values_all = {}  # Store all pl values for each component
    
    # Initialize intervals dictionary
    for alpha in alphas:
        intervals[alpha] = {i: {'lower': None, 'upper': None} for i in range(6)}
    
    # For each component
    for component in range(6):
        Y_all_xe = []  # Store simulation results for all epistemic variables
        
        # Process all epistemic samples
        for i, Xe in enumerate(Xe_samples):
            print(f"Running simulation for epistemic sample {i+1}/{len(Xe_samples)}: {Xe}")
            
            # Monte Carlo simulation with fixed Xe and Xc
            Y_mc, mc_time = monte_carlo_simulation(Xa_samples, Xe, Xc, n_mc_samples, exe_path)
            mc_times.append(mc_time)
            
            if Y_mc is not None:
                Y_all_xe.append(Y_mc)
                print(f"  Simulation completed in {mc_time:.2f} seconds")
            else:
                print(f"  Simulation failed for epistemic sample {i+1}")
        
        if not Y_all_xe:
            print(f"No valid simulations for component {component}, skipping")
            continue
        
        # Extract component data from all simulations
        Y_component_all = np.concatenate([Y[:, :, component] for Y in Y_all_xe], axis=0)
        
        # Define range of thresholds
        Y_min, Y_max = np.min(Y_component_all), np.max(Y_component_all)
        thresholds = np.linspace(Y_min, Y_max, 5000)
        
        # Calculate max pu and pl for all thresholds
        max_pu_values, max_pl_values = calculate_prob_bounds(Y_all_xe, component, thresholds)
        
        # Store for Zipf analysis and visualization later
        pu_values_all[component] = dict(zip(thresholds, max_pu_values))
        pl_values_all[component] = dict(zip(thresholds, max_pl_values))
        
        # Find intervals for each alpha
        for alpha in alphas:
            # Upper bound: inf{u | max pu(Xe, Xc, u) ≤ 1-α}
            upper_candidates = [u for u, p in zip(thresholds, max_pu_values) if p <= 1-alpha]
            upper_bound = min(upper_candidates) if upper_candidates else Y_max
            
            # Lower bound: sup{l | max pl(Xe, Xc, l) ≤ 1-α}
            lower_candidates = [l for l, p in zip(thresholds, max_pl_values) if p <= 1-alpha]
            lower_bound = max(lower_candidates) if lower_candidates else Y_min
            
            intervals[alpha][component]['lower'] = lower_bound
            intervals[alpha][component]['upper'] = upper_bound
            
            print(f"Component {component}, Alpha {alpha}:")
            print(f"  Lower bound: {lower_bound:.6f}")
            print(f"  Upper bound: {upper_bound:.6f}")
    
    # Print Monte Carlo timing statistics
    if mc_times:
        print("\nMonte Carlo Timing Statistics:")
        print(f"Average run time: {np.mean(mc_times):.2f} seconds")
        print(f"Total computation time: {np.sum(mc_times):.2f} seconds")
    
    return intervals, pu_values_all, pl_values_all

def plot_prediction_intervals(intervals, alphas, save_path=None):
    """
    Plot the prediction intervals for each component and confidence level
    
    Args:
        intervals: Dictionary of intervals
        alphas: Confidence levels
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Different colors for different alphas
    colors = ['blue', 'red', 'green', 'orange']
    
    for component in range(6):
        ax = axes[component]
        
        for i, alpha in enumerate(alphas):
            lower = intervals[alpha][component]['lower']
            upper = intervals[alpha][component]['upper']
            
            # Create a horizontal line for the interval
            ax.plot([alpha, alpha], [lower, upper], color=colors[i % len(colors)], linewidth=2)
            
            # Add markers for the endpoints
            ax.plot(alpha, lower, 'o', color=colors[i % len(colors)])
            ax.plot(alpha, upper, 'o', color=colors[i % len(colors)])
            
            # Add text labels
            ax.text(alpha, lower - 0.1, f"{lower:.3f}", ha='center', va='top', fontsize=9)
            ax.text(alpha, upper + 0.1, f"{upper:.3f}", ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f"Component {component+1}")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Response Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set the x-ticks to the alpha values
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.3f}" for a in alphas])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Intervals plot saved to: {save_path}")
    else:
        plt.show()

def plot_probability_curves(pu_values, pl_values, component, save_path=None):
    """
    Plot the probability curves (pu and pl) for a specific component
    
    Args:
        pu_values: Dictionary of upper bound probability values
        pl_values: Dictionary of lower bound probability values
        component: Component index
        save_path: Path to save the plot
    """
    thresholds_pu = np.array(list(pu_values[component].keys()))
    probs_pu = np.array(list(pu_values[component].values()))
    
    thresholds_pl = np.array(list(pl_values[component].keys()))
    probs_pl = np.array(list(pl_values[component].values()))
    
    # Sort by thresholds
    pu_sorted = np.argsort(thresholds_pu)
    pl_sorted = np.argsort(thresholds_pl)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot pu and pl curves
    ax.plot(thresholds_pu[pu_sorted], probs_pu[pu_sorted], 'b-', label='pu(u): P(max y > u)')
    ax.plot(thresholds_pl[pl_sorted], probs_pl[pl_sorted], 'r-', label='pl(l): P(min y < l)')
    
    # Add horizontal line at common alpha values
    for alpha in [0.001, 0.05]:
        ax.axhline(y=1-alpha, color='k', linestyle='--', 
                   label=f'1-α = {1-alpha:.3f}')
    
    ax.set_xlabel('Threshold Value')
    ax.set_ylabel('Probability')
    ax.set_title(f'Probability Curves for Component {component+1}')
    ax.grid(True)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Probability curves saved to: {save_path}")
    else:
        plt.show()

def main():
    # Set paths
    data_path = Path("/home/droysar1/data_sgoswam4/Dibakar/NASA_DNV/UM_model_pred/Samples_Xa/")
    results_dir = Path("prediction_intervals_results")
    results_dir.mkdir(exist_ok=True)
    
    # Set parameters
    n_mc_samples = 100  # Monte Carlo samples per simulation
    alphas = [0.999, 0.95]  # Confidence levels
    n_epistemic_samples = 10  # Number of epistemic samples to generate
    
    # Fixed control parameters (baseline design)
    Xc = np.array([0.533, 0.666, 0.5])
    
    # Load aleatory samples from .npy files
    print("Loading aleatory samples...")
    xa1 = np.load(data_path / "Xa1_stratified_samples_100.npy")
    xa2 = np.load(data_path / "Xa2_stratified_samples_100.npy")
    
    # Reshape and combine into a single array
    xa1 = xa1.reshape(-1, 1)  # Ensure column vector
    xa2 = xa2.reshape(-1, 1)  # Ensure column vector
    Xa_samples = np.hstack((xa1, xa2))
    print(f"Loaded {len(Xa_samples)} aleatory samples")
    
    # Define bounds for epistemic variables
    xe1_min, xe1_max = 0.322, 0.323  # Bounds for Xe1
    xe2_min, xe2_max = 0.588, 0.592  # Bounds for Xe2
    xe3_min, xe3_max = 0.5258, 0.5659  # Bounds for Xe3
    
    # Generate epistemic samples using uniform distributions
    print("Generating epistemic samples...")
    Xe_samples = np.zeros((n_epistemic_samples, 3))
    Xe_samples[:, 0] = xe1_min + (xe1_max - xe1_min) * np.random.random(n_epistemic_samples)
    Xe_samples[:, 1] = xe2_min + (xe2_max - xe2_min) * np.random.random(n_epistemic_samples)
    Xe_samples[:, 2] = xe3_min + (xe3_max - xe3_min) * np.random.random(n_epistemic_samples)
    
    print(f"Generated {n_epistemic_samples} epistemic samples")
    print(f"Xe1 range: [{xe1_min:.4f}, {xe1_max:.4f}]")
    print(f"Xe2 range: [{xe2_min:.4f}, {xe2_max:.4f}]")
    print(f"Xe3 range: [{xe3_min:.4f}, {xe3_max:.4f}]")
    
    # Path to the executable
    exe_path = os.path.abspath(Path("local_model_linux"))
    
    # Measure total execution time
    total_start_time = time.time()
    
    # Find prediction intervals
    intervals, pu_values, pl_values = find_prediction_intervals(
        Xa_samples, 
        Xe_samples, 
        Xc, 
        alphas, 
        n_mc_samples, 
        exe_path
    )
    
    # Plot results
    print("\nGenerating plots...")
    plot_prediction_intervals(
        intervals, 
        alphas, 
        save_path=results_dir / "prediction_intervals.png"
    )
    
    # Plot probability curves for each component
    for component in range(6):
        plot_probability_curves(
            pu_values, 
            pl_values, 
            component, 
            save_path=results_dir / f"probability_curves_component_{component+1}.png"
        )
    
    # Save results to files
    print("\nSaving results...")
    
    # Save intervals as JSON
    intervals_json = {}
    for alpha in alphas:
        intervals_json[str(alpha)] = {}
        for component in range(6):
            intervals_json[str(alpha)][str(component)] = {
                'lower': float(intervals[alpha][component]['lower']),
                'upper': float(intervals[alpha][component]['upper'])
            }
    
    with open(results_dir / "prediction_intervals.json", 'w') as f:
        json.dump(intervals_json, f, indent=2)
    
    # Save intervals as text file
    with open(results_dir / "prediction_intervals.txt", 'w') as f:
        f.write("Prediction Intervals:\n")
        for alpha in alphas:
            f.write(f"\nAlpha = {alpha}\n")
            for component in range(6):
                f.write(f"\nComponent {component + 1}:\n")
                f.write(f"Lower bound: {intervals[alpha][component]['lower']:.6f}\n")
                f.write(f"Upper bound: {intervals[alpha][component]['upper']:.6f}\n")
    
    # Save data as numpy arrays for later use
    np.save(results_dir / "Xa_samples.npy", Xa_samples)
    np.save(results_dir / "Xe_samples.npy", Xe_samples)
    np.save(results_dir / "intervals.npy", intervals)
    
    # Print timing information
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main()
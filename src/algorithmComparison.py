import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_optimizers_results(base_dir, optimizers, num_sets=5):
    colors = ["blue", "orange", "green", "red", "purple"]

    # Create a figure 
    fig, axs = plt.subplots(len(optimizers), 2, figsize=(15, 5 * len(optimizers)))

    for opt_idx, optimizer in enumerate(optimizers):
        epoch_ax = axs[opt_idx, 0]  # Left: Epoch vs Loss
        time_ax = axs[opt_idx, 1]   # Right: Time vs Loss

        for i in range(1, num_sets + 1):
            # Construct the file path based on the script's location and the relative path
            file_path = os.path.join(base_dir, f"../results/{optimizer}_results_{i}.csv")

            try:
                # Load data
                data = pd.read_csv(file_path)
                data.columns = data.columns.str.strip().str.lower()
                epochs = data['epoch']
                loss = data['loss']
                time = data['time']

                # Plot Epoch vs Loss
                epoch_ax.plot(epochs, loss, label=f"Set {i}", color=colors[i - 1])

                # Plot Time vs Loss
                time_ax.plot(time, loss, label=f"Set {i}", color=colors[i - 1])

            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping.")
            except KeyError as e:
                print(f"KeyError in {file_path}: {e}. Check column names.")

        # Customize Epoch vs Loss plot
        epoch_ax.set_title(f"{optimizer.upper()} - Epoch vs Loss")
        epoch_ax.set_xlabel("Epoch")
        epoch_ax.set_ylabel("Loss")
        epoch_ax.legend()
        epoch_ax.grid()

        # Customize Time vs Loss plot
        time_ax.set_title(f"{optimizer.upper()} - Time vs Loss")
        time_ax.set_xlabel("Time")
        time_ax.set_ylabel("Loss")
        time_ax.legend()
        time_ax.grid()

    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Ensure this points to the script location
    
    # Create the 'figures' directory in the correct location (relative to the script)
    figures_dir = os.path.join(base_dir, "../results")
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "../results/2all_optimizers_results.png"))
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Automatically points to the script's location

    # Define the path to the 'results' folder relative to the script's directory
    results_dir = os.path.join(base_dir, '..', 'results')  # Moves one level up and enters 'results'

    # List of optimizers to plot
    optimizers = ["gd", "sgd", "adam"]

    # Call the function to plot results
    plot_optimizers_results(base_dir, optimizers)

import numpy as np
import matplotlib.pyplot as plt

# Function to simulate signal strength for a given number of antennas
def simulate_signal_strength(grid_size, num_antennas):
    signal_strength_data = np.zeros((grid_size, grid_size))
    for _ in range(num_antennas):
        signal_strength_data += np.random.rand(grid_size, grid_size)
    signal_strength_data /= num_antennas  # Average the signal strength
    return signal_strength_data

# Preprocess data (e.g., normalize signal strength values)
def normalize_data(signal_strength_data):
    return (signal_strength_data - np.min(signal_strength_data)) / (np.max(signal_strength_data) - np.min(signal_strength_data))

# Function to detect jamming locations
def detect_jamming(signal_strength_data, threshold):
    jamming_locations = np.where(signal_strength_data < threshold)
    return jamming_locations

# Function to plot the detected jamming locations
def plot_jamming_locations(normalized_signal_strength_data, jamming_locations, title):
    plt.imshow(normalized_signal_strength_data, cmap='viridis', origin='lower')
    plt.colorbar(label='Normalized Signal Strength')
    plt.title(title)
    plt.scatter(jamming_locations[1], jamming_locations[0], color='red', label='Jamming Locations')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (Hz)')
    plt.legend()
    plt.show()

# Parameters
grid_size = 10  # Size of the grid (e.g., 10x10)
threshold = 0.2  # Example threshold for detecting jamming activity
num_runs = 1000  # Number of simulations for statistical analysis

# Arrays to store the number of jamming locations for each run
jamming_counts = {num_antennas: [] for num_antennas in [3, 2, 1]}

# Run simulations
for _ in range(num_runs):
    for num_antennas in jamming_counts.keys():
        signal_strength_data = simulate_signal_strength(grid_size, num_antennas)
        normalized_signal_strength_data = normalize_data(signal_strength_data)
        jamming_locations = detect_jamming(normalized_signal_strength_data, threshold)
        jamming_counts[num_antennas].append(len(jamming_locations[0]))

# Plotting results for a single run
for num_antennas in [3, 2, 1]:
    signal_strength_data = simulate_signal_strength(grid_size, num_antennas)
    normalized_signal_strength_data = normalize_data(signal_strength_data)
    jamming_locations = detect_jamming(normalized_signal_strength_data, threshold)
    plot_jamming_locations(normalized_signal_strength_data, jamming_locations, f'Detected Jamming Locations ({num_antennas} Antenna(s))')

# Statistical analysis
mean_jamming_counts = {num_antennas: np.mean(jamming_counts[num_antennas]) for num_antennas in jamming_counts}
std_jamming_counts = {num_antennas: np.std(jamming_counts[num_antennas]) for num_antennas in jamming_counts}

print("Mean number of jamming locations:")
for num_antennas, mean in mean_jamming_counts.items():
    print(f"{num_antennas} Antenna(s): {mean:.2f}")

print("\nStandard deviation of jamming locations:")
for num_antennas, std in std_jamming_counts.items():
    print(f"{num_antennas} Antenna(s): {std:.2f}")

# Plot histograms for visual comparison
plt.hist(jamming_counts[3], bins=range(0, grid_size**2 + 1), alpha=0.5, label='3 Antennas')
plt.hist(jamming_counts[2], bins=range(0, grid_size**2 + 1), alpha=0.5, label='2 Antennas')
plt.hist(jamming_counts[1], bins=range(0, grid_size**2 + 1), alpha=0.5, label='1 Antenna')
plt.xlabel('Number of Jamming Locations')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Jamming Locations')
plt.show()

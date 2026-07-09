import sys
import numpy as np
import pandas as pd
from scipy.signal import welch, csd, detrend, butter, sosfilt
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib.animation as animation

# ================================================================
# 1. USER INPUTS
# ================================================================
# This section defines the key parameters for your analysis.
csv_file = r"C:\Users\12345\OneDrive - Louisiana State University\Tests on Pedestrian Bridges\January 19 - Vibration Test on Pedestrian Bridge - 5th Day\January 19 - Vibration Test on Pedestrian Bridge - 5th Day - 1st Bridge.csv"

fs = 1000              # Sampling rate (Hz): How many data points per second.
nperseg = 8192        # Welch segment length: Controls frequency resolution. Higher = finer detail. 
                        #(for our data, 1024, or 2048 is good. Freq resolution = fs/nperseg). We typically like to have 0.05 Hz resolution in our frequency domain.
overlap_frac = 0.5    # Fractional overlap (50%): Standard for Welch's method to reduce variance (can be also 60%)
smooth_sigma = 0.1    # Smoothing: A small blur applied to the singular value curve to make it cleaner (don't make this higher than .5)
                        # typically smoothing is not applied, keep it .1 for now.

# Bridge Geometry (Meters)
# Defines the physical size of the structure for the 3D plot.
L = 14.0  # Length
W = 5.0   # Width

# ================================================================
# 2. FILTERING
# ================================================================
def bandpass_filter(data, fs, lowcut=0.5, highcut=30.0, order=1):
    """
    Removes low-frequency drift (gravity/tilt) and high-frequency electrical noise.
    Keeps only vibrations between 0.5 Hz and 30.0 Hz.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, data, axis=0)

# ================================================================
# 3. PSD & FDD ALGORITHM
# ================================================================
def compute_psd_matrix(data, fs, nperseg, noverlap):
    """
    Calculates the Power Spectral Density (PSD) Matrix.
    For 18 channels, this creates an 18x18 matrix at EVERY frequency line.
    - Diagonal: Auto-PSD (Energy of one sensor).
    - Off-Diagonal: Cross-PSD (Correlation between two sensors).
    """
    Nch = data.shape[1]
    freq, _ = welch(data[:, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    PSD = np.zeros((len(freq), Nch, Nch), dtype=complex)

    for i in range(Nch):
        _, Pii = welch(data[:, i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        PSD[:, i, i] = Pii
        for j in range(i + 1, Nch):
            # Calculate Cross-Spectral Density
            _, Pij = csd(data[:, i], data[:, j], fs=fs,
                         nperseg=nperseg, noverlap=noverlap)
            PSD[:, i, j] = Pij
            PSD[:, j, i] = np.conj(Pij) # Symmetry property
    return freq, PSD

def run_fdd_first_sv(data, fs, nperseg, overlap_frac, smooth_sigma):
    """
    The Core FDD Function.
    1. Computes the PSD Matrix.
    2. Performs Singular Value Decomposition (SVD) at each frequency.
    3. Returns the singular values (energy) and mode shapes (vectors).
    """
    Ns = data.shape[0]
    local_nperseg = min(nperseg, Ns)
    noverlap = int(local_nperseg * overlap_frac)

    freq, PSD = compute_psd_matrix(data, fs, local_nperseg, noverlap)

    Nf = len(freq)
    Nch = data.shape[1]
    singular_values = np.zeros((Nf, Nch))
    mode_shapes = np.zeros((Nf, Nch, Nch), dtype=complex)

    for k in range(Nf):
        # SVD decomposes the PSD matrix into U, S, Vh
        # U contains the mode shape vectors.
        # S contains the singular values (amplitudes).
        U, S, Vh = svd(PSD[k])
        singular_values[k, :] = S
        mode_shapes[k, :, :] = U

    # Smooth the singular value curve for better visualization
    sv_smooth = gaussian_filter1d(singular_values, sigma=smooth_sigma, axis=0)
    return freq, singular_values, sv_smooth, mode_shapes

# ================================================================
# 4. LOAD & PREPROCESS DATA
# ================================================================
df = pd.read_csv(csv_file, header=0)

# CRITICAL STEP: Skip the first column (Time)
# We select all rows (:) and columns from index 1 onwards (1:)
acc_raw = df.iloc[:, 1:].values
acc_detrended = detrend(acc_raw, axis=0)
acc_filtered = bandpass_filter(acc_detrended, fs)

Ns, Nch = acc_filtered.shape
print(f"Final data (filtered): {Ns} samples × {Nch} channels")

# ================================================================
# 5. RUN FDD
# ================================================================
freq, singular_values, sv_smooth, mode_shapes = run_fdd_first_sv(
    acc_filtered, fs, nperseg, overlap_frac, smooth_sigma
)

# Convert to Decibels (dB) for better plotting
sv_db = 10 * np.log10(sv_smooth + 1e-20)

# Limit the view to 0-30 Hz (relevant for bridges)
mask = (freq >= 0.0) & (freq <= 30.0)
freq_lim = freq[mask]
sv_db_lim = sv_db[mask, :]
modes_lim = mode_shapes[mask, :, :]

# ================================================================
# 6. INTERACTIVE PEAK PICKING (UPDATED)
# ================================================================
selected_peaks = {}            
selected_freq_vector = []      
is_picking_done = False

fig, ax = plt.subplots(figsize=(14, 7))

# 1. Plot the First Singular Value (The primary structural mode) - Bold & Navy
ax.plot(freq_lim, sv_db_lim[:, 0], linewidth=2.5, color='navy', label="SV1")

# 2. Plot ALL remaining Singular Values (SV2 to SV18) - Thin & Faded
# Nch is the total number of channels (18) calculated earlier
for i in range(1, Nch):
    ax.plot(freq_lim, sv_db_lim[:, i], linewidth=1.5, alpha=0.4, color='gray')

ax.set_title("1. Click Peaks to Select.  2. Press ENTER to Animate.", fontsize=18, fontweight="bold")
ax.set_xlabel("Frequency (Hz)", fontsize=16)
ax.set_ylabel("Singular Values (dB)", fontsize=16)
ax.grid(True)

# Custom legend to keep it clean (don't list all 18 items)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='navy', lw=2.5),
                Line2D([0], [0], color='gray', lw=2.5, alpha=0.4)]
ax.legend(custom_lines, ['1st Singular Value', 'Rest of the Singular Values'], loc="upper right")

def on_key(event):
    global is_picking_done
    if event.key == 'enter':
        is_picking_done = True
        plt.close(fig) 

def onclick(event):
    global selected_freq_vector
    if event.button != MouseButton.LEFT: return
    if event.inaxes is None: return
    if event.canvas.toolbar.mode != "": return

    # Find the closest frequency bin to where the user clicked
    f0 = event.xdata
    idx = np.argmin(np.abs(freq_lim - f0))
    f_selected = freq_lim[idx]

    if idx in selected_peaks:
        selected_peaks[idx].remove()
        del selected_peaks[idx]
        print(f"[Removed] {f_selected:.4f} Hz")
    else:
        line = ax.axvline(f_selected, color="red", linestyle="--", linewidth=2)
        selected_peaks[idx] = line
        print(f"[Selected] {f_selected:.4f} Hz")

    selected_freq_vector = sorted([freq_lim[i] for i in selected_peaks.keys()])
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_key)

print("Please select peaks in the window...")
plt.show(block=True) # Pauses execution until window is closed

# ================================================================
# 7. GEOMETRY, ANIMATION & SURFACE
# ================================================================
if not selected_peaks:
    print("No peaks were selected. Exiting.")
    sys.exit()

print("Starting Animation for selected frequencies:", np.round(selected_freq_vector, 3))

my_animations = []
results_list = []

# Define the 10 Nodes of the Bridge Wireframe
node_coords = np.array([
    [0, 0, 0],      # 0: Fixed Start Left
    [3.5, 0, 0],    # 1: Sensor 1
    [7.0, 0, 0],    # 2: Sensor 2
    [10.5, 0, 0],   # 3: Sensor 3
    [14.0, 0, 0],   # 4: Fixed End Left
    [0, W, 0],      # 5: Fixed Start Right
    [3.5, W, 0],    # 6: Sensor 4
    [7.0, W, 0],    # 7: Sensor 5
    [10.5, W, 0],   # 8: Sensor 6
    [14.0, W, 0]    # 9: Fixed End Right
])

# Define lines connecting the nodes
lines = [
    [0, 1], [1, 2], [2, 3], [3, 4],        # Side 1
    [5, 6], [6, 7], [7, 8], [8, 9],        # Side 2
    [0, 5], [1, 6], [2, 7], [3, 8], [4, 9] # Cross beams
]

# Map sensor data channels to physical nodes
# e.g., First 3 channels -> Node 1, Next 3 channels -> Node 2
sensor_node_map = [1, 2, 3, 6, 7, 8]

# Grid indices for creating the surface mesh (2 rows x 5 columns)
grid_indices = np.array([
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]
])

def animate_mode(f_hz, mode_vec):
    # Scale factor makes tiny displacements visible
    scale_factor = 1.0 / np.max(np.abs(mode_vec)) * 2.5 

    fig_anim = plt.figure(figsize=(10, 6))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    
    ax_anim.set_xlim(-1, L+1)
    ax_anim.set_ylim(-1, W+1)
    ax_anim.set_zlim(-2, 2)
    ax_anim.set_xlabel("X (m)")
    ax_anim.set_ylabel("Y (m)")
    ax_anim.set_zlabel("Z")
    ax_anim.set_title(f"Mode: {f_hz:.2f} Hz")

    # Draw static wireframe (undeformed) in grey
    for start, end in lines:
        p1, p2 = node_coords[start], node_coords[end]
        ax_anim.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k--', alpha=0.1)

    # Initialize moving blue lines
    plot_lines = []
    for _ in lines:
        line, = ax_anim.plot([], [], [], 'b-', linewidth=2)
        plot_lines.append(line)
        
    surf_artist = [None] # Container to hold the surface object

    def update(frame):
        # Calculate phase angle for simple harmonic motion
        phase = 2 * np.pi * frame / 40.0
        current_coords = node_coords.copy()
        
        # Apply complex mode shape rotation
        # Real part gives physical displacement at this instant
        # this rotates the complex vector by a varying angle (phase).
        rotated_mode = (mode_vec * np.exp(1j * phase)).real
        
        for i_sens in range(6):
            node_idx = sensor_node_map[i_sens]
            dx = rotated_mode[i_sens*3 + 0] * scale_factor
            dy = rotated_mode[i_sens*3 + 1] * scale_factor
            dz = rotated_mode[i_sens*3 + 2] * scale_factor
            
            current_coords[node_idx, 0] += dx
            current_coords[node_idx, 1] += dy
            current_coords[node_idx, 2] += dz

        # Update lines
        for i_line, (start, end) in enumerate(lines):
            p1 = current_coords[start]
            p2 = current_coords[end]
            plot_lines[i_line].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            plot_lines[i_line].set_3d_properties([p1[2], p2[2]])
        
        # Update Surface (Remove old one, plot new one)
        if surf_artist[0]:
            surf_artist[0].remove()
        
        X = current_coords[grid_indices, 0]
        Y = current_coords[grid_indices, 1]
        Z = current_coords[grid_indices, 2]
        
        surf = ax_anim.plot_surface(X, Y, Z, color='cyan', alpha=0.3, rstride=1, cstride=1)
        surf_artist[0] = surf
            
        return plot_lines + [surf]

    ani = animation.FuncAnimation(fig_anim, update, frames=40, interval=50, blit=False)
    return ani

# Loop through selected peaks and animate them sequentially
for idx in sorted(selected_peaks.keys()):
    f_hz = freq_lim[idx]
    mode_shape = modes_lim[idx, :, 0]
    
    results_list.append((f_hz, mode_shape))
    
    anim_obj = animate_mode(f_hz, mode_shape)
    my_animations.append(anim_obj)
    
    print(f"Displaying animation for {f_hz:.2f} Hz (Close window to see next)...")
    plt.show(block=True) 

# ================================================================
# 7b. DETAILED MAGNITUDE & PHASE ANALYSIS
# ================================================================
print("\n" + "="*80)
print("       DETAILED MODE SHAPE MAGNITUDE & PHASE")
print("="*80)

# Channel labels: 6 sensors × 3 directions = 18 channels
directions = ['X', 'Y', 'Z']
channel_labels = []
for i_sens in range(1, num_sensors + 1):
    for direction in directions:
        channel_labels.append(f"Sensor {i_sens}-{direction}")

for i, (f_hz, mode_vec) in enumerate(results_list):
    # Normalize to the highest-amplitude channel
    max_idx = np.argmax(np.abs(mode_vec))
    phase_shift = np.angle(mode_vec[max_idx])

    mode_vec_norm = mode_vec * np.exp(-1j * phase_shift)
    mode_vec_norm = mode_vec_norm / np.abs(mode_vec_norm[max_idx])

    magnitude = np.abs(mode_vec_norm)
    phase_deg = np.angle(mode_vec_norm, deg=True)

    print(f"\nMode {i+1}: {f_hz:.3f} Hz (Reference: {channel_labels[max_idx]})")
    print(f"{'Ch':<4} | {'Location':<15} | {'Magnitude':<12} | {'Phase (deg)':<12}")
    print("-" * 55)

    for ch in range(len(mode_vec)):
        label = channel_labels[ch] if ch < len(channel_labels) else f"Ch {ch+1}"
        print(f"{ch+1:<4} | {label:<15} | {magnitude[ch]:.4f}       | {phase_deg[ch]:.2f}")

    # Plot Magnitude & Phase
    fig_mp, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_mp.suptitle(f"Mode {i+1}: {f_hz:.2f} Hz", fontsize=16, fontweight='bold')

    channels = np.arange(len(mode_vec))

    ax1.stem(channels, magnitude, basefmt=" ")
    ax1.set_ylabel("Normalized Magnitude")
    ax1.set_title("Mode Shape Magnitude")
    ax1.grid(True, alpha=0.5)

    ax2.stem(channels, phase_deg, basefmt=" ", linefmt='r-', markerfmt='ro')
    ax2.set_ylabel("Phase (Degrees)")
    ax2.set_title("Mode Shape Phase")
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_ylim(-200, 200)
    ax2.grid(True, alpha=0.5)

    ax2.set_xticks(channels)
    ax2.set_xticklabels(channel_labels, rotation=45, ha='right', fontsize=9)

    plt.subplots_adjust(bottom=0.2)
    print(f"Displaying Magnitude/Phase plot for {f_hz:.2f} Hz...")
    plt.show(block=True)

# 8. SAVE RESULTS
# ================================================================
columns = ['Frequency_Hz'] + [f'Channel_{i+1}' for i in range(18)]
data_rows = []

for freq, vec in results_list:
    # Flatten the data for CSV storage
    row = [freq] + vec.tolist()
    data_rows.append(row)

df_modes = pd.DataFrame(data_rows, columns=columns)
df_modes.to_csv("FDD_Results.csv", index=False)
print("Results saved to FDD_Results.csv")
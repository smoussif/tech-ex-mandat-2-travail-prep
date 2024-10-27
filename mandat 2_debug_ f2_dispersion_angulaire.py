import numpy as np
import matplotlib.pyplot as plt

def calculate_dispersion(wavelength, f2, d=1.6e-6, m=1):
    """Calculate position using proper dispersion equation"""
    lambda_m = wavelength * 1e-9
    theta_m = np.arcsin(m * lambda_m / d)  # from grating equation
    dispersion = (m * f2) / (d * np.cos(theta_m))  # dx/dλ
    return dispersion * lambda_m * 1000  # Convert to mm

def create_peak(x, position, width):
    """Create a rectangular peak"""
    return np.where((x >= position) & (x <= position + width), 1.0, 0.0)

def plot_f2_variation():
    x = np.linspace(-0.2, 8, 1000)
    wavelengths = [400, 450, 500, 550, 600, 700]  # nm
    colors = ['purple', 'blue', 'cyan', 'lime', 'yellow', 'red']
    f2_values = [10, 20, 40]  # mm
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    for ax, f2 in zip(axes, f2_values):
        first_pos = None
        f2_m = f2 / 1000  # Convert to meters
        
        for wavelength, color in zip(wavelengths, colors):
            # Calculate peak position using dispersion
            position = calculate_dispersion(wavelength, f2_m)
            
            # Align first peak to x=0
            if first_pos is None:
                first_pos = position
            position = position - first_pos
            
            width = 0.1  # We'll modify this later for resolution
            
            # Create and plot peak
            peak = create_peak(x, position, width)
            ax.plot(x, peak, color=color, label=f'λ={wavelength}nm')
        
        # Add camera limit
        ax.axvline(x=5.2, color='black', linestyle='--', label='Limite de la caméra')
        
        # Configure subplot
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Intensité')
        ax.set_title(f'f2 = {f2} mm')
        ax.set_xlim(-0.2, 8)
        ax.set_ylim(0, 1.3)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Run simulation
plot_f2_variation()

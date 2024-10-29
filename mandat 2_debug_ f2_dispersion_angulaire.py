import numpy as np
import matplotlib.pyplot as plt

def setup_parameters(f2_value):
    # Convert angle from degrees and minutes to radians
    blaze_angle_deg = 8 + 37/60  # 8 degrees 37 minutes
    blaze_angle = np.deg2rad(blaze_angle_deg)
    
    # Given parameters in mm
    wavelengths = np.array([400, 500, 600, 700]) * 1e-6  # wavelengths in mm (from nm)
    m = 1  # Diffraction order
    grooves_per_mm = 600
    pitch = 1/grooves_per_mm  # Already in mm
    f1 = 25  # mm
    f2 = f2_value  # mm
    a = 0.070  # aperture width in mm (70 μm)
    
    return {
        'blaze_angle': blaze_angle,
        'wavelengths': wavelengths,
        'f1': f1,
        'f2': f2,
        'a': a,
        'pitch': pitch
    }

def calculate_intensity(x, params, wavelength):
    """Calculate the normalized intensity distribution"""
    # Calculate diffraction angle for this wavelength
    theta = np.arcsin(wavelength/params['pitch'])  # First order m=1
    
    # Calculate expected peak position - direct f2 scaling
    x_peak = params['f2'] * np.tan(theta)
    
    # Calculate the scaled position relative to peak position
    x_scaled = (x - x_peak)/(wavelength*params['f2']/params['a'])
    
    # Calculate the phase term (must be dimensionless)
    phase = 2 * np.pi * params['f2'] * np.sin(params['blaze_angle']) / params['pitch']
    
    # Calculate the complete response and its intensity
    response = np.sinc(x_scaled) * np.cos(phase)
    intensity = np.abs(response)**2
    
    return intensity / np.max(intensity)

def plot_for_f2(f2_value, ax):
    # Setup parameters
    params = setup_parameters(f2_value)
    
    # Create fixed position range in mm - same for all plots
    x = np.linspace(0, 24, 1000)  # Fixed range for all plots
    
    # Colors for different wavelengths
    colors = ['b', 'g', 'r', 'm']
    wavelength_labels = ['400 nm', '500 nm', '600 nm', '700 nm']
    
    # Plot each wavelength
    for i, wavelength in enumerate(params['wavelengths']):
        # Calculate normalized intensity
        intensity = calculate_intensity(x, params, wavelength)
        
        # Plot intensity
        ax.plot(x, intensity, color=colors[i], 
               label=wavelength_labels[i], linewidth=2)
        
        # Print peak position for verification
        theta = np.arcsin(wavelength/params['pitch'])
        x_peak = f2_value * np.tan(theta)
        print(f"f2={f2_value}mm, λ={wavelength*1e6}nm: peak at {x_peak:.2f}mm")
    
    ax.set_title(f'f2 = {f2_value} mm')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Wavelength', loc='upper right')
    ax.set_ylabel('Normalized Intensity I(x)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 24)

def main():
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot for each f2 value
    plot_for_f2(10, ax1)
    plot_for_f2(20, ax2)
    plot_for_f2(40, ax3)
    
    plt.suptitle('Main Peaks - Normalized Intensity Distribution\nAperture = 70μm, f1 = 25 mm', y=0.95)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

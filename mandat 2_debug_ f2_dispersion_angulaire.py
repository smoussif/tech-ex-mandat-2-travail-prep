import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy import signal

def define_parameters():
    grooves_per_mm = 600
    blaze_angle = np.radians(8 + 37/60)
    d = 1/grooves_per_mm  # in mm
    N = 25/d  # 25mm divided by d(mm)
    wavelengths = np.array([400e-6, 500e-6, 600e-6, 700e-6])  # in mm (e.g., 400nm = 400e-6 mm)
    blaze_wavelength = 500e-6  # in mm
    
    return {
        'blaze_angle': blaze_angle,
        'blaze_wavelength': blaze_wavelength,
        'grooves_per_mm': grooves_per_mm,
        'N_g': grooves_per_mm,
        'Lambda': d,
        'N': N,
        'wavelengths': wavelengths,
    }

def calculate_Z1(x, a, wavelength, f1):
    E_in = signal.windows.boxcar(len(x))
    fx = x / (wavelength * f1)
    Z1 = a * np.sinc(a * fx)
    return Z1
def calculate_intensity_1D(Z1, x, params, wavelength, f1, f2):
    Lambda = params['Lambda']
    N = params['N']
    blaze_angle = params['blaze_angle']
    
    # Calculate beta
    beta = 2 * (2 * np.pi * np.tan(blaze_angle)) / wavelength
    
    # Modified rect_arg to have narrower width (0.2mm instead of 0.5mm)
    rect_arg = (x - wavelength*f2/Lambda) / 0.1  # Reduced width to 0.2mm
    
    # Calculate the rect function
    rect = np.where(np.abs(rect_arg) <= 1.0, 1, 0)
    
    # Calculate sinc term
    sinc_term = np.sinc(-Lambda * beta)
    
    # Calculate total intensity
    intensity = rect * (sinc_term)**2
    
    return intensity / np.max(intensity)

def plot_1D_subplot(ax, x, Z1_dict, f1, f2):
    plot_params = {
        'xlabel': 'Position (mm)',
        'ylabel': 'Intensity',
        'title': f'f1={f1}mm, f2={f2}mm',
        'xlim': (-0.1, 9),  # Small negative range to show left edge
        'ylim': (0, 1.1),
        'camera_position': 5.2,
        'wavelengths': [400e-6, 500e-6, 600e-6, 700e-6],  # in mm
        'colors': ['purple', 'blue', 'cyan', 'red']
    }
    
    # Calculate 400nm peak position
    reference_wavelength = 400e-6
    reference_intensity = calculate_intensity_1D(Z1_dict[reference_wavelength], x, params, reference_wavelength, f1, f2)
    
    # Find where the 400nm peak starts (left edge)
    nonzero_indices = np.where(reference_intensity > 0.01)[0]
    if len(nonzero_indices) > 0:
        left_edge_position = x[nonzero_indices[0]]
    else:
        left_edge_position = 0
    
    # Plot each wavelength with shifted x axis
    for wavelength, color in zip(plot_params['wavelengths'], plot_params['colors']):
        intensity = calculate_intensity_1D(Z1_dict[wavelength], x, params, wavelength, f1, f2)
        ax.plot(x - left_edge_position, intensity, color=color, label=f'{wavelength*1e6:.0f} nm')
    
    # Add camera line at fixed 5.2mm position
    ax.axvline(x=5.2, color='black', linestyle='--', label='Camera')
    
    # Set labels and title
    ax.set_xlabel(plot_params['xlabel'])
    ax.set_ylabel(plot_params['ylabel'])
    ax.set_title(plot_params['title'])
    
    # Set axis limits
    ax.set_xlim(plot_params['xlim'])
    ax.set_ylim(plot_params['ylim'])
    
    ax.grid(True, alpha=0.3)
    ax.legend()

def main():
    global params
    params = define_parameters()
    
    # Extended x range with higher resolution
    x = np.linspace(-0.5, 15, 3000)  # Increased resolution
    
    f1 = 25  # in mm
    f2_values = [10, 20, 40]  # in mm
    
    fig_1d, axes_1d = plt.subplots(3, 1, figsize=(10, 15))
    
    for i, f2 in enumerate(f2_values):
        Z1_dict = {}
        for wavelength in params['wavelengths']:
            Z1 = calculate_Z1(x, 1, wavelength, f1)
            Z1_dict[wavelength] = Z1
        
        plot_1D_subplot(axes_1d[i], x, Z1_dict, f1, f2)
    
    fig_1d.suptitle('1D Spectral Peaks for Different f2 Values')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

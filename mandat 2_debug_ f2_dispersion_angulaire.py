import numpy as np
import matplotlib.pyplot as plt

def setup_parameters(f2_value, f1_value, aperture_size=0.070):
    blaze_angle_deg = 8 + 37/60
    blaze_angle = np.deg2rad(blaze_angle_deg)
    wavelengths = np.array([400, 500, 600, 700]) * 1e-6
    grooves_per_mm = 600
    pitch = 1/grooves_per_mm
    return {
        'blaze_angle': blaze_angle,
        'wavelengths': wavelengths,
        'f1': f1_value,
        'f2': f2_value,
        'a': aperture_size,
        'pitch': pitch
    }

def calculate_intensity_f2(x, params, wavelength):
    theta = np.arcsin(wavelength/params['pitch'])
    x_peak = params['f2'] * np.tan(theta)
    x_scaled = (x - x_peak)/(wavelength*params['f2']/params['a'])
    phase = 2 * np.pi * params['f2'] * np.sin(params['blaze_angle']) / params['pitch']
    response = np.sinc(x_scaled) * np.cos(phase)
    intensity = np.abs(response)**2
    return intensity / np.max(intensity), x_peak

def calculate_intensity_f1(x, params, wavelength):
    theta = np.arcsin(wavelength/params['pitch'])
    x_peak = params['f1'] * np.tan(theta)
    x_scaled = (x - x_peak)/(wavelength*params['f1']/params['a'])
    phase = 2 * np.pi * params['f1'] * np.sin(params['blaze_angle']) / params['pitch']
    response = np.sinc(x_scaled) * np.cos(phase)
    intensity = np.abs(response)**2
    return intensity / np.max(intensity), x_peak

def calculate_fwhm(x, intensity):
    half_max_indices = np.where(intensity >= 0.5)[0]
    if len(half_max_indices) > 0:
        return x[half_max_indices[-1]] - x[half_max_indices[0]]
    return None

def analyze_resolution(f2, f1, aperture_microns, wavelength_nm):
    wavelength = wavelength_nm * 1e-6
    aperture = aperture_microns * 1e-3
    params = setup_parameters(f2, f1, aperture)
    
    theta = np.arcsin(wavelength/params['pitch'])
    if f2 == 25:  # If analyzing f1 variation
        x_peak = params['f1'] * np.tan(theta)
    else:  # If analyzing f2 variation
        x_peak = params['f2'] * np.tan(theta)
        
    x = np.linspace(x_peak - 1, x_peak + 1, 1000)
    if f2 == 25:  # If analyzing f1 variation
        intensity, _ = calculate_intensity_f1(x, params, wavelength)
    else:  # If analyzing f2 variation
        intensity, _ = calculate_intensity_f2(x, params, wavelength)
    fwhm = calculate_fwhm(x, intensity)
    
    return {
        'f2': f2,
        'f1': f1,
        'aperture': aperture_microns,
        'wavelength': wavelength_nm,
        'peak_position': x_peak,
        'spatial_resolution': fwhm
    }

def main():
    # Create figures
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    fig2, (ax5, ax6, ax7, ax8) = plt.subplots(4, 1, figsize=(10, 12))
    fig3, ax9 = plt.subplots(figsize=(12, 8))
    
    # Define wavelength colors that match their electromagnetic spectrum
    wavelength_colors = ['#9B30FF',  # 400nm - violet
                        '#00FF00',    # 500nm - green
                        '#FFA500',    # 600nm - orange
                        '#FF0000']    # 700nm - red
    wavelength_labels = ['400 nm', '500 nm', '600 nm', '700 nm']
    
    print("\nSpatial Resolution Analysis")
    print("==========================")
    
    # 1. f2 variations (f1=25mm, a=70μm fixed)
    print("\nf2 Variations (f1=25mm, a=70μm, λ=500nm):")
    print("----------------------------------------")
    f2_values = [10, 20, 25, 40]
    for f2 in f2_values:
        result = analyze_resolution(f2, 25, 70, 500)
        print(f"f2 = {f2}mm:")
        print(f"  Spatial resolution (FWHM): {result['spatial_resolution']:.3f} mm")
        print(f"  Peak position: {result['peak_position']:.3f} mm")
    
    # 2. f1 variations (f2=25mm, a=70μm fixed)
    print("\nf1 Variations (f2=25mm, a=70μm, λ=500nm):")
    print("----------------------------------------")
    f1_values = [10, 20, 25, 40]
    for f1 in f1_values:
        result = analyze_resolution(25, f1, 70, 500)
        print(f"f1 = {f1}mm:")
        print(f"  Spatial resolution (FWHM): {result['spatial_resolution']:.3f} mm")
        print(f"  Peak position: {result['peak_position']:.3f} mm")
    
    # 3. Aperture variations (f1=25mm, f2=25mm fixed)
    print("\nAperture Variations (f1=25mm, f2=25mm, λ=500nm):")
    print("---------------------------------------------")
    aperture_values = [30, 45, 60, 70, 100]
    for a in aperture_values:
        result = analyze_resolution(25, 25, a, 500)
        print(f"Aperture = {a}μm:")
        print(f"  Spatial resolution (FWHM): {result['spatial_resolution']:.3f} mm")
        print(f"  Peak position: {result['peak_position']:.3f} mm")
    
    # Plot f2 variations
    axes1 = [ax1, ax2, ax3, ax4]
    for f2, ax in zip(f2_values, axes1):
        params = setup_parameters(f2, 25)
        x = np.linspace(0, 24, 2000)
        for i, wavelength in enumerate(params['wavelengths']):
            intensity, x_peak = calculate_intensity_f2(x, params, wavelength)
            ax.plot(x, intensity, color=wavelength_colors[i], 
                   label=wavelength_labels[i], linewidth=2)
        ax.set_title(f'f1=25 mm, f2 = {f2} mm', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Wavelength', loc='upper right')
        ax.set_ylabel('Normalized Intensity I(x)')
        ax.set_xlabel('Position (mm)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 24)
    
    # Plot f1 variations
    axes2 = [ax5, ax6, ax7, ax8]
    for f1, ax in zip(f1_values, axes2):
        params = setup_parameters(25, f1)
        x = np.linspace(0, 24, 2000)
        for i, wavelength in enumerate(params['wavelengths']):
            intensity, x_peak = calculate_intensity_f1(x, params, wavelength)
            ax.plot(x, intensity, color=wavelength_colors[i], 
                   label=wavelength_labels[i], linewidth=2)
        ax.set_title(f'f2=25mm, f1 = {f1} mm', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Wavelength', loc='upper right')
        ax.set_ylabel('Normalized Intensity I(x)')
        ax.set_xlabel('Position (mm)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 24)
    
    # Plot aperture variations
    wavelength = 500e-6
    theta = np.arcsin(wavelength/(1/600))
    x_peak = 25 * np.tan(theta)
    x = np.linspace(x_peak - 1, x_peak + 1, 2000)
    
    # Use shades of green for aperture plot (since using 500nm)
    aperture_colors = ['#90EE90',    # Light green
                      '#3CB371',    # Medium sea green
                      '#228B22',    # Forest green
                      '#006400',    # Dark green
                      '#004000']    # Very dark green
    
    aperture_sizes = [30, 45, 60, 70, 100]
    
    for i, aperture in enumerate(aperture_sizes):
        params = setup_parameters(25, 25, aperture/1000)
        intensity, _ = calculate_intensity_f2(x, params, wavelength)
        fwhm = calculate_fwhm(x, intensity)
        ax9.plot(x, intensity, color=aperture_colors[i], 
                label=f'a={aperture}μm, FWHM={fwhm:.3f}mm', 
                linewidth=2)
    
    ax9.set_title('Intensity Profiles for Different Aperture Sizes (λ=500nm)', fontsize=12)
    ax9.grid(True, alpha=0.3)
    ax9.legend(title='Aperture Size', loc='upper right')
    ax9.set_ylabel('Normalized Intensity I(x)')
    ax9.set_xlabel('Position (mm)')
    ax9.set_ylim(-0.05, 1.25)
    ax9.set_xlim(x_peak - 1, x_peak + 1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

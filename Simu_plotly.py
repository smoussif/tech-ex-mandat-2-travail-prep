# Annexe code python

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm

def setup_parameters(f2_value, f1_value=1000, aperture_size=0.070):
    d = 4.968  # mm, camera width
    blaze_angle_deg = 8 + 37/60
    blaze_angle = np.deg2rad(blaze_angle_deg)
    grooves_per_mm = 600
    pitch = 1 / grooves_per_mm  # grating pitch in mm
    wavelengths = np.linspace(400, 700, 300) * 1e-6 # wavelengths in mm
    return {
        'largeur_camera': d,
        'blaze_angle': blaze_angle,
        'wavelengths': wavelengths,
        'f1': f1_value,
        'f2': f2_value,
        'a': aperture_size,
        'pitch': pitch
    }

def calculate_dispersion(params):
    f2 = np.linspace(0, 50, 1000)  # mm
    dispersion = f2 * (max(params['wavelengths']) - min(params['wavelengths'])) / params['pitch']
    f2_spotted = params['largeur_camera'] * params['pitch'] / (max(params['wavelengths']) - min(params['wavelengths']))
    return f2, dispersion, {'f2_spotted': f2_spotted}

def calculate_resolution(params, a_ref=100e-6, f1_ref=50):
    f1 = np.linspace(1, 50, 100)  # mm
    a = np.linspace(1e-6, 100e-6, 100)  # mm
    f1_grid, a_grid = np.meshgrid(f1, a)
    resolution = params['pitch'] * a_grid / f1_grid
    resolution_f1 = params['pitch'] * a_ref / f1
    resolution_a = params['pitch'] * a / f1_ref
    return f1_grid, a_grid, resolution, resolution_f1, resolution_a

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1, 0)

def calculate_u2(params, f1=1000):
    x = np.linspace(5, 12.5, 10000)
    _, _, f2_result = calculate_dispersion(params)
    f2 = f2_result['f2_spotted']
    Lambda = params['pitch']
    a = params['a']
    lambda_wavelengths = params['wavelengths']
    u2_results = []
    
    # Calculate the first wavelength's peak position to determine shift
    first_lambda = lambda_wavelengths[0]
    u2_rect = rect((x - (first_lambda * f2 / Lambda)) * f1 / (a * f2))
    u2_sinc = np.sinc(x * Lambda / (first_lambda * f2))
    u2_first = np.abs(u2_rect * u2_sinc) ** 2
    peak_index = np.argmax(u2_first)
    shift_value = x[peak_index]  # Calculate the x-shift needed

    # Shift x by the peak position of the first wavelength
    shifted_x = x - shift_value

    for lambda_ in lambda_wavelengths:
        u2_rect = rect((x - (lambda_ * f2 / Lambda)) * f1 / (a * f2))
        u2_sinc = np.sinc(x * Lambda / (lambda_ * f2))
        u2 = np.abs(u2_rect * u2_sinc) ** 2
        norm_u2 = u2 / np.max(np.abs(u2))
        u2_results.append(norm_u2)

    return shifted_x, u2_results, lambda_wavelengths

def main():
    params = setup_parameters(f2_value=11, f1_value=25)
    f2, dispersion, f2_result = calculate_dispersion(params)
    f2_spotted = f2_result['f2_spotted']

    # Dispersion plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f2, y=dispersion, mode='lines', name="Dispersion curve"))
    fig.add_trace(go.Scatter(x=[f2_spotted], y=[params['largeur_camera']], mode='markers+text',
                             text=f"f2_spotted = {f2_spotted:.2f} mm", textposition="top center"))
    fig.update_layout(
        title="Dispersion vs f2", 
        xaxis_title="f2 (mm)", 
        yaxis_title="Dispersion (mm)",
        width=800,  # Width and height adjusted for square shape
        height=600
    )

    # Resolution plot (3D surface)
    
    # Custom darker green colorscale
    moderate_greens = [
        [0, "rgb(0, 50, 0)"],
        [0.2, "rgb(0, 80, 0)"],
        [0.4, "rgb(0, 100, 0)"],
        [0.6, "rgb(0, 120, 0)"],
        [0.8, "rgb(0, 140, 0)"],
        [1, "rgb(0, 160, 0)"]
    ]
    
    f1_grid, a_grid, resolution, resolution_f1, resolution_a = calculate_resolution(params)
    fig3d = go.Figure(data=[go.Surface(z=resolution, x=f1_grid, 
                                       y=a_grid, colorscale=moderate_greens)])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="f1 (mm)",
            yaxis_title="a (mm)",
            zaxis_title="R (mm)"
        ),
        title="Resolution as a function of f1 and aperture width",
        width=900,
        height=900
    )

    # Resolution plots for f1 and a individually
    fig_f1 = go.Figure(go.Scatter(x=f1_grid[0, :], y=resolution_f1, mode='lines'))
    fig_f1.update_layout(
        title="Resolution vs f1 for given aperture", 
        xaxis_title="f1 (mm)", 
        yaxis_title="R (mm)",
        width=800,
        height=600
    )

    fig_a = go.Figure(go.Scatter(x=a_grid[:, 0], y=resolution_a, mode='lines'))
    fig_a.update_layout(
        title="Resolution vs aperture width for given f1", 
        xaxis_title="a (mm)", 
        yaxis_title="R (mm)",
        width=800,
        height=600
    )

    # U2 Function for different wavelengths with first peak-aligned x-axis
    shifted_x, u2_results, wavelengths = calculate_u2(params)
    fig_u2 = go.Figure()
    
    # Generate colors from cm.jet for the number of wavelengths
    jet_colors = [cm.jet(i / len(wavelengths)) for i in range(len(wavelengths))]
    
    for u2, lambda_, color in zip(u2_results, wavelengths, jet_colors):
        rgba_color = f'rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, {color[3]})'
        fig_u2.add_trace(go.Scatter(x=shifted_x, y=u2, mode='lines', 
                                    name=f'λ = {lambda_ * 1e6:.0f} nm', 
                                    line=dict(color=rgba_color)))
    
    fig_u2.update_layout(
        title="U2 Function for Different Wavelengths (Aligned at First Peak)", 
        xaxis_title="Shifted x (mm)", 
        yaxis_title="U2",
        width=800,
        height=600
    )
    
    
    # Display the figure
    fig.show()
    fig3d.show()
    fig_a.show()
    fig_f1.show()
    fig_u2.show()
    
if __name__ == "__main__":
    main()

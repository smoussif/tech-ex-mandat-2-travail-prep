import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
from lmfit import Model
from scipy.signal import peak_widths
import matplotlib.colors as mcolors
import pandas as pd

# Set Plotly to render in the browser
pio.renderers.default = 'browser'

# Define image paths
image_paths = {
    "Blue": r'C:/Users/tomde/Desktop/Cours/Autonme 2024/Tech Ex/Mandat 2/Process_picture/Bleu_carac.png',
    "Red": r'C:/Users/tomde/Desktop/Cours/Autonme 2024/Tech Ex/Mandat 2/Process_picture/Rouge_carac.png',
    "Green": r'C:/Users/tomde/Desktop/Cours/Autonme 2024/Tech Ex/Mandat 2/Process_picture/Vert_spectre.png',
    "White": r'C:/Users/tomde/Desktop/Cours/Autonme 2024/Tech Ex/Mandat 2/Process_picture/Blanc_spectre.png'
}

# Define base colors
base_colors = {
    "Blue": "seagreen",
    "Red": "firebrick",
    "Green": "teal",
    "White": "slategray"
}

# Load images function with error handling
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image at {path} not found.")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Model functions
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * wid**2 / ((x - cen)**2 + wid**2)

# Extract peak intensity profile
def get_peak_intensity_profile(image):
    y_max = np.argmax(np.max(image, axis=1))
    max_profile = image[y_max, :] / np.max(image[y_max, :])
    return max_profile, y_max

# Model fitting process
def lmfit_process(profile):
    x = np.arange(len(profile))
    sigma = peak_widths(profile, [np.argmax(profile)], rel_height=0.99)[0][0]
    
    # Fit models
    gmodel_gauss, gmodel_lorentz = Model(gaussian), Model(lorentzian)
    params_gauss = gmodel_gauss.make_params(amp=np.max(profile), cen=np.argmax(profile), wid=sigma)
    params_lorentz = gmodel_lorentz.make_params(amp=np.max(profile), cen=np.argmax(profile), wid=sigma)
    
    result_gauss = gmodel_gauss.fit(profile, params_gauss, x=x)
    result_lorentz = gmodel_lorentz.fit(profile, params_lorentz, x=x)
    
    # Choose best fit
    best_result, best_fit_label = (result_gauss, "Gaussian Fit") if result_gauss.chisqr < result_lorentz.chisqr else (result_lorentz, "Lorentzian Fit")
    
    x_mean = best_result.params['cen'].value
    width = best_result.params['wid'].value
    delta_x_mean = best_result.params['cen'].stderr or 0
    delta_width = best_result.params['wid'].stderr or 0
    
    return x, best_result.best_fit, best_fit_label, x_mean, delta_x_mean, width, delta_width

# Generate faded colors
def generate_fade_colors(base_color, num_shades=5):
    base_rgb = np.array(mcolors.to_rgb(base_color))
    return [mcolors.to_hex(base_rgb * (1 - (i / (num_shades - 1)) * 0.4)) for i in range(num_shades)]

# Wavelength calculation in nm with error handling
def calculate_lambda(profile, image_dict):
    constants = {
        "lambda_blue": 400e-6, "lambda_red": 700e-6,
        "pitch": 1 / 600, "f1": 50, "f2": 25, "pixel_size": 5.6e-3
    }

    try:
        _, _, _, x_mean_i, _, width_i, _ = lmfit_process(profile)
        blue_profile, _ = get_peak_intensity_profile(image_dict['Blue'])
        red_profile, _ = get_peak_intensity_profile(image_dict['Red'])
        
        _, _, _, x_mean_b, _, _, _ = lmfit_process(blue_profile)
        _, _, _, x_mean_r, _, _, _ = lmfit_process(red_profile)

        lambda_b = (constants["lambda_blue"] + constants["pixel_size"] * constants["pitch"] / constants["f2"] * 
                    (x_mean_i - x_mean_b - width_i * constants["f2"] / constants["f1"])) * 1e6
        lambda_r = (constants["lambda_red"] - constants["pixel_size"] * constants["pitch"] / constants["f2"] * 
                    (x_mean_r - x_mean_i - width_i / 2)) * 1e6
        lambda_mean = np.mean([lambda_b, lambda_r])

        return lambda_b, lambda_r, lambda_mean
    except KeyError as e:
        raise KeyError("Missing required color profile for wavelength calculation") from e

# Load all images
images = {label: load_image(path) for label, path in image_paths.items()}
combined_image = np.clip(sum(images.values()), 0, 255).astype(np.uint8)

# 3D surface plot function
def plot_3d_surface(image):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    fig = go.Figure(data=[go.Surface(z=image, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(title="3D Intensity Plot of Combined Image", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Intensity"))
    fig.show()

# Plot combined 3D image
plot_3d_surface(combined_image)

# Plot intensity profiles with fading effect
fig_line = go.Figure()
stats = []
lambda_stats = []

for label, img in images.items():
    profile, _ = get_peak_intensity_profile(img)
    x_fit, best_fit, best_fit_label, x_mean, delta_x_mean, width, delta_width = lmfit_process(profile)
    
    # Calculate wavelengths in nm
    lambda_b, lambda_r, lambda_mean = calculate_lambda(profile, images)
    
    # Append stats for DataFrame
    stats.append([label, x_mean, delta_x_mean, width, delta_width])
    lambda_stats.append([label, lambda_b, lambda_r, lambda_mean])
    
    # Faded line plots
    fade_colors = generate_fade_colors(base_colors[label], num_shades=7)
    for i, fade_color in enumerate(fade_colors):
        intermediate_profile = profile + (best_fit - profile) * (i + 1) / len(fade_colors)
        fig_line.add_trace(go.Scatter(x=np.arange(len(profile)), y=intermediate_profile, mode='lines', line=dict(color=fade_color), showlegend=False))
    
    fig_line.add_trace(go.Scatter(x=np.arange(len(profile)), y=profile, mode='lines', name=f'{label} Profile', line=dict(color=base_colors[label])))
    fig_line.add_trace(go.Scatter(x=x_fit, y=best_fit, mode='lines', name=f'{label} Best Fit ({best_fit_label})', line=dict(color=base_colors[label], dash='dash')))

# Update layout and show profile plot
fig_line.update_layout(title="Intensity Profiles", xaxis_title="X Position", yaxis_title="Intensity", legend_title="Profiles and Fits")
fig_line.show()

# Display stats DataFrame
stats_df = pd.DataFrame(stats, columns=["Couleur", "x maximum", "Erreur sur le x maximum", "Résolution", "Erreur sur la résolution"]).sort_values(by="x maximum")
lambda_df = pd.DataFrame(lambda_stats, columns=["Couleur", "Lambda (Ref : Blue, nm)", "Lambda (Ref : Red, nm)", "Lambda Mean (nm)"])

print(stats_df)
print(lambda_df)

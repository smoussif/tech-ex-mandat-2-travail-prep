import ctypes
import numpy as np
import threading
from queue import Queue
import time
import sys

# For the live graph
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

# For model fitting and plots
import plotly.graph_objects as go
from lmfit import Model
from scipy.signal import peak_widths
import pandas as pd

# For non-blocking keyboard input on Windows
if sys.platform == 'win32':
    import msvcrt
else:
    import tty
    import termios
    import select

# Load the DLL
dll_path = r"C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\OtherDrivers\LabVIEW\For_64-bit_LabVIEW\uc480_64.dll"
camera_sdk = ctypes.CDLL(dll_path)

print("DLL loaded successfully:", camera_sdk)

# Define constants
IS_SUCCESS = 0
IS_CM_MONO8 = 6  # 8-bit monochrome
IS_WAIT = 1      # Wait flag for capture functions

# Camera settings
DESIRED_PIXEL_CLOCK = 24       # MHz
DESIRED_FRAME_RATE = 13.95     # FPS
DESIRED_EXPOSURE_TIME = 0.07   # milliseconds

# Create threading events for safely stopping threads and triggering image capture
stop_event = threading.Event()
image_capture_event = threading.Event()
image_captured_event = threading.Event()

# Shared data dictionary for inter-thread communication
shared_data = {}

def initialize_camera():
    """Initialize the camera and return the camera handle."""
    hCam = ctypes.c_int()
    status = camera_sdk.is_InitCamera(ctypes.byref(hCam))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to initialize the camera, status code: {status}")
    print("Camera initialized successfully")
    return hCam

def set_camera_parameters(hCam):
    """Set camera parameters: color mode, pixel clock, frame rate, and exposure time."""
    status = camera_sdk.is_SetColorMode(hCam, IS_CM_MONO8)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set color mode, status code: {status}")
    print("Color mode set to IS_CM_MONO8")

    status = camera_sdk.is_SetPixelClock(hCam, DESIRED_PIXEL_CLOCK)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set pixel clock, status code: {status}")
    print(f"Pixel clock set to {DESIRED_PIXEL_CLOCK} MHz")

    actual_frame_rate = ctypes.c_double()
    status = camera_sdk.is_SetFrameRate(hCam, ctypes.c_double(DESIRED_FRAME_RATE), ctypes.byref(actual_frame_rate))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set frame rate, status code: {status}")
    print(f"Frame rate set to {actual_frame_rate.value} FPS")

    actual_exposure_time = ctypes.c_double()
    status = camera_sdk.is_SetExposureTime(hCam, ctypes.c_double(DESIRED_EXPOSURE_TIME), ctypes.byref(actual_exposure_time))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set exposure time, status code: {status}")
    print(f"Exposure time set to {actual_exposure_time.value} ms")

def allocate_image_memory(hCam, width, height, bit_depth):
    """Allocate and set image memory for the camera."""
    image_memory = ctypes.POINTER(ctypes.c_ubyte)()
    mem_id = ctypes.c_int()

    status = camera_sdk.is_AllocImageMem(hCam, width, height, bit_depth, ctypes.byref(image_memory), ctypes.byref(mem_id))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to allocate image memory, status code: {status}")
    print("Image memory allocated successfully")

    status = camera_sdk.is_SetImageMem(hCam, image_memory, mem_id)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set image memory, status code: {status}")
    print("Image memory set successfully")

    return image_memory, mem_id

def capture_images(hCam, image_memory, mem_id, width, height, frame_queue):
    """Thread function to capture frames for live plotting and on-demand image processing."""
    print("Capture thread started.")
    try:
        while not stop_event.is_set():
            status = camera_sdk.is_FreezeVideo(hCam, IS_WAIT)
            if status != IS_SUCCESS:
                print(f"Failed to capture frame, status code: {status}")
                break

            image_size = width * height
            buffer_type = ctypes.c_ubyte * image_size
            image_buffer = ctypes.cast(image_memory, ctypes.POINTER(buffer_type)).contents
            image_array = np.ctypeslib.as_array(image_buffer).reshape((height, width))

            if not frame_queue.full():
                frame_queue.put(image_array.copy())
            else:
                frame_queue.get()
                frame_queue.put(image_array.copy())

            if image_capture_event.is_set():
                shared_data['captured_image'] = image_array.copy()
                image_captured_event.set()
                image_capture_event.clear()

            time.sleep(0.01)
    finally:
        camera_sdk.is_FreeImageMem(hCam, image_memory, mem_id)
        camera_sdk.is_ExitCamera(hCam)
        print("Capture thread terminated.")

def get_peak_intensity_profile(image):
    """Extract the intensity profile along the X-axis at the row with peak intensity."""
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    valid_y = [y for y in range(image.shape[0]) if image[y, x_max] >= 0.95 * image[y_max, x_max]]
    profiles = [image[y, :] for y in valid_y]
    mean_profile = np.mean(profiles, axis=0)
    return mean_profile, y_max

def live_plot(doc, frame_queue):
    """Thread function to process frames and plot them live using Bokeh."""
    print("Live plot thread started.")
    source = ColumnDataSource(data=dict(x=[], y=[]))
    p = figure(x_range=(0, WIDTH), y_range=(0, 256), title='Live Intensity Profile',
               x_axis_label='X-axis', y_axis_label='Intensity', sizing_mode='stretch_both',
               background_fill_color='#fafafa', border_fill_color='white')

    p.line('x', 'y', source=source, line_color='black', line_width=1)

    doc.add_root(p)

    def update():
        if not frame_queue.empty():
            image = frame_queue.get()
            profile_max, y_max = get_peak_intensity_profile(image)
            if profile_max is not None:
                x = np.arange(len(profile_max))
                y = profile_max
                source.data = dict(x=x, y=y)

    doc.add_periodic_callback(update, 50)  # Update every 50 ms

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * wid**2 / ((x - cen)**2 + wid**2)

def lmfit_process(profile):
    """Fit the intensity profile using Gaussian and Lorentzian models."""
    x = np.arange(len(profile))
    sigma = peak_widths(profile, [np.argmax(profile)], rel_height=0.99)[0][0]

    gmodel_gauss = Model(gaussian)
    gmodel_lorentz = Model(lorentzian)

    params_gauss = gmodel_gauss.make_params(amp=np.max(profile), cen=np.argmax(profile), wid=sigma)
    params_lorentz = gmodel_lorentz.make_params(amp=np.max(profile), cen=np.argmax(profile), wid=sigma)

    result_gauss = gmodel_gauss.fit(profile, params_gauss, x=x)
    result_lorentz = gmodel_lorentz.fit(profile, params_lorentz, x=x)

    if result_gauss.chisqr < result_lorentz.chisqr:
        best_result = result_gauss
        best_fit_label = "Gaussian Fit"
    else:
        best_result = result_lorentz
        best_fit_label = "Lorentzian Fit"

    x_mean = best_result.params['cen'].value
    width = best_result.params['wid'].value

    return x, best_result.best_fit, best_fit_label, x_mean, width, best_result

def calculate_lambda(profile):
    """Calculate wavelength based on peak fitting."""
    constants = {
        "lambda_blue": 400e-6,  # reference wavelength in mm
        "lambda_red": 700e-6,   # reference wavelength in mm
        "x_blue": 84.428500,    # Blue reference in mm
        "x_red": 915.436240,    # Blue reference in mm
        "w_blue": 4.390604,     # Pinhole approx with width of blue 
        "w_red": 10.075923,      # Red width
        "pitch": 1 / 600,       # grating pitch in mm
        "f1": 50,               # focal length in mm
        "f2": 25,               # focal length in mm
        "pixel_size": 5.6e-3    # pixel size in mm
    }

    x, _, _, x_mean, width, best_result = lmfit_process(profile)

    lambda_b = (constants["lambda_blue"] + constants["pixel_size"] * constants["pitch"] / constants["f2"] * 
                    (x_mean - constants["x_blue"] - constants["w_blue"] * constants["f2"] / constants["f1"])) * 1e6
    lambda_r = (constants["lambda_red"] - constants["pixel_size"] * constants["pitch"] / constants["f2"] * 
                (constants["x_red"] - x_mean - constants["w_blue"] / 2)) * 1e6

    if x_mean != constants["x_blue"] and x_mean != constants["x_red"]:
            lambda_value = np.array([lambda_b, lambda_r])
            value_weight = np.array([
                constants["w_red"] / ((constants["w_blue"] + constants["w_red"])), 
                constants["w_blue"] / ((constants["w_blue"] + constants["w_red"]))
            ])
    else:
        lambda_value = np.array([lambda_b, lambda_r])
        value_weight = np.array([
            constants["w_red"] / (constants["w_blue"] + constants["w_red"]), 
            constants["w_blue"] / (constants["w_blue"] + constants["w_red"])
        ])
        
    # Normalize weights
    value_weight /= np.sum(value_weight)

    lambda_mean = np.average(lambda_value, weights=value_weight)

    return lambda_mean, best_result

def get_key_pressed():
    """Function to detect key presses in a non-blocking way."""
    if sys.platform == 'win32':
        if msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8')
    else:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
    return None

def main():
    print("Starting main function.")
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = 1280, 1024
    BIT_DEPTH = 8

    hCam = initialize_camera()
    set_camera_parameters(hCam)
    image_memory, mem_id = allocate_image_memory(hCam, WIDTH, HEIGHT, BIT_DEPTH)

    frame_queue = Queue(maxsize=10)

    # Start the capture thread
    capture_thread = threading.Thread(target=capture_images, args=(hCam, image_memory, mem_id, WIDTH, HEIGHT, frame_queue))
    capture_thread.start()

    # Start the Bokeh server for live plotting
    apps = {'/': Application(FunctionHandler(lambda doc: live_plot(doc, frame_queue)))}
    server = Server(apps, port=5006)
    server_thread = threading.Thread(target=server.io_loop.start)
    server_thread.start()
    print("Bokeh server started.")

    try:
        print("Press 't' to take a picture and process it. Press 'q' to quit.")
        if sys.platform != 'win32':
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        server.io_loop.add_callback(server.show, "/")
        while not stop_event.is_set():
            key = get_key_pressed()
            if key == 't':
                print("Capturing and processing image...")
                image_capture_event.set()
                image_captured_event.wait(timeout=5)
                if image_captured_event.is_set():
                    image_captured_event.clear()
                    captured_image = shared_data.get('captured_image')
                    if captured_image is not None:
                        profile, y_max = get_peak_intensity_profile(captured_image)
                        lambda_value, best_result = calculate_lambda(profile)
                        x_mean = best_result.params['cen'].value
                        x_mean_err = best_result.params['cen'].stderr or 0.0
                        resolution = best_result.params['wid'].value
                        resolution_err = best_result.params['wid'].stderr or 0.0

                        print(f"Calculated wavelength (nm): {lambda_value:.2f}")
                        print(f"x_mean: {x_mean:.2f} ± {x_mean_err:.2f}")
                        print(f"Resolution: {resolution:.2f} ± {resolution_err:.2f}")

                        x_fit = np.arange(len(profile))
                        best_fit = best_result.best_fit
                        fig = go.Figure()

                        # Add main profile and best fit to the plot
                        fig.add_trace(go.Scatter(x=x_fit, y=profile, mode='lines', name='Profile', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=x_fit, y=best_fit, mode='lines', name=f'{best_result.model.name} Fit', line=dict(color='red')))

                        # Add text as legend items (simulated legend entries)
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(color='rgba(0,0,0,0)'),
                            showlegend=True,
                            name=f"Calculated Wavelength: {lambda_value:.2f} nm"
                        ))
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(color='rgba(0,0,0,0)'),
                            showlegend=True,
                            name=f"x_mean: {x_mean:.2f} ± {x_mean_err:.2f} pixels"
                        ))
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(color='rgba(0,0,0,0)'),
                            showlegend=True,
                            name=f"Resolution: {resolution:.2f} ± {resolution_err:.2f} pixels"
                        ))


                        fig.update_layout(
                            title=f"Intensity Profile with {best_result.model.name} Fit",
                            xaxis_title="X Position",
                            yaxis_title="Intensity"
                        )

                        fig.show()
                    else:
                        print("No captured image found.")
                else:
                    print("Failed to capture image.")
            elif key == 'q':
                print("Exit signal received.")
                stop_event.set()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected.")
        stop_event.set()
    finally:
        print("Stopping threads and server.")
        stop_event.set()
        capture_thread.join()
        server.io_loop.stop()
        server_thread.join()
        if sys.platform != 'win32':
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("All threads stopped. Exiting program.")

if __name__ == "__main__":
    main()

import ctypes
import numpy as np
import threading
from queue import Queue
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
import time

# Load the DLL
dll_path = r"C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\OtherDrivers\LabVIEW\For_64-bit_LabVIEW\uc480_64.dll"
camera_sdk = ctypes.CDLL(dll_path)

print("DLL loaded successfully:", camera_sdk)

# Define constants
IS_SUCCESS = 0
IS_CM_MONO8 = 6  # 8-bit monochrome
IS_WAIT = 1      # Wait flag for capture functions

# Define IS_PIXELCLOCK_CMD constants
IS_PIXELCLOCK_CMD_SET = 0x8001
IS_PIXELCLOCK_CMD_GET_RANGE = 0x8003

# Camera settings
DESIRED_PIXEL_CLOCK = 24       # MHz
DESIRED_FRAME_RATE = 13.95     # FPS
DESIRED_EXPOSURE_TIME = 0.07   # milliseconds

# Threading lock for camera access
camera_lock = threading.Lock()

# Set restype and argtypes for functions
camera_sdk.is_InitCamera.restype = ctypes.c_int
camera_sdk.is_InitCamera.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]

camera_sdk.is_ExitCamera.restype = ctypes.c_int
camera_sdk.is_ExitCamera.argtypes = [ctypes.c_int]

camera_sdk.is_SetColorMode.restype = ctypes.c_int
camera_sdk.is_SetColorMode.argtypes = [ctypes.c_int, ctypes.c_int]

camera_sdk.is_PixelClock.restype = ctypes.c_int
camera_sdk.is_PixelClock.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]

camera_sdk.is_SetFrameRate.restype = ctypes.c_int
camera_sdk.is_SetFrameRate.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]

camera_sdk.is_SetExposureTime.restype = ctypes.c_int
camera_sdk.is_SetExposureTime.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]

camera_sdk.is_AllocImageMem.restype = ctypes.c_int
camera_sdk.is_AllocImageMem.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)
]

camera_sdk.is_SetImageMem.restype = ctypes.c_int
camera_sdk.is_SetImageMem.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]

camera_sdk.is_FreezeVideo.restype = ctypes.c_int
camera_sdk.is_FreezeVideo.argtypes = [ctypes.c_int, ctypes.c_int]

camera_sdk.is_FreeImageMem.restype = ctypes.c_int
camera_sdk.is_FreeImageMem.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]

def initialize_camera():
    """Initialize the camera and return the camera handle."""
    hCam = ctypes.c_int()
    status = camera_sdk.is_InitCamera(ctypes.byref(hCam), None)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to initialize the camera, status code: {status}")
    print("Camera initialized successfully")
    return hCam

def close_camera(hCam):
    """Close the camera and release resources."""
    status = camera_sdk.is_ExitCamera(hCam)
    if status != IS_SUCCESS:
        print(f"Failed to close the camera, status code: {status}")
    else:
        print("Camera closed successfully")

def set_camera_parameters(hCam):
    """Set camera parameters: color mode, frame rate, and exposure time."""
    # Set color mode to 8-bit monochrome
    status = camera_sdk.is_SetColorMode(hCam, IS_CM_MONO8)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set color mode, status code: {status}")
    print("Color mode set to IS_CM_MONO8")

    # Skip pixel clock settings if querying is not supported
    print("Skipping pixel clock setting due to unsupported function or camera model.")

    # Set frame rate
    actual_frame_rate = ctypes.c_double()
    status = camera_sdk.is_SetFrameRate(hCam, DESIRED_FRAME_RATE, ctypes.byref(actual_frame_rate))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set frame rate, status code: {status}")
    print(f"Frame rate set to {actual_frame_rate.value} FPS")

    # Set exposure time
    actual_exposure_time = ctypes.c_double()
    status = camera_sdk.is_SetExposureTime(hCam, DESIRED_EXPOSURE_TIME, ctypes.byref(actual_exposure_time))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set exposure time, status code: {status}")
    print(f"Exposure time set to {actual_exposure_time.value} ms")

def allocate_image_memory(hCam, width, height, bit_depth):
    """Allocate and set image memory for the camera."""
    image_memory = ctypes.POINTER(ctypes.c_ubyte)()
    mem_id = ctypes.c_int()

    # Allocate image memory
    status = camera_sdk.is_AllocImageMem(
        hCam, width, height, bit_depth, ctypes.byref(image_memory), ctypes.byref(mem_id)
    )
    if status != IS_SUCCESS:
        raise Exception(f"Failed to allocate image memory, status code: {status}")
    print("Image memory allocated successfully")

    # Set image memory
    status = camera_sdk.is_SetImageMem(hCam, image_memory, mem_id)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set image memory, status code: {status}")
    print("Image memory set successfully")

    return image_memory, mem_id

def capture_frames(hCam, image_memory, mem_id, width, height, frame_queue, stop_event):
    """Capture frames in a loop and add them to a queue for processing."""
    try:
        while not stop_event.is_set():
            start_time = time.time()
            with camera_lock:
                status = camera_sdk.is_FreezeVideo(hCam, IS_WAIT)
            if status != IS_SUCCESS:
                print(f"Failed to capture frame, status code: {status}")
                break

            # Access the image data
            image_size = width * height
            buffer_type = ctypes.c_ubyte * image_size
            image_buffer = ctypes.cast(image_memory, ctypes.POINTER(buffer_type)).contents
            image_array = np.ctypeslib.as_array(image_buffer)
            image_array = image_array.reshape((height, width))

            # Downsample the image
            image_array = image_array[::2, ::2]

            # Put the frame into the queue
            if not frame_queue.full():
                frame_queue.put(image_array.copy())
            else:
                frame_queue.get_nowait()
                frame_queue.put(image_array.copy())

            # Small delay to prevent CPU overuse
            time.sleep(0.005)
    finally:
        # Release resources
        with camera_lock:
            camera_sdk.is_FreeImageMem(hCam, image_memory, mem_id)
        print("Capture thread has terminated.")

def get_peak_intensity_profile(image):
    """Extract the normalized intensity profile using vectorized operations."""
    # Find the maximum intensity and its position
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    max_intensity = image[y_max, x_max]
    if max_intensity == 0:
        return None, None

    # Create a mask for rows where the intensity at x_max is >= 95% of max_intensity
    threshold = 0.95 * max_intensity
    valid_rows_mask = image[:, x_max] >= threshold

    # Extract valid profiles
    valid_profiles = image[valid_rows_mask, :]

    if valid_profiles.size == 0:
        return None, None

    # Normalize the profiles
    max_values = valid_profiles.max(axis=1, keepdims=True)
    normalized_profiles = valid_profiles / max_values

    # Compute the mean profile
    mean_profile = normalized_profiles.mean(axis=0)
    return mean_profile, y_max

def process_frames(frame_queue):
    """Process frames in real-time and display a live intensity profile using Dash."""
    # Create a Dash app
    app = dash.Dash(__name__)

    # Shared data variable
    latest_profile = {'x': [], 'y': []}

    # Define the layout of the app
    app.layout = html.Div([
        html.H2("Live Graph"),
        dcc.Graph(id="live-graph", animate=True),
        dcc.Interval(id="graph-update", interval=500, n_intervals=0),
    ])

    # Callback function to update the graph
    @app.callback(
        Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals')]
    )
    def update_graph(n):
        # Update the latest profile if new data is available
        if not frame_queue.empty():
            image = frame_queue.get()
            profile_max, y_max = get_peak_intensity_profile(image)
            if profile_max is not None:
                latest_profile['x'] = np.arange(len(profile_max))
                latest_profile['y'] = profile_max

        # Create the figure
        fig = go.Figure(
            data=[go.Scatter(
                x=latest_profile['x'],
                y=latest_profile['y'],
                mode='lines',
                line=dict(color='yellow')
            )],
            layout=go.Layout(
                title='Live Intensity Profile',
                width=800,  # Adjust the width as desired
                height=600,  # Adjust the height as desired
                xaxis=dict(
                    title='X-axis',
                    range=[0, len(latest_profile['x']) if len(latest_profile['x']) > 0 else 640]
                ),
                yaxis=dict(title='Normalized Intensity', range=[0, 1])
            )
        )
        return fig

    # Run the app using the built-in server
    app.run_server(debug=False, port=8051, use_reloader=False)

def main():
    # Image dimensions and bit depth
    WIDTH, HEIGHT = 1280, 1024
    BIT_DEPTH = 8  # bits per pixel

    hCam = None
    image_memory = None
    mem_id = None
    stop_event = threading.Event()
    frame_queue = Queue(maxsize=10)  # Increased queue size
    capture_thread = None  # Initialize capture_thread to None

    try:
        # Initialize the camera
        hCam = initialize_camera()

        # Set camera parameters
        set_camera_parameters(hCam)

        # Allocate image memory
        image_memory, mem_id = allocate_image_memory(hCam, WIDTH, HEIGHT, BIT_DEPTH)

        # Start the frame capture thread
        capture_thread = threading.Thread(
            target=capture_frames,
            args=(hCam, image_memory, mem_id, WIDTH, HEIGHT, frame_queue, stop_event),
            daemon=True
        )
        capture_thread.start()

        # Run the process_frames function
        process_frames(frame_queue)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Signal the capture thread to stop
        stop_event.set()
        if capture_thread is not None:
            capture_thread.join()

        # Release resources
        if hCam is not None:
            with camera_lock:
                if image_memory is not None and mem_id is not None:
                    camera_sdk.is_FreeImageMem(hCam, image_memory, mem_id)
                close_camera(hCam)

if __name__ == "__main__":
    main()

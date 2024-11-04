import ctypes
import numpy as np
import threading
from queue import Queue
import time
import sys

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from functools import partial

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
    # Set color mode to 8-bit monochrome
    status = camera_sdk.is_SetColorMode(hCam, IS_CM_MONO8)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set color mode, status code: {status}")
    print("Color mode set to IS_CM_MONO8")

    # Set pixel clock
    status = camera_sdk.is_SetPixelClock(hCam, DESIRED_PIXEL_CLOCK)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set pixel clock, status code: {status}")
    print(f"Pixel clock set to {DESIRED_PIXEL_CLOCK} MHz")

    # Set frame rate
    actual_frame_rate = ctypes.c_double()
    status = camera_sdk.is_SetFrameRate(hCam, ctypes.c_double(DESIRED_FRAME_RATE), ctypes.byref(actual_frame_rate))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set frame rate, status code: {status}")
    print(f"Frame rate set to {actual_frame_rate.value} FPS")

    # Set exposure time
    actual_exposure_time = ctypes.c_double()
    status = camera_sdk.is_SetExposureTime(hCam, ctypes.c_double(DESIRED_EXPOSURE_TIME), ctypes.byref(actual_exposure_time))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set exposure time, status code: {status}")
    print(f"Exposure time set to {actual_exposure_time.value} ms")

def allocate_image_memory(hCam, width, height, bit_depth):
    """Allocate and set image memory for the camera."""
    image_memory = ctypes.POINTER(ctypes.c_ubyte)()
    mem_id = ctypes.c_int()

    # Allocate image memory
    status = camera_sdk.is_AllocImageMem(hCam, width, height, bit_depth, ctypes.byref(image_memory), ctypes.byref(mem_id))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to allocate image memory, status code: {status}")
    print("Image memory allocated successfully")

    # Set image memory
    status = camera_sdk.is_SetImageMem(hCam, image_memory, mem_id)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set image memory, status code: {status}")
    print("Image memory set successfully")

    return image_memory, mem_id

def capture_and_display_frames(hCam, image_memory, mem_id, width, height, frame_queue):
    """Capture frames in a loop and add them to a queue for processing."""
    try:
        while True:
            # Capture a single frame
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

            # Put the frame into the queue for processing
            if frame_queue is not None:
                frame_queue.put(image_array.copy())

            # Small delay to prevent CPU overuse
            time.sleep(0.01)

    finally:
        # Release resources
        camera_sdk.is_FreeImageMem(hCam, image_memory, mem_id)
        camera_sdk.is_ExitCamera(hCam)
        print("Resources released and camera closed.")

def get_peak_intensity_profile(image):
    """Extract the intensity profile along the X-axis at the row with peak intensity."""
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    valid_y = [y for y in range(image.shape[0]) if image[y, x_max] >= 0.95 * image[y_max, x_max]]
    profiles = [image[y, :] for y in valid_y]
    mean_profile = np.mean(profiles, axis=0)
    return mean_profile, y_max

def process_frames(doc, frame_queue):
    """Process frames in real time and display a live intensity profile using Bokeh."""
    # Create a Bokeh ColumnDataSource
    source = ColumnDataSource(data=dict(x=[], y=[]))
    
    # Create a Bokeh figure with styling enhancements
    p = figure(x_range=(0, WIDTH), y_range=(0, 256), title='Live Intensity Profile',
               x_axis_label='X-axis', y_axis_label='Intensity', sizing_mode='stretch_both',
               background_fill_color='#fafafa', border_fill_color='white')
    
    # Customizing the line color and width
    p.line('x', 'y', source=source, line_color='black', line_width=1)
    
    # Styling the grid lines and axes
    p.xgrid.grid_line_color = "lightgray"
    p.ygrid.grid_line_color = "lightgray"
    p.axis.major_label_text_font_size = "12pt"
    p.axis.axis_label_text_font_size = "14pt"
    p.axis.axis_label_text_font_style = "bold"
    p.title.text_font_size = "16pt"
    p.title.text_font_style = "bold"

    # Add the plot to the document
    doc.add_root(p)

    # Define the update function
    def update():
        if not frame_queue.empty():
            image = frame_queue.get()
            profile_max, y_max = get_peak_intensity_profile(image)
            if profile_max is not None:
                x = np.arange(len(profile_max))
                y = profile_max
                source.data = dict(x=x, y=y)

    # Add periodic callback to update the plot
    doc.add_periodic_callback(update, 50)  # Update every 50 ms

def main():
    # Camera resolution and bit depth
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = 1280, 1024
    BIT_DEPTH = 8  # bits per pixel

    # Initialize the camera
    hCam = initialize_camera()

    # Set camera parameters
    set_camera_parameters(hCam)

    # Allocate image memory
    image_memory, mem_id = allocate_image_memory(hCam, WIDTH, HEIGHT, BIT_DEPTH)

    # Create a queue for frames
    frame_queue = Queue()

    # Start the camera capture thread
    capture_thread = threading.Thread(target=capture_and_display_frames, args=(hCam, image_memory, mem_id, WIDTH, HEIGHT, frame_queue))
    capture_thread.start()

    # Create a partial function to pass frame_queue to process_frames
    process_frames_with_queue = partial(process_frames, frame_queue=frame_queue)

    # Start the Bokeh server
    apps = {'/': Application(FunctionHandler(process_frames_with_queue))}
    server = Server(apps, port=5006)
    server.start()

    # Open the Bokeh app in the default browser
    server.io_loop.add_callback(server.show, "/")

    try:
        # Run the Bokeh server's IOLoop
        server.io_loop.start()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the server
        server.io_loop.stop()

    # Wait for the capture thread to finish
    capture_thread.join()

if __name__ == "__main__":
    main()

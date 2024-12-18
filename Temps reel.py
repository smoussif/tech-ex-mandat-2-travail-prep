import ctypes
import numpy as np
import cv2
import threading
from queue import Queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import sys

# Load the DLL for the Thorlabs camera SDK
dll_path = r"C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\OtherDrivers\LabVIEW\For_64-bit_LabVIEW\uc480_64.dll"
camera_sdk = ctypes.CDLL(dll_path)

print("DLL loaded successfully:", camera_sdk)

# Define constants for camera operations
IS_SUCCESS = 0
IS_CM_MONO8 = 6  # 8-bit monochrome color mode
IS_WAIT = 1      # Wait flag for capturing video

# Camera configuration settings
DESIRED_PIXEL_CLOCK = 24       # MHz
DESIRED_FRAME_RATE = 13.95      # Frames per second
DESIRED_EXPOSURE_TIME = 0.07   # Exposure time in milliseconds

def initialize_camera():
    """Initialize the camera and return the camera handle."""
    hCam = ctypes.c_int()  # Camera handle as a ctypes integer
    status = camera_sdk.is_InitCamera(ctypes.byref(hCam))  # Initialize the camera
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

    # Set the pixel clock
    status = camera_sdk.is_SetPixelClock(hCam, DESIRED_PIXEL_CLOCK)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set pixel clock, status code: {status}")
    print(f"Pixel clock set to {DESIRED_PIXEL_CLOCK} MHz")

    # Set the frame rate
    actual_frame_rate = ctypes.c_double()
    status = camera_sdk.is_SetFrameRate(hCam, ctypes.c_double(DESIRED_FRAME_RATE), ctypes.byref(actual_frame_rate))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set frame rate, status code: {status}")
    print(f"Frame rate set to {actual_frame_rate.value} FPS")

    # Set the exposure time
    actual_exposure_time = ctypes.c_double()
    status = camera_sdk.is_SetExposureTime(hCam, ctypes.c_double(DESIRED_EXPOSURE_TIME), ctypes.byref(actual_exposure_time))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set exposure time, status code: {status}")
    print(f"Exposure time set to {actual_exposure_time.value} ms")

def allocate_image_memory(hCam, width, height, bit_depth):
    """Allocate and set image memory for the camera."""
    image_memory = ctypes.POINTER(ctypes.c_ubyte)()  # Pointer to the allocated image memory
    mem_id = ctypes.c_int()  # Memory ID for the image memory block

    # Allocate image memory
    status = camera_sdk.is_AllocImageMem(hCam, width, height, bit_depth, ctypes.byref(image_memory), ctypes.byref(mem_id))
    if status != IS_SUCCESS:
        raise Exception(f"Failed to allocate image memory, status code: {status}")
    print("Image memory allocated successfully")

    # Set the allocated image memory for the camera
    status = camera_sdk.is_SetImageMem(hCam, image_memory, mem_id)
    if status != IS_SUCCESS:
        raise Exception(f"Failed to set image memory, status code: {status}")
    print("Image memory set successfully")

    return image_memory, mem_id

def capture_and_display_frames(hCam, image_memory, mem_id, width, height, frame_queue):
    """Capture frames in a loop and display them using OpenCV, adding them to a queue for processing."""
    try:
        while True:
            # Capture a single frame with the wait flag
            status = camera_sdk.is_FreezeVideo(hCam, IS_WAIT)
            if status != IS_SUCCESS:
                print(f"Failed to capture frame, status code: {status}")
                break

            # Access the image data and convert it to a NumPy array
            image_size = width * height
            buffer_type = ctypes.c_ubyte * image_size
            image_buffer = ctypes.cast(image_memory, ctypes.POINTER(buffer_type)).contents
            image_array = np.ctypeslib.as_array(image_buffer)
            image_array = image_array.reshape((height, width))

            # Display the image (commented out but can be activated if needed)
            # cv2.imshow('Live Feed', image_array)

            # Add the frame to the queue for further processing
            if frame_queue is not None:
                frame_queue.put(image_array.copy())

            # Check if 'q' is pressed to break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Free the allocated image memory and exit the camera
        camera_sdk.is_FreeImageMem(hCam, image_memory, mem_id)
        camera_sdk.is_ExitCamera(hCam)
        cv2.destroyAllWindows()
        print("Resources released and camera closed.")

def get_peak_intensity_profile(image):
    """Extract the normalized intensity profile from the row with peak intensity and average rows within 95% of the peak."""
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)  # Find the peak pixel position
    valid_y = [y for y in range(image.shape[0]) if image[y, x_max] >= 0.95 * image[y_max, x_max]]  # Find rows with intensity close to the peak
    profile_max = image[y_max, :] / np.max(image[y_max, :])  # Normalize the peak row
    profiles = [image[y, :] / np.max(image[y, :]) for y in valid_y]  # Normalize each valid row
    mean_profile = np.mean(profiles, axis=0)  # Average the profiles
    return mean_profile, y_max

def process_frames(frame_queue):
    """Process frames in real-time and display a live intensity profile using PyQtGraph."""
    # Create a PyQtGraph application window
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Live Intensity Profile")
    plot = win.addPlot(title="Intensity Profile")
    curve = plot.plot(pen='y')  # Yellow line for the plot

    # Configure the plot's appearance
    plot.setLabel('bottom', 'X-axis')
    plot.setLabel('left', 'Normalized Intensity')
    plot.showGrid(x=True, y=True)

    # Flag to control the running state
    running = True

    # Function to update the plot with new data
    def update_plot():
        nonlocal running
        if not frame_queue.empty():
            frame = frame_queue.get()
            profile_max, y_max = get_peak_intensity_profile(frame)
            if profile_max is not None:
                # Update the plot with the new profile data
                curve.setData(np.arange(len(profile_max)), profile_max)

        # Check if 'q' is pressed to stop the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            app.quit()  # Close the PyQtGraph application gracefully

    # Timer to periodically update the plot
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(50)  # Update interval in milliseconds

    # Run the PyQtGraph application event loop
    if running:
        sys.exit(app.exec_())

def main():
    # Define the camera resolution and bit depth
    WIDTH, HEIGHT = 1280, 1024
    BIT_DEPTH = 8  # 8 bits per pixel for monochrome images

    # Initialize the camera and get the handle
    hCam = initialize_camera()

    # Set the camera's operational parameters
    set_camera_parameters(hCam)

    # Allocate image memory for capturing frames
    image_memory, mem_id = allocate_image_memory(hCam, WIDTH, HEIGHT, BIT_DEPTH)

    # Create a queue for passing frames between threads
    frame_queue = Queue()

    # Start a thread for capturing and displaying frames
    capture_thread = threading.Thread(target=capture_and_display_frames, args=(hCam, image_memory, mem_id, WIDTH, HEIGHT, frame_queue))
    capture_thread.start()

    # Start a thread for processing and plotting the frames
    processing_thread = threading.Thread(target=process_frames, args=(frame_queue,))
    processing_thread.start()

    # Wait for the capture and processing threads to finish
    capture_thread.join()
    processing_thread.join()

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()

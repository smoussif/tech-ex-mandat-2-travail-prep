import ctypes
import numpy as np
import cv2
import threading
from queue import Queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


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
    """Extract the normalized intensity profile."""
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    valid_y = [y for y in range(image.shape[0]) if image[y, x_max] >= 0.95 * image[y_max, x_max]]
    profiles = [image[y, :] / np.max(image[y, :]) for y in valid_y]
    mean_profile = np.mean(profiles, axis=0)
    return mean_profile, y_max

def process_frames(frame_queue):
    """Process frames in real time and display a live intensity profile using Matplotlib."""
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'y-')  # Yellow line for the profile plot
    ax.set_xlim(0, 1280)  # Adjust according to your image width
    ax.set_ylim(0, 1)     # Normalized intensity range
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Live Intensity Profile')
    ax.grid(True)

    def init():
        """Initialize the line for animation."""
        line.set_data([], [])
        return line,

    def update(frame):
        """Update the plot with new frame data."""
        if not frame_queue.empty():
            image = frame_queue.get()
            profile_max, y_max = get_peak_intensity_profile(image)
            if profile_max is not None:
                line.set_data(np.arange(len(profile_max)), profile_max)
        return line,

    # Use Matplotlib's animation module to update the plot
    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

    # Display the plot
    plt.show()


def main():
    # Camera resolution and bit depth
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

    # Run the Dash app using Waitress in the main thread
    process_frames(frame_queue)

    # Wait for the capture thread to finish
    capture_thread.join()

if __name__ == "__main__":
    main()

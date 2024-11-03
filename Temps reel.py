import ctypes
import numpy as np
import cv2

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
DESIRED_FRAME_RATE = 30.0      # FPS
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

def capture_and_display_frames(hCam, image_memory, mem_id, width, height):
    """Capture frames in a loop and display them using OpenCV."""
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

            # Display the image
            cv2.imshow('Live Feed', image_array)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        camera_sdk.is_FreeImageMem(hCam, image_memory, mem_id)
        camera_sdk.is_ExitCamera(hCam)
        cv2.destroyAllWindows()
        print("Resources released and camera closed.")

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

    # Start capturing and displaying frames
    capture_and_display_frames(hCam, image_memory, mem_id, WIDTH, HEIGHT)

if __name__ == "__main__":
    main()

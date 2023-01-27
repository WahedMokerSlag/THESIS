import time
import cv2
from signal import pause
from buildhat import Motor
from camera import Camera
from face import Face
from window import Window
from tracking.track import track
from manual_reset import reset
import numpy as np

left_motor = Motor('C')
right_motor = Motor('B')
camera_motor = Motor('D')
BASE_SPEED = 0

def rotate(rotation):
    """Rotates the robot with a rotation speed

    Args:
        rotation (float): the speed with which to rotate
    """    
    left_speed = rotation
    right_speed = rotation
    left_motor.start(left_speed)
    right_motor.start(right_speed)

def adjust_camera(speed):
    """Adjusts the camera angle according to a given speed

    Args:
        speed (float): The speed with which to adjust the camera angle
    """    
    camera_motor.set_default_speed(speed)
    
def pd_controller(p_value, d_value, optimal_value, new_value, old_error):
    """The PD-Controller

    Args:
        p_value (float): The p-value for the controller
        d_value (float): The d-value for the controller
        optimal_value (float): The optimal value for the PD-Controller
        new_value (float): The value with which to calculate the error
        old_error (float): The previous error

    Returns:
        (float, float): The new error value and the pd value
    """    
    error = optimal_value - new_value
    error_change = error - old_error

    p = p_value * error
    d = d_value * error_change
    pd = p + d

    return error, pd

def horizontal_pd_to_rotation(pd):
    """Converts the horizontal pd value to a rotation

    Args:
        pd (float): The PD-value

    Returns:
        float: the rotation
    """    
    pd = int(pd)
    if pd > 1:
        pd = pd + BASE_SPEED
    elif pd < -1:
        pd = pd - BASE_SPEED
    rotation = int(min(100,max(-100,-pd)))
    return rotation

def vertical_pd_to_speed(pd):
    """Converts the vertical pd value to a speed

    Args:
        pd (float): The vertical pd value

    Returns:
        float: the speed calculated from the PD
    """    
    pd = int(pd)
    speed = min(100,max(-100,-pd))
    return speed


def main():
    """The main function
    """    
    # Initiate the camera
    camera = Camera()

    # Set the PD controller values
    p_value_horizontal = 0.2
    d_value_horizontal = 0.2

    p_value_vertical = 0.2
    d_value_vertical = 1

    # Initialise values to 0
    old_horizontal_error = 0
    old_vertical_error = 0

    # In the optimal situation our error is 0
    optimal_value = 0

    # Set the amount of time the robot will continue looking for a person
    # before stopping.
    time_lost = 0
    time_lost_max = 30

    # Initialise values to 0
    vertical_distance = 0
    horizontal_distance = 0

    # Set default motor speeds to 0
    left_motor.set_default_speed(0)
    right_motor.set_default_speed(0)
    camera_motor.set_default_speed(0)

    # Initial values of the data
    nr_frames = 0
    nr_frames_with_face = 0
    errors_horizontal = []
    errors_vertical = []

    # Compare (if possible) the face from the frame with a photo of a face
    face = Face(user_location="/Location/Of/File")

    # Start the experiment
    print("Start!")
    start_time = time.time()

    # The main loop
    while True:
        # make a picture
        ret, frame = camera.read()

        # collect the face vector by tracking this picture
        face_vector = face.detect_faces(frame)

        # If there is a face, gather the distance to the center of the frame
        if(face_vector):
            horizontal_distance, vertical_distance = face_vector.to_distance()
            time_lost = 0
            nr_frames_with_face += 1
        # Otherwise set the vertical distance to 0 otherwise keep the old
        # distance for a maximum of time_lost_max frames
        else:
            vertical_distance = 0
            time_lost += 1
            if(time_lost == time_lost_max):
                horizontal_distance = 0

        # Gather values from the PD controller
        old_horizontal_error, pd_horizontal = pd_controller(p_value_horizontal, d_value_horizontal, optimal_value, horizontal_distance, old_horizontal_error)
        old_vertical_error, pd_vertical = pd_controller(p_value_vertical, d_value_vertical, optimal_value, vertical_distance, old_vertical_error)

        # If a face was detected, add the errors
        if(face_vector):
            errors_horizontal.append(old_horizontal_error)
            errors_vertical.append(old_vertical_error)
        
        # Convert PD-values to speeds
        rotation_horizontal = horizontal_pd_to_rotation(pd_horizontal)
        speed_vertical = vertical_pd_to_speed(pd_vertical)
        
        # Update the motor with the new speeds
        rotate(rotation_horizontal)
        camera_motor.start(speed_vertical)

        # If q was pressed quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print all the data of the perticular experiment
    hor_error = sum(list(map(abs, errors_horizontal)))/len(errors_horizontal)
    ver_error = sum(list(map(abs, errors_vertical)))/len(errors_vertical)

    print(f"FPS: {nr_frames/(time.time() - start_time)}")
    print(f"Face in sight: {(nr_frames_with_face/nr_frames) * 100}%")
    print(f"Vector error: {np.sqrt(hor_error**2 + ver_error**2)}")
    
    # Disable the motors
    camera_motor.stop()
    left_motor.stop()
    right_motor.stop()
    
    # Release handle to the webcam
    camera.release()
    cv2.destroyAllWindows()

# start the program
if __name__ == "__main__":
    main()
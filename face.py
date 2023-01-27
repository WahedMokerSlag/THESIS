import face_recognition
import cv2
import numpy as np
from face_vector import FaceVector
from window import Window

class Face:
    def __init__(self, user_location="user.jpeg"):
        self.user = "user"
        self.user_image = face_recognition.load_image_file(user_location)
        self.user_face_encoding = face_recognition.face_encodings(self.user_image)[0]
        self.known_face_encodings = [self.user_face_encoding ]
        self.known_face_names = [self.user]
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []



    def detect_faces(self, frame):
        # print("DETECT FACES")

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        self.face_names = [] #? We have removed the original code for this

        # print(len(self.face_locations))

        for face_encoding in self.face_encodings:
            # print("Found a face")

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face wi th the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            self.face_names.append(name)

        

        # print(np.shape(frame))
        if face_vector := self.get_face_vector(frame):
            frame = Window.draw_overlay(frame, face_vector)
            cv2.imshow('Video', frame)
            return face_vector
        # print(np.shape(frame))
        cv2.imshow('Video', frame)
        return None

        

        

    def get_face_vector(self, frame):
        frame_center = (int(frame.shape[1]/2), int(frame.shape[0]/2))
        face_center = None

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # print(name)
            if name == self.user:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                width = right - left
                height = bottom - top

                face_center = (int(left+0.5*width), int(top+0.5*height))
                # print("FOUND USER")
                return FaceVector(face_center,frame_center)
        return None
        

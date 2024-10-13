# Base code that contains Imports, Declarations, and Functions necessary for all other files

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
#-----------------------------------------------------------MEDIAPIPE HOLISTIC SETUP--------------------------------------------------------------------------------------------------------------------
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


#-----------------------------------------------------------FUNCTIONS-------------------------------------------------------------------------------------------------------------------------------------
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion BGR 2 RGB
    #image.flags.writeable = False                  # Image is no longer writable
    results = model.process(image)                 # Make prediction
    #results.flags.writeable = True                 # Image is now writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=4)) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) # Draw right hand connections

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])



def collect_all_data():
        # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_DATA')

    # Actions that we try to detect
    actions = np.array(['hello', 'my', 'name'])

    # Thirdy videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30

    # Create necessary folders
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop Through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    print(results)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # Apply collection logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action,sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # prepare for next recording
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action,sequence), (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)


                    # Export Keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # show to screen
                    #cv2.imshow('OpenCV Feed', image)
            
                    # Break gracefully
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
                    

def data_collection(sign, initial_rep, repititions, num_frames, folder_name, prep_time):
    
    # directory path for the sign's data collection to be saved under
    DATA_PATH = os.path.join(folder_name)
   
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        
        # Loop through each video being recorded
        for rep in range(repititions-initial_rep):

            # give time to prepare before each repitition.
            # display image to feed with intervals of 1 millisecond looped through 'prep_time' amount of times     
            for i in range(prep_time):

                ret, frame = cap.read() # read feed
                image, results = mediapipe_detection(frame, holistic) # Make detections
                draw_styled_landmarks(image, results)   # Draw landmarks

                cv2.putText(image, 'STARTING COLLECTION in {}'.format((prep_time-i)/10), (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(sign,rep+initial_rep), (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # wait for 100 milliseconds
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(9) & 0xFF == ord('q'):
                    return 
                
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    cv2.waitKey(5000)

            # create directory for each rep as they are being collected
            try:
                os.makedirs(os.path.join(DATA_PATH, sign, str(rep+initial_rep)))
            except:
                pass


            # Loop through number of frames in each video
            # Apply collection logic
            for frame_num in range(num_frames):
                
                ret, frame = cap.read() # read feed
                image, results = mediapipe_detection(frame, holistic)   # Make detections
                draw_styled_landmarks(image, results) # Draw landmarks

                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(sign,rep+initial_rep), (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)


                # Export Keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, sign, str(rep+initial_rep), str(frame_num))
                np.save(npy_path, keypoints)

                cv2.waitKey(1)

                
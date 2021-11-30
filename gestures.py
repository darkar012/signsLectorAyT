
import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import threading
from multiprocessing.pool import ThreadPool

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils
# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands
# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def reproducir(song):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(song)
    sound.play()
    time.sleep(30)

def timer ():
    pygame.mixer.init()
    alarm = pygame.mixer.Sound("alarm_ding.ogg")
    alarm_len = alarm.get_length()
    minutes = 0
    seconds = 5
    timer = minutes + seconds
    for i in range(timer):
        print("", str(timer - i), end="\r")
        time.sleep(1)
    alarm.play()
    #print("time's over")
    time.sleep(alarm_len)
    return True

def detectHandsLandmarks(image, hands, draw=False, display=True):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image.
        display: A boolean value that is if set to true the function displays the original input image, and the output
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                     thickness=2, circle_radius=2))

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results


def countFingers(image, results, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image of the hands.
        draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                 output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:     A copy of the input image with the fingers count written, if it was specified.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
    '''

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()

    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}

    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label

        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:

            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]

            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper() + "_" + finger_name] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (
                hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper() + "_THUMB"] = True

            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1

    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:
        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_image, " Total Fingers: ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width // 2 - 150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20, 255, 155), 10, 10)

    # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image, the status of each finger and the count of the fingers up of both hands.
        return output_image, fingers_statuses, count


def recognizeGestures(image, fingers_statuses, count, draw=True, display=True):
    '''
    This function will determine the gesture of the left and right hand in the image.
    Args:
        image:            The image of the hands on which the hand gesture recognition is required to be performed.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        draw:             A boolean value that is if set to true the function writes the gestures of the hands on the
                          output image, after recognition.
        display:          A boolean value that is if set to true the function displays the resultant image and
                          returns nothing.
    Returns:
        output_image:   A copy of the input image with the left and right hand recognized gestures written if it was
                        specified.
        hands_gestures: A dictionary containing the recognized gestures of the right and left hand.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Store the labels of both hands in a list.
    hands_labels = ['RIGHT', 'LEFT']

    # Initialize a dictionary to store the gestures of both hands in the image.
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN", 'BOTH':"UNKNOWN"}

    # Iterate over the left and right hand.
    for hand_index, hand_label in enumerate(hands_labels):

        # Initialize a variable to store the color we will use to write the hands gestures on the image.
        # Initially it is red which represents that the gesture is not recognized.
        color = (0, 0, 255)

        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################

        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        if count[hand_label] == 2 and fingers_statuses[hand_label + '_INDEX'] and fingers_statuses[hand_label + '_MIDDLE']:

            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "serenidad"
            return hands_gestures[hand_label]
            break

            # Update the color value to green.
            color = (0, 255, 0)

        ####################################################################################################################

        # Check if the person is making the 'SPIDERMAN' gesture with the hand.
        ##########################################################################################################################################################

        # Check if the number of fingers up is 3 and the fingers that are up, are the thumb, index and the pinky finger.
        elif count[hand_label] == 2  and fingers_statuses[hand_label + '_THUMB'] and fingers_statuses[hand_label + '_PINKY']:

            # Update the gesture value of the hand that we are iterating upon to SPIDERMAN SIGN.
            hands_gestures[hand_label] = "alegria"
            return hands_gestures[hand_label]
            break
            # Update the color value to green.
            color = (0, 255, 0)

        ##########################################################################################################################################################

        # Check if the person is making the 'HIGH-FIVE' gesture with the hand.
        ####################################################################################################################

        # Check if the number of fingers up is 5, which means that all the fingers are up.
        elif count[hand_label] == 1 and fingers_statuses[hand_label + '_THUMB'] :

            # Update the gesture value of the hand that we are iterating upon to HIGH-FIVE SIGN.
            hands_gestures[hand_label] = "thumb"

            if hands_gestures['RIGHT'] == "thumb" and hands_gestures['LEFT'] == "thumb":
                hands_gestures[hand_label] = "aceptacion"
                return hands_gestures[hand_label]
                break

            # Update the color value to green.
            color = (0, 255, 0)
        elif count[hand_label] == 5:

            # Update the gesture value of the hand that we are iterating upon to HIGH-FIVE SIGN.
            hands_gestures[hand_label] = "high five"

            if hands_gestures['RIGHT'] == "high five" and hands_gestures['LEFT'] == "high five":
                hands_gestures[hand_label] = "esperanza"
                return hands_gestures[hand_label]
                break
        ####################################################################################################################

        elif count[hand_label] == 2 and fingers_statuses[hand_label + '_INDEX'] and fingers_statuses[hand_label + '_THUMB']:
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "L pose"
            if hands_gestures['RIGHT'] == "L pose" and hands_gestures['LEFT'] == "L pose":
                hands_gestures[hand_label] = "melancolia"
                return hands_gestures[hand_label]
                break

        elif count[hand_label] == 3 and fingers_statuses[hand_label + '_INDEX'] :
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "tristeza"
            return hands_gestures[hand_label]
            break

        elif count[hand_label] == 4 and fingers_statuses[hand_label + '_INDEX'] and fingers_statuses[
            hand_label + '_MIDDLE'] and fingers_statuses[hand_label + '_PINKY'] and fingers_statuses[
            hand_label + '_THUMB']:
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "miedo"
            return hands_gestures[hand_label]
            break

        elif count[hand_label] == 4 and fingers_statuses[hand_label + '_INDEX'] and fingers_statuses[
            hand_label + '_MIDDLE']:
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "dk"
            if hands_gestures['RIGHT'] == "dk" and hands_gestures['LEFT'] == "dk":
                hands_gestures[hand_label] = "incertidumbre"
                return hands_gestures[hand_label]
                break




        # Check if the hands gestures are specified to be written.
        if draw:
            # Write the hand gesture on the output image.
            cv2.putText(output_image, hand_label + ': ' + hands_gestures[hand_label], (10, (hand_index + 1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 5)
           # print(hands_gestures[hand_label])


    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and the gestures of the both hands.
        return output_image, hands_gestures

camera_video = cv2.VideoCapture(2)
camera_video.set(3, 720)
camera_video.set(4, 480)

reads = []
veces = 0

x = ""

timeroff = False


# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

            # Read a frame.
            ok, frame = camera_video.read()

            # Check if frame is not read properly then continue to the next iteration to read the next frame.
            if not ok:
                continue

            # Flip the frame horizontally for natural (selfie-view) visualization.
            frame = cv2.flip(frame, 1)

            if timeroff == False:
                if veces == 0:
                    image_path = r'C:\Users\Cristian\PycharmProjects\pythonProject\Presentacion.png'
                elif veces == 1:
                    image_path = r'C:\Users\Cristian\PycharmProjects\pythonProject\Pregunta1.png'
                elif veces == 2:
                    image_path = r'C:\Users\Cristian\PycharmProjects\pythonProject\pregunta2.png'
                elif veces == 3:
                    image_path = r'C:\Users\Cristian\PycharmProjects\pythonProject\pregunta3.png'
                elif veces == 4:
                    image_path = r'C:\Users\Cristian\PycharmProjects\pythonProject\pregunta4.png'
                elif veces == 5:
                    image_path = r'C:\Users\Cristian\PycharmProjects\pythonProject\procesando.png'

                img = cv2.imread(image_path)

                scale_percent = 50

                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)

                dsize = (width, height)

                # resize image
                output = cv2.resize(img, dsize)
                # cv2.putText(frame, "esperando estar listo", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv2.imshow('Spectrum', output)
                k = cv2.waitKey(1) & 0xFF;
                if (k == 27):
                    break
                pool = ThreadPool(processes=1)
                async_result = pool.apply_async(timer)
                timeroff = async_result.get()
                cv2.destroyWindow("Spectrum")


            #print(timeroff)

            if timeroff == True:
                # Display the frame.
                if len(reads) < 4:
                    cv2.imshow('Spectrum Read', frame)

                    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
                    k = cv2.waitKey(1) & 0xFF;

                    # Check if 'ESC' is pressed and break the loop.
                    if (k == 27):
                        break

                    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)

                if veces == 0:
                    cv2.destroyWindow("Spectrum Read")
                    veces = veces + 1
                    timeroff = False

                elif results.multi_hand_landmarks and veces < 5:
                        output_image, fingers_statuses, count = countFingers(frame, results, draw=True, display=False)
                        #
                        #
                        #recognizeGestures(frame, fingers_statuses, count, draw=True)

                        if recognizeGestures(frame, fingers_statuses, count, draw=True) == "serenidad":

                            reads.append("serenidad")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False
                        elif recognizeGestures(frame, fingers_statuses, count, draw=True) == "alegria":
                            reads.append("alegria")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False
                        elif recognizeGestures(frame, fingers_statuses, count, draw=True) == "aceptacion":
                            reads.append("aceptacion")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False
                        elif recognizeGestures(frame, fingers_statuses, count, draw=True) == "esperanza":
                            reads.append("esperanza")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False
                        elif recognizeGestures(frame, fingers_statuses, count, draw=True) == "melancolia":
                            reads.append("melancolia")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False
                        elif recognizeGestures(frame, fingers_statuses, count, draw=True) == "tristeza":
                            reads.append("tristeza")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False
                        elif recognizeGestures(frame, fingers_statuses, count, draw=True) == "miedo":
                            reads.append("miedo")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False
                        elif recognizeGestures(frame, fingers_statuses, count, draw=True) == "incertidumbre":
                            reads.append("incertidumbre")
                            camera_video = cv2.VideoCapture(2)
                            cv2.destroyWindow("Spectrum Read")
                            veces = veces + 1
                            timeroff = False

                elif len(reads) == 4:

                    for i in reads:
                        if (i == 'melancolia'):
                            cancion = "melancolia.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                        if (i == 'serenidad'):
                            cancion = "serenidad.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                        if (i == 'alegria'):
                            cancion = "felicidad.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                        if (i == 'tristeza'):
                            cancion = "tristeza.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                        if (i == 'aceptacion'):
                            cancion = "aceptacion.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                        if (i == 'esperanza'):
                            cancion = "esperanza.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                        if (i == 'miedo'):
                            cancion = "miedo.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                        if (i == 'incertidumbre'):
                            cancion = "incertidumbre.mp3"
                            t = threading.Thread(target=reproducir, args=(cancion,))
                            t.start()
                    break
                    # image_path = r'C:\Users\Cristian\PycharmProjects\pythonProject\end.png'
                    # img = cv2.imread(image_path)
                    # scale_percent = 50
                    #
                    # width = int(img.shape[1] * scale_percent / 100)
                    # height = int(img.shape[0] * scale_percent / 100)
                    #
                    # dsize = (width, height)
                    # output = cv2.resize(img, dsize)
                    # cv2.imshow('Spectrum Enjoy', output)
                    # k = cv2.waitKey(1) & 0xFF;
                    # if (k == 27):
                    #     break




# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()

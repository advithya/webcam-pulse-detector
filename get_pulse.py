from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
#TODO: work on serial port comms, if anyone asks for it
import socket
import sys
import joblib
import pandas as pd

class getPulseApp(object):
    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, args):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)
        serial = args.serial
        baud = args.baud
        self.send_serial = False
        self.send_udp = False
        if serial:
            self.send_serial = True
            if not baud:
                baud = 9600
            else:
                baud = int(baud)
            self.serial = Serial(port=serial, baudrate=baud)

        udp = args.udp
        if udp:
            self.send_udp = True
            if ":" not in udp:
                ip = udp
                port = 5005
            else:
                ip, port = udp.split(":")
                port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET, # Internet
                                      socket.SOCK_DGRAM) # UDP

        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.w, self.w, = 0, 0
        self.pressed = 0
        # Containerized analysis of received image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits=[60, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv}

        # Load the columns and model for heart attack prediction
        with open('columns.txt', 'r') as f:
            self.columns = [line.strip() for line in f]
        self.heart_attack_model = joblib.load('heart_attack_model.pkl')

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def write_csv(self):
        """
        Writes current data to a csv file
        """
        fna = "Webcam-pulse" + str(datetime.datetime.now())
        fna = fna.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fna + ".csv", data, delimiter=',')
        print("Writing csv with bpm")

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been successfully isolated.
        """
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)
        # Initialize attributes for heart attack prediction
        self.age = None
        self.sex = None
        self.o2_saturation = None
        self.cp = None

    def set_age(self, age):
        self.age = age

    def set_sex(self, sex):
        self.sex = sex

    def set_o2_saturation(self, o2_saturation):
        self.o2_saturation = o2_saturation

    def set_cp(self, cp):
        self.cp = cp

    def predict_heart_attack(self):
        # Collect the necessary features for prediction
        features = {
            'age': self.age,
            'sex': self.sex,
            'o2Saturation': self.o2_saturation,
            'cp': self.cp
        }
        print(features.keys())
        # Example: Populate features with default values if needed
        if 'cp' not in features:
            features['cp'] = 0  # Define a sensible default
        for feature in self.columns:
            if feature not in features:
                print(f"Missing feature: {feature}")
                # Optionally, handle missing keys with defaults or warnings
                features[feature] = 0  # Example default, adjust as needed

        # Ensure the features are in the correct order
        feature_values = [features[feature] for feature in self.columns]
        print(self.columns)
        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        print(all(col in self.columns for col in expected_columns))  # Should be True

        # Predict heart attack
        prediction = self.heart_attack_model.predict([feature_values])
        return prediction[0]

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            if self.send_serial:
                self.serial.close()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _c = frame.shape

        # display unaltered frame
        # imshow("Original",frame)

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cam)
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        imshow("Processed", output_frame)

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        if self.send_serial:
            self.serial.write(str(self.processor.bpm) + "\r\n")

        if self.send_udp:
            self.sock.sendto(str(self.processor.bpm), self.udp)

        # Example: Set the attributes with actual data
        self.set_age(45)  # Replace with actual data
        self.set_sex(1)   # Replace with actual data
        self.set_o2_saturation(98.6)  # Replace with actual data
        self.set_cp(3)  # Replace with actual data

        # Predict heart attack
        heart_attack_risk = self.predict_heart_attack()
        if heart_attack_risk == 1:
            print("Warning: High risk of heart attack detected!")

        # handle any key presses
        self.key_handler()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('--serial', default=None,
                        help='serial port destination for bpm data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='udp address:port destination for bpm data')

    args = parser.parse_args()
    App = getPulseApp(args)
    while True:
        App.main_loop()
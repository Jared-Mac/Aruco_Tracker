"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import math



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
#------------ CREATE ARUCO BOARD OBJECT
# length from the generated markers
# TODO maker a configuration file
aruco_marker_length_meters = 20

# creating an aruco board

# board_face = np.array(
#     [[-0.25, 0.31, -0.25], [-0.25, 0.31, 0.25], [0.25, 0.31, 0.25], [0.25, 0.31, -0.25]], dtype=np.float32)

face_1 = np.array(
    [[1.225, 10.191, 14.668], [11.947, 10.191, 8.588], [7.121, 2.72, 0.063], [-3.616, 2.714, 6.141]], dtype=np.float32)
face_2 = np.array(
    [[12.089, 10.207, -8.378], [1.464, 10.207, -14.631], [-3.504, 2.72, -6.19], [7.121, 2.72, 0.063]], dtype=np.float32)
face_3 = np.array(
    [[-13.296, 10.193, -6.282], [-13.416, 10.193, 6.046], [-3.616, 2.714, 6.141], [-3.504, 2.72, -6.19]], dtype=np.float32)


face_4 = np.array([[13.97, 20.658, 6.291], [14.071, 20.658, -6.039],
                  [11.752, 8.548, -6.057], [11.651, 8.548, 6.272]], dtype=np.float32)

face_5 = np.array([[-1.534, 20.654, -15.23], [-12.266, 20.654, -9.161],
                   [-11.127, 8.544, -7.147], [-0.395, 8.544, -13.215]], dtype=np.float32)

face_6 = np.array([[-12.415, 20.646, 8.948], [-1.791, 20.646, 15.205],
                   [-0.619, 8.534, 13.215], [-11.243, 8.534, 6.958]], dtype=np.float32)


face_7 = np.array([[0.412, 24.029, 13.217], [11.137, 24.029, 7.153], [
    12.275, 11.928, 9.166], [1.551, 11.928, 15.23]], dtype=np.float32)
face_8 = np.array([[11.256, 24.042, -6.947], [0.627, 24.042, -13.207],
                   [1.80, 11.925, -15.198], [12.429, 11.925, -8.938]], dtype=np.float32)
face_9 = np.array([[-11.644, 24.028, -6.27], [-11.744, 24.028, 6.064],
                   [-14.064, 11.914, 6.045], [-13.963, 11.914, -6.289]], dtype=np.float32)


# face_10 = np.array([[3.505, 29.861, 6.185], [3.624, 29.861, -6.135],
#                     [13.419, 22.385, -6.04], [13.299, 22.385, 6.281]], dtype=np.float32)
# face_11 = np.array([[3.626, 29.859, -6.139], [-7.103, 29.859, -0.055],
#                     [-11.94, 22.38, -8.586], [-1.211, 22.38, -14.67]], dtype=np.float32)
# face_12 = np.array([[-7.103, 29.859, -0.055], [3.505, 29.861, 6.185],
#                     [-1.456, 22.369, 14.638], [-12.08, 22.369, 8.385]], dtype=np.float32)

origin_board_face = np.array([[-9.5, 9.5, .0], [9.5, 9.5, .0],
                        [9.5, -9.5, .0], [-9.5, -9.5, .0]], dtype=np.float32)
board_faces = [face_1, face_2, face_3, face_4, face_5,
               face_6, face_7, face_8, face_9 ]


board_ids = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [
                     9]], dtype=np.int32)
board = aruco.Board_create(board_faces,
                           aruco.getPredefinedDictionary(
                               aruco.DICT_ARUCO_ORIGINAL),
                           board_ids)
origin_board = aruco.Board_create([origin_board_face], aruco.getPredefinedDictionary(
                    aruco.DICT_ARUCO_ORIGINAL),
                    np.array([[10]], dtype=np.int32))
angles = [0,0,0]
# File storage in OpenCV
cv_file = cv2.FileStorage("calib_images/cu81.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

print("camera_matrix : ", mtx.tolist())
print("dist_matrix : ", dist.tolist())


rvec = np.empty((3, 1))
tvec = np.empty((3, 1))


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=3,
              text_color=(0, 0, 255),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h+10), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

    return text_size





###------------------ ARUCO TRACKER ---------------------------
first = True
testvec = None
reference_rvec = None
reference_tvec = None
camera_rvec = None 
camera_R = None
camera_tvec = None
transformation = None

euler_angles = None
rvec = np.zeros((3, 1))
tvec = np.zeros((3, 1)) 
R = np.zeros((3,3))
output_rvec = np.zeros((3, 1))
output_tvec = np.zeros((3, 1))

while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # identify markers and
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)


    corners, ids, rejectedImgPoints, recoveredIdxs = aruco.refineDetectedMarkers(
        gray, board, corners, ids, rejectedImgPoints, mtx, dist)
    # frame = aruco.drawDetectedMarkers(frame, corners)


    if(ids is not None):
        if 10 in ids:
            # Track Pose of Origin Board
            retval, reference_rvec, reference_tvec = aruco.estimatePoseBoard(
                corners, ids, origin_board, mtx, dist, reference_rvec, reference_tvec, False)

            frame = aruco.drawAxis(
                frame, mtx, dist, reference_rvec, reference_tvec, aruco_marker_length_meters)

            # Calcuates camera R matrix and translation
            R = cv2.Rodrigues(reference_rvec)[0]

            transformation = np.hstack((R, reference_tvec))


            transformation = np.vstack((transformation,np.array([0,0,0,1])))
            transformation = np.linalg.inv(transformation)

        if transformation is not None:
            # Track Pose of dodecahedron
            retval, rvec, tvec = aruco.estimatePoseBoard(
                corners, ids, board, mtx, dist, rvec, tvec, False)
            if retval > 0:
                tvec = np.reshape(tvec, (1,3))
                # print(tvec)
                output_tvec = np.hstack((tvec, np.array([[1]])))
                output_tvec.flatten()
                output_tvec = transformation @ output_tvec.reshape(4,1)
                output_tvec.flatten()
                output_tvec = np.delete(output_tvec, 3, axis=0)



                output_rvec = rvec
                angle = np.linalg.norm(output_rvec)
                output_rvec = output_rvec / angle

                output_rvec.flatten()
                output_rvec = R @ output_rvec
                output_rvec = angle * output_rvec
                T_R = cv2.Rodrigues(rvec)[0]
                print(f"R={R}")
                print(f"T_R={T_R}")
                tracked_R = R @ T_R
                print(f"tracked_R = {tracked_R}")
                tracked_transformation = np.hstack((tracked_R, tvec.T))

                euler_angles = cv2.decomposeProjectionMatrix(
                    tracked_transformation)[6]

                output_rvec.flatten()
                frame = aruco.drawAxis(
                    frame, mtx, dist, rvec, tvec, aruco_marker_length_meters)

        frame = aruco.drawDetectedMarkers(frame, corners, ids)
            


    
    draw_text(frame, "origin tvec: " + str(reference_tvec), pos=(25, 50))
    draw_text(frame, "transformed tvec: " + str(output_tvec), pos=(25, 150))
    draw_text(frame, "origin rvec: " + str(rvec), pos=(25, 250))
    draw_text(frame, "transformed rvec: " +
              str(euler_angles), pos=(25, 350))

    frame = cv2.resize(frame, (1920, 1080))

    cv2.imshow('frame', frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        cv_file.release()
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

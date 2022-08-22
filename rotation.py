import numpy as np 



def rotate_x(array,angle):
    array = np.transpose(array)
    rotation_matrix = np.array([[1.0, 0., 0.],
                       [0., np.cos(angle), -1*np.sin(angle)],
                       [0., np.sin(angle), np.cos(angle)]],dtype=np.float32)
    return np.transpose(np.matmul(rotation_matrix, array))


def rotate_y(array, angle):
    array = np.transpose(array)
    rotation_matrix = np.array([[np.cos(angle), 0., np.sin(angle)],
                       [0., 1, 0],
                        [-1*np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
    return np.transpose(np.matmul(rotation_matrix, array))


def rotate_z(array, angle):
    array = np.transpose(array)
    rotation_matrix = np.array([[np.cos(angle), -1*np.sin(angle), 0.],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]],dtype=np.float32)
    return np.transpose(np.matmul(rotation_matrix, array))


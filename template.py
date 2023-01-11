import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    train_path_list =  os.listdir(root_path)
    train_names = train_path_list
    return train_names

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    train_image_list = []
    image_classes_list = []
    for index, class_path in enumerate(train_names):
    #Dataset/Train/aeroplane
        image_path_list = os.listdir(root_path + '/' + class_path)
        for image_path in image_path_list:
            train_image_list.append(root_path + '/' + class_path + '/' + image_path)
            image_classes_list.append(index)
    return train_image_list, image_classes_list

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    face_list = list()
    class_list = list()
    
    for idx, image in enumerate(image_list) : 
        img =  cv2.imread(image, 0)
        faces = face_cascade.detectMultiScale(img, 1.2, 5)
        
        if len (faces) < 1 : continue
        
        for face in faces :
            x,y,h,w = face
            face_img = img[y:y+h, x:x+w]
            face_list.append(face_img)
            class_list.append(idx)
    
    return face_list

def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))
    return face_recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    test_path_list = os.listdir(test_root_path)
    test_image_list = []
    
    for path in test_path_list:
        gray = cv2.imread(f"{test_root_path}/{path}", 0)
        test_image_list.append(gray)
        
    return test_image_list
    
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    prediction_result_list = []
    for face in test_faces_gray:
        prediction_result = recognizer.predict(face)
        prediction_result_list.append(prediction_result)
        
    return prediction_result_list
        
def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    for index, test_image in enumerate(test_image_list):
        for (x,y,w,h) in test_faces_rects[index]:
            cv2.rectangle(test_image, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(test_image, train_names[predict_results[index][0]], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if predict_results[index][1] > 100:
                cv2.putText(test_image, "Unknown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return test_image_list
    

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

def test() : 
    get_test_images_data("dataset/test")

if __name__ == "__main__":

    # test()
    
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train" #"[PATH_TO_TRAIN_ROOT_DIRECTORY]"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test" #"[PATH_TO_TEST_ROOT_DIRECTORY]
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list)
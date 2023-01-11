import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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
    train_path_list = os.listdir(root_path)
    return train_path_list  

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
    # print(f"DEF : get_class_id\n============")
    # print(f"Root Path : {root_path}")
    # print(f"Train Names : {train_names}")
    # print(f"=============")
    
    train_image_list = []
    class_id_list = []
    
    # Read Image
    for idx, folder in enumerate(train_names):
        # print(f"Folder : {folder}")
        train_folder_path_list = f"{root_path}{folder}"
        # print(f"Train Folder Path List : {train_folder_path_list}")
        
        train_folder_images_list = os.listdir(train_folder_path_list)
        for image_name in train_folder_images_list :
            # print(f"Image name : {image_name}")
            img = cv2.imread(f"{train_folder_path_list}/{image_name}")
            class_id = idx
            train_image_list.append(img)
            class_id_list.append(class_id)
            
    # print(f"Train Image List Length: {len(train_image_list)}")
    # print(f"Class ID List Length: {len(class_id_list)}")
    return train_image_list, class_id_list

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
    # print(f"DEF : detect_faces_and_filter\n============")
    
    try :
        face_cascade = cv2.CascadeClassifier('haarcascades\haarcascades\haarcascade_frontalface_default.xml')
        # print("Haarcasade detected.")
    except :
        print("Haarcasade not detected.")
    
    face_img_list = []
    face_img_class_id_list = []
    rect_image_list = []
    
    if image_classes_list is None :
        for test_img in image_list:
            faces = face_cascade.detectMultiScale(test_img, 1.3, 5)
            
            if len(faces) < 1 : continue
            
            for face in faces :
                x,y,w,h = face
                rect_image = cv2.rectangle(test_img, (x,y), (x+w, y+h), (0,255,0), 2)
                rect_image_list.append(rect_image)
                
                test_face_img = test_img[y:y+h, x:x+w] # Cropped Image
                face_img_list.append(test_face_img)
            
        return face_img_list, rect_image_list, None
        
    for train_img, train_img_class in zip(image_list, image_classes_list):
        faces = face_cascade.detectMultiScale(train_img, 1.3, 5)
        
        if len(faces) < 1 : continue
        
        for face in faces :
            x,y,w,h = face
            rect_image = cv2.rectangle(train_img, (x,y), (x+w, y+h), (0,255,0), 2)
            rect_image_list.append(rect_image)
            
            train_face_img = train_img[y:y+h, x:x+w] # Cropped Image
            train_face_img = cv2.cvtColor(train_face_img, cv2.COLOR_BGR2GRAY)
            
            face_img_class_id_list.append(train_img_class)
            face_img_list.append(train_face_img)
        
        
    return face_img_list, rect_image_list, face_img_class_id_list

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
    # print(f"DEF : train\n============")
    # print(f"class list : {len(image_classes_list)}")
    # # print(f"train face grays : {len(train_face_grays)}")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))
    
    return recognizer

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
    test_filename_list = os.listdir(test_root_path)
    test_img_list = []
    
    for test_filename in test_filename_list : 
        test_img = cv2.imread(f"{test_root_path}/{test_filename}")
        test_img_list.append(test_img)
        
    return test_img_list

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
    result_list = []
    
    for img in test_faces_gray : 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        result, _ = recognizer.predict(img)
        result_list.append(result)
    
    return result_list
        
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
    
    
    # print(predict_results)
    
    predict_result_name_list = []
    result_img_list = []
    
    # print(test_image_list)
    for result in predict_results :
        # print(f"result : {train_names[result]}")
        predict_result_name_list.append(train_names[result])

    for idx, img in enumerate(test_faces_rects) :
        cv2.putText(img, f"{predict_result_name_list[idx]}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        result_img_list.append(img)

    return result_img_list
def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    
    resized_img = []
    for img in image_list :
        tmp = cv2.resize(img, (350, 350), cv2.INTER_AREA)
        resized_img.append(tmp)
    
    res = np.hstack((resized_img[0], resized_img[1], resized_img[2], resized_img[3], resized_img[4]))
    cv2.imshow("Final Result", res)
    cv2.waitKey(0)
    
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
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "./dataset/train/"
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
    test_root_path = "./dataset/test/"
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
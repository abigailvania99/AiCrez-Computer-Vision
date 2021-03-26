import os
import cv2
import numpy as np

def get_path_list(root_path):

    list_name = os.listdir(root_path)
    return list_name
    
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of each person
    '''

def get_class_id(root_path, train_names):

    img_list=[]
    img_class_id=[]
    
    for idx, name in enumerate(train_names):
        
        path = root_path+'\\'+name

        for img_name in os.listdir(path):
            img_path = path + '\\' + img_name
            img = cv2.imread(img_path)
            img_list.append(img)
            img_class_id.append(idx)

    return img_list, img_class_id
            
        
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

def detect_train_faces_and_filter(image_list, image_classes_list):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    train_face_img = []
    train_face_img_id = []

    print('Please wait...')

    for i, img_class_id in enumerate(image_classes_list):
        
        img = image_list[i]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

        if(len(detected_faces)<1):
            continue
        
        else:
            for face in detected_faces:
                x,y,h,w = face
                crop_face = img_gray[y:y+h, x:x+w]
                train_face_img.append(crop_face)
                train_face_img_id.append(img_class_id)

    #print('done')

    return train_face_img, train_face_img_id

    
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered image classes id
    '''

def detect_test_faces_and_filter(image_list):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    test_face_img = []
    test_face_rectangle = []

    for img_test in image_list:
        img_test_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)    
        detected_faces = face_cascade.detectMultiScale(img_test_gray, scaleFactor=1.2, minNeighbors=5)

        if(len(detected_faces)<1):
            
            continue
        
        else:
            
            for face in detected_faces:
                x,y,h,w = face
                rect = [x,y,h,w]
                test_face_rectangle.append(rect)
                crop_face = img_test_gray[y:y+h, x:x+w]
                test_face_img.append(crop_face)

    return test_face_img, test_face_rectangle
        

    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
    '''

def train(train_face_grays, image_classes_list):

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer
    
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

def get_test_images_data(test_root_path):

    img_test_list = []

    for img_test_name in os.listdir(test_root_path):
        img_test_path = test_root_path + '\\' + img_test_name
        img_test = cv2.imread(img_test_path)
        img_test_list.append(img_test)

    return img_test_list

    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image in the test directories
    '''
    
def predict(recognizer, test_faces_gray):

    prediction_result_id_list = []
    
    for img_predict in test_faces_gray:
        res_id, confidence = recognizer.predict(img_predict)
        #confidence = math.floor(confidence*100)/100
        #text = train_names[res_id] + ' ' + str(confidence)+'%'
        #print(text)
        prediction_result_id_list.append(res_id)
        
    return prediction_result_id_list
        
    
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        test_faces_gray : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, size):

    final_image_list = []

    for i, img_prediction in enumerate(test_image_list):
        
        text = train_names[predict_results[i]]
        x, y, w, h = test_faces_rects[i][0], test_faces_rects[i][1], test_faces_rects[i][2], test_faces_rects[i][3]

        cv2.rectangle(img_prediction, (x,y), (x+w, y+h), (150,140,42), 3)
        final_img = cv2.resize(img_prediction, (size,size), 100, 100)
        cv2.putText(final_img, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255))
        final_image_list.append(final_img)

    return final_image_list

    '''
        To draw prediction results on the given test images and resize the image

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
        size : number
            Final size of each test image

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    

def combine_and_show_result(image_list, size):
    
    #res = np.hstack(image_list)
    #cv2.imshow('result', res)
    #cv2.waitKey(0)
      
    target_combine_img_list = []

    for imggg in image_list:
        final_img_rezise_again = cv2.resize(imggg, (size, size), 100, 100)
        target_combine_img_list.append(final_img_rezise_again)

    b = np.hstack(target_combine_img_list)

    #cv2.imshow('result after resize resize', b)
    cv2.imshow('result', b)
    cv2.waitKey(0)
    

    
    
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
        size : number
            Final size of each test image
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

if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset\\train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset\\test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, 200)
    
    combine_and_show_result(predicted_test_image_list, 200)

# SET THESE VARIABLES AND YOU'RE GOOD TO GO!

TF_MODEL = "Acne.h5"
ONNX_MODEL = "Acne.onnx"
BREAK_ON_FIRST_FAIL = False
# The Directory has to have sub-directories inside NOT IMAGES even if testing one image
TEST_DIRECTORY = "test_dataset"






import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np


class KerasModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        model =  keras.models.load_model(self.model_path)
        self.loaded_model =  model

class KerasModelRunner:
    def __init__(self, model):
        self.model = model.loaded_model


    def get_img(self, img_path):
        
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224), cv2.INTER_LINEAR)/225.0
        img = img.reshape(-1, 224, 224, 3)
        return img

    def predict(self, img_path):
        img = self.get_img(img_path)
        prob = self.model.predict(img)
        return prob
        # predIdx = np.argmax(prob, axis=1) 
        # return predIdx


keras_model = KerasModelLoader(TF_MODEL)
keras_model.loaded_model
tf_running_model = KerasModelRunner(keras_model)




import onnxruntime as ort

class OnnxModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        # providers could be changed to CUDA if the server had a dedicated GPU
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(self.model_path, providers=providers)
        self.inference_session =  session

class OnnxModelRunner:
    def __init__(self, model):
        self.inference_session = model.inference_session
        self.input_name = self.inference_session.get_inputs()[0].name
        self.label_name = self.inference_session.get_outputs()[0].name
        

    def get_img(self, img_path):
        
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224), cv2.INTER_LINEAR)/225.0
        img = img.reshape(-1, 224, 224, 3)
        return img

    def predict(self, img_path):
        img = self.get_img(img_path)
        pred = self.inference_session.run([self.label_name], {self.input_name:img.astype(np.float32)})[0]
        return pred
        # return np.argmax(pred)

onnx_model = OnnxModelLoader(ONNX_MODEL)
onnx_running_model = OnnxModelRunner(onnx_model)






import os
def image_path_generator(outter_dir):
    inner_dirs = list()
    # iterate over internal dirs and files and append only dirs
    for (root, dirs, files) in os.walk(outter_dir):
        inner_dirs.append(root)
    # remove the root directory containing sub-dirs
    inner_dirs.pop(0)
    for curr_dir in inner_dirs:
        # concatinating the outter dir to each inner dir
        # curr_dir = '/'.join([outter_dir, curr_dir])
        # list images inside each dir
        images = os.listdir(curr_dir)
        for img in images:
            # concatinating the outter and inner dir to each image
            img = f"{curr_dir}\\{img}"
            yield img



import unittest 
class VerifyOnnx(unittest.TestCase):

    def test_prediction(self):
        """ TESTING TENSORFLOW CONVERSION INTO ONNX """
        for img in image_path_generator(TEST_DIRECTORY):
            curr_dir = img[:img.rfind('\\')]
            with self.subTest(f"Testing The {curr_dir} Category"):
                # self.assertAlmostEquals(onnx_running_model.predict(img)[0],  tf_running_model.predict(img)[0])
                onnx_result = onnx_running_model.predict(img)
                tf_result = tf_running_model.predict(img)
                # tolerance in the difference between array elements
                # 1 to the power of negative 5
                decimals = 1e-5
                error_msg = f"{''.join(['*'])*50} \nThe Onnx and Tensorflow Models have conflicting output testing {curr_dir} with ImagePath = {img}"
                #assert_allclost returns None if there no assertion errors
                assertion = np.testing.assert_allclose(tf_result, onnx_result, atol=decimals, err_msg=error_msg , verbose=True, equal_nan=True)
                self.assertEqual(assertion, None)
                self.failIf(assertion and BREAK_ON_FIRST_FAIL)




if __name__ == "__main__":
    unittest.main()
        







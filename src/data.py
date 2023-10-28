
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Data:
    def __init__(self, path):
        self.path = path
        self.input_path=self.path+'/'+'RGB'
        self.output_path=self.path+'/'+'Depth'

    def load_dataset_filepaths(self):
        # to get all paths of 940 images
        input_files = [f for f in os.listdir(self.input_path) if os.path.isfile(os.path.join(self.input_path, f))]
        output_files = [f for f in os.listdir(self.output_path) if os.path.isfile(os.path.join(self.output_path, f))]
        return input_files, output_files

    def show_Img(self,filename):
        #Image show
        file_path=os.path.join(self.path,filename)
        print(file_path)
        img=cv2.imread(file_path)
        print(img)
        if img is not None:
                cv2.imshow('Image',img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
                print("Image not found !")


    def load_image(self,filename):
        #load a particular image and show it 
        if("rgb" in filename):
            filename=os.path.join("RGB",filename)
            self.show_Img(filename)

        elif("depth" in filename):
            filename=os.path.join("Depth",filename)
            self.show_Img(filename)    
            

    def preprocessing(self):
        pass


    def visualize_dataset(self):
        pass




if __name__ == '__main__':
    data_handler = Data("/home/vyas/CVIP/project/Dataset")
    data_handler.load_image('dining_room_0001b_rgb_00003.jpg')

import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2  

class Data:
    def __init__(self, path):
        self.path = path
        self.input_path=self.path+'/'+'RGB'
        self.output_path=self.path+'/'+'Depth'
        #scenes raw 
        self.scenes = ['bedroom_0021', 'kitchen_0003', 'living_room_0005', 'office_0011']
        #scenes processed --> bedroom,kitchen,living,office
        self.scenesp=["bedroom","kitchen","living","office"]

    
    def count_images(self):
         scenes=self.scenesp
         rgb_dict={scenes[0]:0,scenes[1]:0,scenes[2]:0,scenes[3]:0}
         depth_dict={scenes[0]:0,scenes[1]:0,scenes[2]:0,scenes[3]:0}
         input_path=self.input_path
         output_path=self.output_path
         for rgb in os.listdir(input_path):
              if scenes[0] in rgb:
                   rgb_dict[scenes[0]]+=1
              elif scenes[1] in rgb:
                   rgb_dict[scenes[1]]+=1
              elif scenes[2] in rgb:
                   rgb_dict[scenes[2]]+=1
              elif scenes[3] in rgb:
                   rgb_dict[scenes[3]]+=1
                   
         for d in os.listdir(output_path):
              if scenes[0] in d:
                   depth_dict[scenes[0]]+=1
              elif scenes[1] in d:
                   depth_dict[scenes[1]]+=1
              elif scenes[2] in d:
                   depth_dict[scenes[2]]+=1
              elif scenes[3] in d:
                   depth_dict[scenes[3]]+=1
         return rgb_dict,depth_dict
         

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
        rgb_dict,depth_dict=self.count_images()
        fig_rgb,ax_rgb=plt.subplots()
        bar_labels = rgb_dict.keys()
        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        ax_rgb.bar(rgb_dict.keys(), rgb_dict.values(), label=bar_labels, color=bar_colors)

        ax_rgb.set_ylabel('Scenes')
        ax_rgb.set_title('No. of RGB Images')
        ax_rgb.legend(title='RGB Image count')

        fig_d,ax_d=plt.subplots()
        bar_labels = rgb_dict.keys()
        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        ax_d.bar(rgb_dict.keys(), rgb_dict.values(), label=bar_labels, color=bar_colors)

        ax_d.set_ylabel('Scenes')
        ax_d.set_title('No. of  Depth Images')
        ax_d.legend(title='Depth Image count')
        plt.show()






if __name__ == '__main__':
    data_handler = Data("/home/vyas/CVIP/project/Dataset")
    # data_handler.load_image('dining_room_0001b_rgb_00003.jpg')
    data_handler.visualize_dataset()
import tensorflow as tf
import numpy as np
import cv2
import os

from classifiers import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tf_explain.core.grad_cam import GradCAM
import efficientnet.tfkeras as effnet

datasets = ['FS']

if __name__ == "__main__":
    for i in range(1, 7):
        model_path = f"../model/MesoNet{i}.h5"
        for dataset in datasets:
            IMAGE_PATH = f"../dataset/{dataset}/test/fake"
            OUTPUT_PATH = f"../heatset/MesoNet/MesoNet{i}/{dataset}"
            print(i, model_path, IMAGE_PATH, OUTPUT_PATH)

            model = Meso4()
            model.load(model_path)
            model=model.model
            model.summary()

            # get the last conv layer name
            last_conv_layer_name = ""
            for layer in model.layers[::-1]:
                if 'conv' in layer.name:
                    last_conv_layer_name = layer.name
                    break
            print(f"Last conv layer name: {last_conv_layer_name}")

            for imagename in os.listdir(IMAGE_PATH):
                try:
                    img=cv2.imread(IMAGE_PATH+'/'+imagename)
                    img=cv2.resize(img,(256,256))
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img=np.array(img).astype(np.float32)/255.0
                    img=np.expand_dims(img,axis = 0)

                    data=(img,None)
                    explainer = GradCAM()
                    try:
                        heatmap = explainer.explain(data, model, class_index=0, layer_name=last_conv_layer_name, use_guided_grads=False)
                        heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        output = heatmap

                        cv2.imwrite(os.path.join(OUTPUT_PATH, f"heatmap_{imagename}"), output)
                    except ValueError as e:
                        print(f"Error processing {imagename}: {e}")
                        continue
       
                except Exception as e:
                    print(str(e))
                    continue

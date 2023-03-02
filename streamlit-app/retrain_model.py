#pip install opencv-python
import sys
import os
from datetime import date
os.sys.path
import cv2
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#Tensorflow keras for CNN model
#pip install tensorflow


from tensorflow.keras.preprocessing.image import ImageDataGenerator


folder_dir = "Retraining_data"
model_path = 'model'

label=[]
data=[]

SIZE = 128

for folder in os.listdir(folder_dir):
    for file in os.listdir(os.path.join(folder_dir,folder)):
        if file.endswith("jpg"):
            label.append(folder)
            img = cv2.imread(os.path.join(folder_dir,folder,file))
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converts
            im = cv2.resize(img_rgb,(SIZE,SIZE))
            data.append(im)
        else:
            continue

data_array = np.array(data)
label_array = np.array(label)


encoder =LabelEncoder()
y = encoder.fit_transform(label)

#print(y)
y =  to_categorical(y,5) # 5 flower type -converts all 5 into categorical
#print(y)
#x = data_arr/255
#x =[]
#for im in data:
 #   x.append(cv2.normalize(im, None, 0, 255,cv2.NORM_MINMAX, dtype=cv2.CV_32F))
x = data_array/255

#training set creation
X_train, X_test, Y_train, Y_test =  train_test_split(x, y, test_size=0.20, random_state=10)

datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range = 0.20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True)


datagen.fit(X_train)

#load model
model = load_model(os.path.join(model_path, "model.h5"))

batch_size=200
epochs=10
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs,
                              validation_data = (X_test,Y_test),
                              verbose = 1)
scores = model.evaluate(X_test, Y_test, verbose=0)


# backup old model and replace with new
file_timestamp = str(date.today().strftime("%Y_%m_%d"))
Model_name=r"model\model.h5"
backup_name=r"model\model_backup.h5"
os.rename(Model_name, backup_name)


#save model

model.save(Model_name)
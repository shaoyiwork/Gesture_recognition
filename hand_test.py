'''Test a gesture model trained by CNN on the given dataset.

'''
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model


model = Sequential()
model=load_model("hand_model.h5")
model.summary()

img1 = image.load_img(path="./hand_data/test/five_197.png", target_size=(234,234,3))
img = image.img_to_array(img1)
img = img.astype('float32')
img /= 255
test_img = img.reshape((1,234,234,3))
preds = model.predict(test_img)
print (preds[0])
#################################
img_class = model.predict_classes(test_img)
classname = img_class[0]
print("class : %d" %(classname))
if classname == 0:
    title = 'Rock&Fist'
    print ("Rock&Fist")
if classname == 1:
    title = 'Scissors'
    print ("Scissors") 
if classname == 2:
    title = 'Paper'
    print ("Paper")
###############################
plt.imshow(img1)
plt.title(title)
plt.show()

from keras.models import Sequential
from keras.layers import Dense , GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16 as vgg

model = Sequential()
vgg16 = vgg( include_top=False , weights="imagenet" ,\
 input_tensor=None , input_shape=( 224 , 224 , 3 ) )

model.add( vgg16 )
model.add( GlobalAveragePooling2D() )
model.add( Flatten() )
model.add( Dense( 256 , activation="relu" ) )
model.add( Dense( 6 , activation="softmax" ) )
print( model.summary() )

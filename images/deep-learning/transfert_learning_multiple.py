import numpy as np
import os
import PIL
import PIL.Image
import pickle
import tikzplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from joblib import Parallel, delayed
import pathlib
from transfert_learning import plot_perf

def plot_perf(history,name_f,save):
  fig,ax = plt.subplots(1,2,figsize=(15,4))
  ax[0].plot(history.history['accuracy'])
  ax[0].plot(history.history['val_accuracy'])
  ax[0].set_title('model accuracy')
  ax[0].set_ylabel('accuracy')
  ax[0].set_xlabel('epoch')
  ax[0].legend(['train', 'val'], loc='upper left')

  ax[1].plot(history.history['loss'])
  ax[1].plot(history.history['val_loss'])
  ax[1].set_title('model loss')
  ax[1].set_ylabel('loss')
  ax[1].set_xlabel('epoch')
  ax[1].legend(['train', 'val'], loc='upper right')
  fig.suptitle(name_f)
  fig.savefig(f'{save}{name_f}.png') 
  tikzplotlib.save(f'{save}{name_f}.tex') 
  
def create_bottom(x,inputs,NombreDeClasses):
    outputs = keras.layers.Dense(NombreDeClasses, activation="softmax")(x)
    #outputs = keras.layers.Dense(NombreDeClasses, input_shape=(2048,))(x)
    model = keras.Model(inputs, outputs)
    return model

def create_model(name_model,img_height, img_width,NombreDeClasses):
    model = getattr(keras.applications, name_model)
    base_model = model(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(img_height, img_width, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    # Create new model on top
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    if name_model=="Xception":
        norm_layer = keras.layers.experimental.preprocessing.Normalization()
        # Scale inputs to [-1, +1]
        x = norm_layer(x)
    else :
        scale_l = keras.layers.Rescaling(1./255)
        x = scale_l(x)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    model = create_bottom(x,inputs,NombreDeClasses)

    return model,base_model

def inference_mod(base_model,model,train_ds,validation_ds,epochs):
  base_model.trainable = False
  model.compile(
      optimizer=keras.optimizers.Adam(),
      #loss=keras.losses.CategoricalCrossentropy(from_logits=False),
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
      #metrics=[tf.keras.metrics.Accuracy()],
      metrics=['accuracy'])
  model.fit(train_ds, epochs=epochs[0], validation_data=validation_ds)
  return model,base_model

def tuning_mod(base_model,model,train_ds,validation_ds,epochs):
  base_model.trainable = True
  model.compile(
      optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
      #loss=keras.losses.CategoricalCrossentropy(from_logits=False),
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
      #metrics=[tf.keras.metrics.Accuracy()],
      metrics=['accuracy'])
  history = model.fit(train_ds, epochs=epochs[1], validation_data=validation_ds)
  return history,model


def main_train(list_model,train_ds,validation_ds,epochs,img_height, img_width,NombreDeClasses,save,name_f):
    input_m=[]
    commonInput = keras.Input(shape=(img_height, img_width, 3))
    for name_model in list_model:
        print(f"----------{name_model}-----------")
        model,base_model = create_model(name_model,img_height, img_width,NombreDeClasses)
        model,base_model = inference_mod(base_model,model,train_ds,validation_ds,epochs)
        history,model = tuning_mod(base_model,model,train_ds,validation_ds,epochs)
        if save!="":
            plot_perf(history,name_model+"_"+name_f,save)
        input_m.append(model(commonInput))

    if len(list_model)>1:
        mergedOut = keras.layers.Concatenate()(input_m)
        temp_mod = keras.Model(commonInput, mergedOut)
        temp_mod.trainable = False
        x = temp_mod(commonInput, training=False)
        #plot_model(temp_mod, to_file='./results/model/fusion_model.png',show_shapes=True, show_layer_names=True)
        outputs = keras.layers.Dense(NombreDeClasses, activation="softmax")(x)
        newModel = keras.Model(commonInput, outputs)
        #plot_model(newModel, to_file='./results/model/model_plot.png',show_shapes=True, show_layer_names=True)
        newModel.summary()
        newModel.compile(
            optimizer=keras.optimizers.Adam(2e-3),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])
        history_global = newModel.fit(train_ds, epochs=epochs[1], validation_data=validation_ds)
        if save!="":
            plot_perf(history_global,"global_"+name_f,save)
    return 1

def load_data_transf(Data_Dir,img_height, img_width,batch_size,
                     ValidationProportion,RandomSeed,GPU):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus!=[] ang GPU:
        tf.config.set_visible_devices([], 'CPU') # hide the CPU
        tf.config.set_visible_devices(gpus[0], 'GPU') # unhide potentially hidden GPU
        tf.config.get_visible_devices()

    print('--------Tensorflow version---------------')
    print(tf.__version__)

    alldata_ds = tf.keras.preprocessing.image_dataset_from_directory(Data_Dir) # type=BatchDataset
    class_names = alldata_ds.class_names
    print('------------------------------------------------')
    print('Identification des classes')
    print(class_names)

    # 80% training
    print('------------------------------------------------')
    print('-------------- Traning dataset -----------------')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        Data_Dir, validation_split=ValidationProportion,
        subset="training", seed=RandomSeed,shuffle=True,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print('------------------------------------------------')
    # 20% validation 
    print('-------------- Validation dataset -------------')
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        Data_Dir, validation_split=ValidationProportion,
        subset="validation", seed=RandomSeed,shuffle=True,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print('------------------------------------------------')

    class_names = train_ds.class_names
    print('Identification des classes')
    print(class_names)
    print('------------------------------------------------')

    NombreDeClasses = len(class_names)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds,validation_ds,NombreDeClasses


def submain(list_model,epochs,img_height, img_width,batch_size,
            ValidationProportion,RandomSeed,save,dir_data,name_f,GPU):

    Data_Dir = pathlib.Path(f"{dir_data}{name_f}/")
    train_ds,validation_ds,NombreDeClasses = load_data_transf(Data_Dir,img_height, img_width,
                                                              batch_size,ValidationProportion,RandomSeed,GPU)
    main_train(list_model,train_ds,validation_ds,epochs,img_height, img_width,NombreDeClasses,save,name_f)
    return 2


if __name__=="__main__":
    list_model = ["MobileNetV2","DenseNet121","VGG16","ResNet50","EfficientNetB0"]    
    dir_data"./datatest/crop/"
    name_f = "ETE"
    img_height = 32
    img_width = 32
    epochs = [2,2]
    batch_size = 32
    ValidationProportion = 0.35
    RandomSeed = 12
    save = "./results/deep/"
    GPU = False
    submain(list_model,epochs,img_height,img_width,batch_size,ValidationProportion,RandomSeed,save,dir_data,name_f,GPU)
                    

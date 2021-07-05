#%%
import itertools


import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from opt_einsum.backends import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

from tensorflow.python.keras.utils.vis_utils import plot_model, model_to_dot



test_single_image=False
train=False
is_train_generator=False
display=True



# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()


# Define data generators
# train_dir = 'data/train'
# val_dir = 'data/test'





num_train = 28714
num_val = 7178
batch_size = 256
num_epoch = 200
input_target=48

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)


if train==False:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_target, input_target),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(input_target, input_target),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
        )
elif train==True:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_target, input_target),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True)

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(input_target, input_target),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True
    )



# Create the model

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu',  input_shape=(input_target,input_target,1)))
# model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))


model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.summary()


#
if train==True and test_single_image==False:
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    plot_model_history(model_info)

    model.save_weights('model.h5')
if train==False and test_single_image==False:
    model.load_weights('model.h5')










    #%%
    target_names = []

    for key in train_generator.class_indices:

        target_names.append(key)

    print(target_names)



    def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):

        """

        This function prints and plots the confusion matrix.

        Normalization can be applied by setting `normalize=True`.

        """

        plt.figure(figsize=(10,10))



        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)

        plt.colorbar()



        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes, rotation=45)

        plt.yticks(tick_marks, classes)



        if normalize:

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            cm = np.around(cm, decimals=2)

            cm[np.isnan(cm)] = 0.0

            print("Normalized confusion matrix")

        else:

            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, cm[i, j],

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()

        plt.ylabel('True label')

        plt.xlabel('Predicted label')
        plt.show()
        if is_train_generator == False:
            plt.savefig('test.png')
        else:
            plt.savefig('train.png')
        # plt.savefig('train.png')


    if is_train_generator == True:
        generator=train_generator
    elif is_train_generator == False:
        generator=validation_generator

    Y_pred = model.predict_generator(generator)

    y_pred = np.argmax(Y_pred, axis=1)

    print('Confusion Matrix')
    print(np.shape(generator.classes))
    cm = confusion_matrix(generator.classes, y_pred)

    plot_confusion_matrix(cm, target_names, title='Confusion Matrix', normalize=False)
    print(cm)
    print('Classification Report')

    print(classification_report(generator.classes, y_pred, target_names=target_names))


    if display == True:

      

        cv2.ocl.setUseOpenCL(False)


        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        cap = cv2.VideoCapture(0)
        saeid = 0
        step = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # if saeid==0:
            #     cv2.imshow('saeid', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            #     saeid=1

            for (x, y, w, h) in faces:
                # cv2.imshow('before saeid', frame)
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                # cv2.imshow('saeid', frame)

                roi_gray = gray[y:y + h, x:x + w]

                cv2.imshow('kalle', roi_gray)
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (input_target, input_target)), -1), 0)
                # print(cropped_img)
                prediction = model.predict(cropped_img)
                print(prediction)

                maxindex1 = int(np.argmax(prediction))
                max1=prediction[0][maxindex1]
                print("max 1")
                print(max1)
                print(np.shape(prediction))
                prediction=np.delete(prediction,maxindex1)
                maxindex2 = int(np.argmax(prediction))
                print(np.shape(prediction))
                max2 = prediction[maxindex2]
                print("max 2")
                print(max2)
                neveshte=""


                if max1-max2<0.1:
                # cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                #             2, cv2.LINE_AA)
                    neveshte= "ya :"+emotion_dict[maxindex1]+", ya :"+emotion_dict[maxindex2]
                else:
                    neveshte = emotion_dict[maxindex1]
                cv2.putText(frame, neveshte, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2, cv2.LINE_AA)
            # print(step)
            # step+=1
            cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if test_single_image==True:
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    gray = cv2.imread('C:\\newMethodOfProject\\data\\train\\happy\\im47.png', cv2.IMREAD_GRAYSCALE)
    print(gray)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    print(np.shape(faces))
    print("before detect face")
    for (x, y, w, h) in faces:
        print("inside face")
        # cv2.imshow('before saeid', frame)
        # cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        # cv2.imshow('saeid', frame)

        roi_gray = gray[y:y + h, x:x + w]

        cv2.imshow('kalle', roi_gray)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (input_target, input_target)), -1), 0)
        # print(cropped_img)
        prediction = model.predict(cropped_img)
        print(prediction)
        maxindex = int(np.argmax(prediction))
        print(emotion_dict[maxindex])
    # gray = cv2.imread('Emotion-detection-master/Emotion-detection-master/src/test/data/train/happy/im0.png', cv2.IMREAD_GRAYSCALE)
    print(gray)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)
    print(prediction)
    maxindex = int(np.argmax(prediction))
    print(maxindex)
    # for i in prediction[0]:
    #     print(i)
    print(emotion_dict[maxindex])

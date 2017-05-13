import cv2
import glob
import random
import numpy as np

emotions = ["anger", "happy", "sadness"]  # Emotion list
# Initialize fisher face classifier
fishface = cv2.createFisherFaceRecognizer()

data = {}

# Define function to get file list, randomly shuffle it and split 80/20


def get_files(emotion):
    files = glob.glob("dataset/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]  # get first 80% of file list
    prediction = files[-int(len(files)*0.2):]  # get last 20% of file list
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-2
        for item in training:
            image = cv2.imread(item)  # open image
            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # append image array to training data list
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer(training_data, training_labels):
    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))
    # print training_labels[:len(training_labels)/4]
    # fishface.train(training_data[:len(training_data)/4], np.asarray(training_labels[:len(training_labels)/4]))
    # fishface.train(training_data[len(training_data)/4:len(training_data)/2], np.asarray(training_labels[len(training_labels)/4:len(training_labels)/2]))
    # fishface.train(training_data[len(training_data)/2:(len(training_data)*3)/4], np.asarray(training_labels[len(training_labels)/2:(len(training_labels)*3)/4]))
    # fishface.train(training_data[-len(training_data)/4:], np.asarray(training_labels[-len(training_labels)/4:]))


def save_model(i):
    print("saving model")
    fishface.save('trained_faceclassifier'+str(i)+'.xml')
    print("model saved!")


def load_model():
    fishface.load(
        'trained_classifiers_new/trained_faceclassifier'+str(i)+'.xml')


def predict_acc(prediction_data, prediction_labels):
    print "predicting classification set"
    cnt = 0
    correct = 0.0
    incorrect = 0.0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            cv2.imwrite("difficult/%s_%s_%s.jpg" %
                        (emotions[prediction_labels[cnt]], emotions[pred], cnt), image)
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))

metascore = []
for i in range(0, 10):
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    run_recognizer(training_data, training_labels)
    save_model(i)

    correct = predict_acc(prediction_data, prediction_labels)
    print "got", correct, "percent correct!"
    metascore.append(correct)

print " \n \n end score:", np.mean(metascore), "percent correct!"

# testing to begin on test cases
# for i in range(0,10):
#   load_model(i)
#   prediction_data = []
#   prediction_labels = []
#   prediction = glob.glob("test/%s/*" %emotion)
#   correct = predict_acc(prediction_data, prediction_labels)

#   print "got", correct, "percent correct!"

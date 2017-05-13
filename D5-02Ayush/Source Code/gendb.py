import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear",
            "happy", "sadness", "surprise"]  # Define emotion order
print len(emotions)
# Returns a list of all folders with participant numbers
participants = glob.glob("source_emotion/*")
print len(participants)
for x in participants:
    part = "%s" % x[-4:]  # store current participant number
    print part
    # Store list of sessions for current participant
    for sessions in glob.glob("%s/*" % x):
        for files in glob.glob("%s/*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            # emotions are encoded as a float, readline as float, then convert
            # to integer.
            emotion = int(float(file.readline()))

            # get last 3 files for emotion
            sourcefile_emotion1 = glob.glob(
                "source_images/%s/%s/*" % (part, current_session))[-1]
            sourcefile_emotion2 = glob.glob(
                "source_images/%s/%s/*" % (part, current_session))[-2]
            sourcefile_emotion3 = glob.glob(
                "source_images/%s/%s/*" % (part, current_session))[-3]

            # set destination emotion folder according to your grouping
            temp = ""
            print emotions[emotion]
            if emotions[emotion] == "contempt":
                temp = "sadness"
            elif emotions[emotion] == "disgust":
                temp = "anger"
            else:
                temp = emotions[emotion]
            print temp  # print destination emotion folder
            # we are doing for 3 emotions only - anger, sadness, happy
            if not temp == "surprise":
                if not temp == "fear":
                    # set destination folder
                    dest_emot1 = "sorted_set/%s/%s" % (
                        temp, sourcefile_emotion1[25:])
                    dest_emot2 = "sorted_set/%s/%s" % (
                        temp, sourcefile_emotion2[25:])
                    dest_emot3 = "sorted_set/%s/%s" % (
                        temp, sourcefile_emotion3[25:])

                    copyfile(sourcefile_emotion1, dest_emot1)  # Copy file
                    copyfile(sourcefile_emotion2, dest_emot2)  # Copy file
                    copyfile(sourcefile_emotion3, dest_emot3)  # Copy file

print "done"

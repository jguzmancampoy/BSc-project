import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# module-level variables ##############################################################################################
RETRAINED_LABELS_TXT_FILE_LOC = "C:/Users/jguzm/Desktop/Project/classifier/animal_classifier/training_images/retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC =  "C:/Users/jguzm/Desktop/Project/classifier/animal_classifier/training_images/retrained_graph.pb"

TEST_IMAGES_DIR = "C:/Users/jguzm/Desktop/Project/classifier/animal_classifier/test_images"
FRAMES_DIR = "C:/Users/jguzm/Desktop/Project/classifier/animal_classifier/frames"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

#######################################################################################################################
def main():
    print("starting program . . .")
    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if
    # get a list of classifications from the labels file
    classifications = []
    # for each line in the label file . . .
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    # end for

    # show the classifications to prove out that we were able to read the label file successfully
    print("classifications = " + str(classifications))

    # load the graph from file
    with tf.gfile.GFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph, note that we don't need to be concerned with the return value
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    # if the test image directory listed above is not valid, show an error message and bail
    if not os.path.isdir(TEST_IMAGES_DIR):
        print("the test image directory does not seem to be a valid directory, check file / directory paths")
        return
    # end if

    def crop(path, input, height, width, k, page, area):
        im = Image.open(input)
        imgwidth, imgheight = im.size
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                a = im.crop(box)
                try:
                    o = a.crop(area)
                    o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
                except:
                    pass
                k +=1

    with tf.Session() as sess:

        for fileName in os.listdir(TEST_IMAGES_DIR):
                # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName.lower().endswith(".mp4") or fileName.lower().endswith(".mov")):
                continue
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            vidcap = cv2.VideoCapture(imageFileWithPath)
            success, frame = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(os.path.join(FRAMES_DIR, "frame%d.jpg" % count) , frame)     # save frame as JPEG file
                success, frame = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
            if fileName is None:
                    print("unable to open " + fileName + " as an OpenCV image")
                    continue

        # for each file in the test images directory . . .
        for fileName_ in os.listdir(FRAMES_DIR):
                # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName_.lower().endswith(".jpg") or fileName_.lower().endswith(".jpeg")):
                continue

            imageWithFilePath = os.path.join(FRAMES_DIR, fileName_)

            OpenCVImage = cv2.imread(imageWithFilePath)

            if fileName_ is None:
                    print("unable to open " + fileName_ + " as an OpenCV image")
                    continue
            # get the final tensor from the graph
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # convert the OpenCV image (numpy array) to a TensorFlow image
            tfImage = np.array(OpenCVImage)[:, :, 0:3]

            # run the network to get the predictions
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # sort predictions from most confidence to least confidence
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            print("---------------------------------------")

            # keep track of if we're going through the next for loop for the first time so we can show more info about
            # the first prediction, which is the most likely prediction (they were sorted descending above)
            onMostLikelyPrediction = True
            # for each prediction . . .
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]
                # end if

                # get confidence, then get confidence rounded to 2 places after the decimal
                confidence = predictions[0][prediction]

                # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
                if onMostLikelyPrediction:
                    # get the score as a %
                    scoreAsAPercent = confidence * 100.0
                    # show the result to std out
                    print("the object" + str(fileName_) + "appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    # write the result on the image
                    writeResultOnImage(OpenCVImage, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    # finally we can show the OpenCV image
                    cv2.imshow(fileName_, OpenCVImage)
                    # mark that we've show the most likely prediction at this point so the additional information in
                    # this if statement does not show again for this image
                    onMostLikelyPrediction = False
                # end if

                # for any prediction, show the confidence as a ratio to five decimal places
                print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")
            # end for

            # pause until a key is pressed so the user can see the current image (shown above) and the prediction info
            cv2.waitKey()
            # after a key is pressed, close the current window to prep for the next time around
            cv2.destroyAllWindows()
        # end for
    # end with

    # write the graph to file so we can view with TensorBoard
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()

# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return False
    # end if

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
def writeResultOnImage(OpenCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = OpenCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 1.0
    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(OpenCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function
#######################################################################################################################
if __name__ == "__main__":
    main()

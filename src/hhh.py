# -*- coding: utf-8 -*-
"""Performs face alignment and calculates L2 distance between the embeddings of images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import pickle
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.externals import joblib
import cv2

def main(args):

    # Detect Face factor 
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    
    # Load Classifier

    ##SVM Classifier
    #classifier_filename_exp="/home/itlgpu/fyk/facenet/model/my_classifier.pkl"
    #classifier_filename_exp="/home/itlgpu/fyk/facenet/model/lfw_classifier.pkl"
    #with open(classifier_filename_exp, 'rb') as infile:
    #                (model, class_names) = pickle.load(infile)

    ## KNN Classifier
    #classifier_filename_exp="/home/itlgpu/fyk/facenet/model/knn.model"
    classifier_filename_exp="/home/itlgpu/fyk/facenet/model/knn-20170620-1.model"
    with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = joblib.load(infile)

    print('Loaded classifier model from file "%s"' % classifier_filename_exp)
    

    # Create Namelist
    #data_dir="/home/itlgpu/fyk/my-facedata/test"
    data_dir="/home/itlgpu/fyk/facenet/dataset/lfw/lfw_mtcnnpy_160"
    dataset = facenet.get_dataset(data_dir)
    # Create a list of class names
    class_names = [ cls.name.replace('_', ' ') for cls in dataset]

    print('Created name list from dir "%s"' % data_dir)

    
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            # Capture video
            #url = "rtsp://192.168.1.94/11"
            url = "rtsp://192.168.1.94/22"
            print('Begin to capture video from ' + url)
            video_capture = cv2.VideoCapture(url)

            if not video_capture.isOpened(): 
                print ('Can not open the video from ' + url)
                sys.exit()           
 
            # Every three frame
            frame_interval = 1
            count = 0

            while True:
                ret, frame = video_capture.read()
                
                if(count%frame_interval == 0): #frame_interval==3, face detection every 3 frames
  
                    # Detect Faces  
                    img = frame
                    img_size = np.asarray(img.shape)[0:2]
                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    #print('here')
                    nrof_faces = bounding_boxes.shape[0]#number of faces
                    print('Faces detected : {}'.format(nrof_faces))

                    #if detected faces num equal zero, failed 
                    if nrof_faces <= 0:
                        continue
                    
                    img_list = [None] * nrof_faces
                    for i in xrange(nrof_faces):
                        det = np.squeeze(bounding_boxes[i,0:4])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-args.margin/2, 0)
                        bb[1] = np.maximum(det[1]-args.margin/2, 0)
                        bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                        bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                        aligned = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                        prewhitened = facenet.prewhiten(aligned)
                        img_list[i] = prewhitened
                    images = np.stack(img_list)

                    # Run forward pass to calculate embeddings
                    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    
                    # Predict 
                    predictions = model.predict_proba(emb)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    #print(len(best_class_indices))               
                    
                    # Draw face rectangle and put names of detected faces 
                    index = 0
                    for face_position in bounding_boxes:

                        face_position=face_position.astype(int)
                        #if best_class_probabilities[index] > 0.45 :
                        print('%4d  %s: %.3f' % (index, class_names[best_class_indices[index]], best_class_probabilities[index]))

                        cv2.rectangle(frame, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0, 255, 0), 2)
                        cv2.putText(frame, class_names[best_class_indices[index]], (face_position[0],face_position[1]),  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
                        index+=1
 
                    # Show
                    cv2.imshow('Video', frame)

                    # Exit 
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                #print(faces)
                count+=1  


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    #parser.add_argument('image_files', type=str, help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

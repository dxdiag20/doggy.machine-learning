#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 29 07:16:04 2018
"""

import tensorflow as tf
import sys
import os
from googletrans import Translator
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Classify images from test folder and predict dog breeds along with score.
'''


def classify_image(image_path):
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("dog_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("dog_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    index = 0
    maxScore = 0
    file = image_path
    with tf.Session() as sess:
        # Read the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        row_dict = {}
        head, tail = os.path.split(file)
        row_dict['id'] = tail.split('.')[0]
        for node_id in top_k:
            human_string = label_lines[node_id]

            # Some breed names are mismatching with breed name in csv header names.
            human_string = human_string.replace(" ", "_")
            if (human_string == 'german_short_haired_pointer'):
                human_string = 'german_short-haired_pointer'
            if (human_string == 'shih_tzu'):
                human_string = 'shih-tzu'
            if (human_string == 'wire_haired_fox_terrier'):
                human_string = 'wire-haired_fox_terrier'
            if (human_string == 'curly_coated_retriever'):
                human_string = 'curly-coated_retriever'
            if (human_string == 'black_and_tan_coonhound'):
                human_string = 'black-and-tan_coonhound'
            if (human_string == 'soft_coated_wheaten_terrier'):
                human_string = 'soft-coated_wheaten_terrier'
            if (human_string == 'flat_coated_retriever'):
                human_string = 'flat-coated_retriever'
            score = predictions[0][node_id]
            #print('%s (score = %.5f)' % (human_string, score))
            row_dict[human_string] = score
            for node_id in top_k:
                score = predictions[0][node_id]
                if (maxScore < score):
                    index = node_id
                    maxScore = score
        #print("what she looks like?")
        #print('%s (score = %.5f)' % (label_lines[index], maxScore))
    f.close()
    translator = Translator()
    translated = translator.translate(label_lines[index], dest='ko')
    return translated.text

def main():
    #classify_image('/Users/hangeulbae/Desktop/jung.jpg')
    classify_image(sys.argv[1])

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_process_video
Author:     robertdcurrier@gmail.com
Created:    2022-01-31
Notes:      Starting from scratch as we are now using Xception net and
            three classes: alexandrium, brevis and detritus. Testing and
            training are going well, so we need to integrate the
            classification code in test_and_train into this app to process
            HABscope videos.

            Note: We are going to test NOT using the ROI filtering but rather
            sending ALL ROIs to the classifier.  Not sure what's going to
            happen but it would greatly simplify the code.

2022-02-10: Added check_focus. We are using ROI filtering once again. Moved
edges and contours to hablab.  Need to start working on adding the
web components. We should use a -w option to signify web deployment. We will
have to bring in all the user auth code and pymongo code from pt4.  I was
hoping to not have everything in one app, but it gets too complicated to keep
two versions in sync. If I make a change in process_video I don't want to
have to remember to make the same change somewhere else.  So we have moved all
the hablab functions OUT of this code, so its only purpose is to process the
video and produce an image with the results. If running with -w true then all
the website stuff will need to happen. DB inserts, etc.

2022-03-09: Had a bit of bother with using the ROI vs writing to file and
loading the image. Turns out PIL used RGBA vs BGRA so we were getting corrupt
image data as far as the classifier was concerned. Writing the ROI to file
showed it as a blue image,and this evidently whacked out the classifier. We now
use cv2 to do the conversion and so far all is working great.  We also
eliminated classify_frame() from pt5_image_classifier_standalone and
pt5_image_classifier_web and just do an import from here. Now we will always
by in synch and not risk having three slightly different versions.

2022-04-14: Adopted masked and dilated segmenting for object detection. No
longer using contours/edges/cell_w/cell_h method. This seems to be much more
effective. BIG CHANGE: Pushed most code into pt5_utils.py, so that we have
one code base for all important modules like process_image, process_video,
classify_frame, etc. That way we can build all the utilities we need leveraging
the core working code.

2022-04-15: Rewrote process_video_sf to use new mask_me. Rewrote classify_frame
to do same. Really looking good now.  TO DO: Train model using new imagery
from pt5_masked_snipper.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Helper libraries
import json
import time
import sys
import os
import logging
import ffmpeg
import argparse
import statistics
import itertools
import glob
import multiprocessing as mp
import cv2 as cv2
import numpy as np
import pymongo as pymongo
from pymongo import MongoClient
from natsort import natsorted
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
from keras.preprocessing import image
# Local utilities
from pt5_utils import (string_to_tuple, get_config, load_scale,
process_video_sf, write_frame, mask_me, classify_frame, caption_frame,
calc_cellcount,load_model, check_focus, clean_tmp)

# Keep TF from yapping incessantly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
# globals
thumbs = 0

def get_cli_args():
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-05-16
    """
    logging.info('get_cli_args()')
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-o", "--output", help="output file",
                       nargs="?")
    arg_p.add_argument("-r", "--raw", help="write raw frame",
                       action='store_true')
    arg_p.add_argument("-i", "--input", help="input file",
                       required='true')
    arg_p.add_argument("-m", "--mask", help="show mask",
                       action='store_true')
    arg_p.add_argument("-c", "--contours", help="contours",
                       action='store_true')
    args = vars(arg_p.parse_args())
    return args

def init_process():
    """ Kick it, yo! """
    tensorflow.keras.backend.set_image_data_format("channels_last")
    logging.basicConfig(level=logging.INFO)
    logging.info('Initializing pt5_Xception...')
    clean_tmp()
    args = get_cli_args()
    (taxa, frame, target_cons) = process_video_sf(args)
    if args['mask']:
        mask = mask_me(taxa, frame)
        fname = 'results/%s_mask.png' % taxa
        cv2.imwrite(fname, mask)
        fname = 'results/%s_frame.png' % taxa
        cv2.imwrite(fname, frame)
    logging.info("%s has %d ROIs" % (taxa, len(target_cons)))
    (frame, matches) = classify_frame(taxa, frame, target_cons)
    # Show all contours in black
    if args['contours']:
        for con in target_cons:
            rect = cv2.boundingRect(con)
            x,y,w,h = rect
            (cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,0), 2))
    logging.info("%s has %d matches" % (taxa, matches))
    write_frame(taxa, frame)

if __name__ == '__main__':
    init_process()

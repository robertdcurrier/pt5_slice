#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_image_classifier_standalone
Author:     robertdcurrier@gmail.com
Created:    2022-03-02
Notes:     This is the web portal version with all the MongoDB and volunteer
functions needed for site locations, db inserts, etc.

2022-04-14: TO DO -- implement the new mask code and import most functions
from pt5_utils.py. We will follow the new convention and include only
necessary code here: watchdog, MongoDB, etc.  Most of the functions are now
defined in pt5_utils.py: classify_frame, process_video, process_image, etc.
NOTE:  process_images needs to be moved to pt5_utils.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Helper libraries
import json
import time
import sys
import os
import logging
import argparse
import shutil
from datetime import datetime
import pymongo as pymongo
from pymongo import MongoClient
import cv2 as cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# Keep TF from yapping incessantly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array

# Local utilities
from pt5_utils import (string_to_tuple, get_config, load_scale,
portal_write_frame, mask_me, classify_frame, caption_frame,
calc_cellcount,load_model, check_focus, clean_tmp, process_image_sf,
process_video_sf, build_db_doc, insert_record)


class hsVideoHandler(PatternMatchingEventHandler):
    def on_created(self,event):
        logging.info(event)
        # Need to check file length in loop so we can tell when upload finished
        size_past = -1
        index = 0
        while True:
            size_now = os.path.getsize(event.src_path)
            logging.info('pt5: size_past: %d size_now: %d index: %d' %
                         (size_past, size_now, index))
            # Check size and add a five second wait for the laggies
            if size_now - size_past == 0:
                # Wait five seconds to be sure
                logging.info('pt5: Sleeping for five seconds...')
                time.sleep(5)
                size_now = os.path.getsize(event.src_path)
                if size_now - size_past == 0:
                    logging.info("File upload complete")
                    # We stuff an arg as that is what the routine needs
                    args = {"input" : event.src_path}
                    (taxa,raw_frame,contours) = process_video_sf(args)
                    logging.info("Processed video successfully")

                    (class_frame, cells) = classify_frame(taxa, raw_frame,
                                                            contours)
                    logging.info('classify_frame() found %d cells' % cells)
                    if cells > 0:
                        portal_write_frame(event.src_path, class_frame)
                    else:
                        logging.warning('No ROIs found.')
                        portal_write_frame(event.src_path, raw_frame)
                        (_, file_name) = os.path.split(event.src_path)
                        doc = build_db_doc(file_name, cells)
                        insert_record(doc)
                        return

                    (_, file_name) = os.path.split(event.src_path)
                    doc = build_db_doc(file_name, cells)
                    insert_record(doc)
                    break
            else:
                index+=1
                size_past = size_now
                logging.info('pt5: Upload in progress...')
                time.sleep(1)


class hsImageHandler(PatternMatchingEventHandler):
    def on_created(self,event):
        logging.info(event)
        # Need to check file length in loop so we can tell when upload finished
        size_past = -1
        index = 0
        while True:
            size_now = os.path.getsize(event.src_path)
            logging.info('pt5: size_past: %d size_now: %d index: %d' %
                         (size_past, size_now, index))
            # Check size and add a five second wait for the laggies
            if size_now - size_past == 0:
                # Wait five seconds to be sure
                logging.info('pt5: Sleeping for five seconds...')
                time.sleep(5)
                size_now = os.path.getsize(event.src_path)
                if size_now - size_past == 0:
                    logging.info("File upload complete")
                    (taxa,raw_frame,contours) = process_image_sf(event.src_path)
                    (class_frame, cells) = classify_frame(taxa, raw_frame,
                                                            contours)
                    logging.info('classify_frame() found %d cells' % cells)
                    if cells > 0:
                        portal_write_frame(event.src_path, class_frame)
                    else:
                        logging.warning('No ROIs found.')
                        portal_write_frame(event.src_path, raw_frame)
                        (_, file_name) = os.path.split(event.src_path)
                        doc = build_db_doc(file_name, cells)
                        insert_record(doc)
                        return

                    (_, file_name) = os.path.split(event.src_path)
                    doc = build_db_doc(file_name, cells)
                    insert_record(doc)
                    break
            else:
                index+=1
                size_past = size_now
                logging.info('pt5_Xception(): Upload in progress...')
                time.sleep(1)


def pt5_watchdog():
    """Turn on watchdog for file processing."""
    config = get_config()
    file_path = config['system']['watch_dir']
    watch_type = config['system']['watch_type']
    logging.info('pt5_watchdog(): Monitoring %s for %s' % (file_path, watch_type))
    observer = Observer()
    #event_handler = hsImageHandler(patterns=[watch_type])
    event_handler = hsVideoHandler(patterns=[watch_type])
    observer.schedule(event_handler, file_path, recursive=True)
    observer.start()
    observer.join()


if __name__ == '__main__':
    tensorflow.keras.backend.set_image_data_format("channels_last")
    logging.basicConfig(level=logging.INFO)
    logging.info('pt5_masked_classifier_web initializing...')
    pt5_watchdog()

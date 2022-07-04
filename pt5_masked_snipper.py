#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_masked_snipper
Author:     robertdcurrier@gmail.com
Created:    2022-04-14
Notes:      Starting from scratch as we have vastly improved the snipping
            routines are using the masking techniques to select ROIs.
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
import numpy as np
import multiprocessing as mp
import cv2 as cv2
from natsort import natsorted
# Local utilities
from pt5_utils import (string_to_tuple, get_config, load_scale,
process_video_all, write_frame, mask_me, classify_frame, caption_frame,
calc_cellcount,load_model, check_focus, clean_tmp, validate_taxa)
thumbs = 0


def get_cli_args():
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2022-02-10

    Notes: Added edges and contour options as we work on hablab
    """
    logging.debug('get_cli_args()')
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-i", "--input", help="input file",
                       required='true')
    arg_p.add_argument("-m", "--mask", help="write masked frames",
                       action='store_true')
    arg_p.add_argument("-f", "--frames", help="write raw frames",
                       action='store_true')
    args = vars(arg_p.parse_args())
    return args


def cell_snipper(frames, good_cons):
    """
    Name:       cell_snipper
    Author:     robertdcurrier@gmail.com
    Created:    2022-02-01
    Modified:   2022-04-14
    Notes: Chops out cells from frame for use in training

    2022-02-10: Added check_focus to improve on image quality and decrease
    number of images to be discarded

    2022-02-14: Totally rewrote code so that we now take a list of frames
    and good_cons returned from process_video_all().  We iterate over each
    frame and snip ROIs. We do this so we can maximize the number of training
    images. Using only the returned 'most cons' image we would only get the
    cells in that one frame. With this methodology we can get thumbs from all
    the frames in the video, greatly increasing the number of usable images.

    2022-06-02: Added pre-mask writing of frames as we are working on object
    detection and need frames, not indvidual cells. We use the -f to toggle
    frame writing.
    """
    args = get_cli_args()
    config = get_config()
    taxa = validate_taxa(args["input"])
    logging.info('cell_snipper(%s)' % taxa)
    # Set our count to 0 thumbs
    thumbs = 0
    max_thumbs = config["taxa"][taxa]["max_thumbs"]
    x_offset = config['taxa'][taxa]['x_offset']
    y_offset = config['taxa'][taxa]['y_offset']
    roi_w = config['taxa'][taxa]['roi_w']
    roi_h = config['taxa'][taxa]['roi_h']
    frame_count = 0
    logging.info('cell_snipper(): %d frames' % len(frames))

    # No thumbs, just frames...
    if args['frames']:
        logging.info('cell_snipper(): Enagaging frame writer.')
        for frame in frames:
            fname = 'tmp/%d_%s_frame.png' % (frame_count, taxa)
            logging.info('cell_snipper(): Writing %s' % fname)
            try:
                cv2.imwrite(fname, frame)
                frame_count+=1
            except:
                logging.warning('Failed to write %s' % fname)
        sys.exit()

    if args['mask']:
        for frame in frames:
            mask = mask_me(taxa, frame)
            fname = 'tmp/%d_%s_masked.png' % (frame_count, taxa)
            logging.info('cell_snipper(): Writing %s' % fname)
            try:
                cv2.imwrite(fname, mask)
                frame_count+=1
            except:
                logging.warning('Failed to write %s' % fname)
        sys.exit()

    # No frames, just thumbs...
    for frame in frames:
        fname = 'tmp/%d_frame.png' % frame_count
        #cv2.imwrite(fname, frame)
        moments = []
        logging.debug("cell_snipper(): Processing frame %d" % frame_count)
        # We need to snip ROI for EACH frames
        for con in good_cons[frame_count]:
            rect = cv2.boundingRect(con)
            x,y,w,h = rect
            logging.debug('Frame: %d Thumbs:%d' % (frame_count, thumbs))
            logging.debug('Frame: %d %d,%d,%d,%d' % (frame_count, x,y,w,h))
            # compute the center of the cons
            M = cv2.moments(con)
            if M["m10"] > 0 and M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if cX > x_offset and cY > y_offset:
                    moments.append((cX, cY))
            # Spin through moments getting center and cutting ROI
        for moment in moments:
            cX = moment[0]
            cY = moment[1]
            y1 = cY-roi_h
            y2 = cY + roi_h
            x1 = cX - roi_w
            x2 = cX + roi_w
            logging.debug("cell_snipper(): Centerpoint: %d,%d" % (cX, cY))
            logging.debug('Frame %d ROI: %d,%d,%d,%d' % (frame_count,y1,y2,
                                                           x1,x2))
            roi = frame[y1:y2,x1:x2]
            if check_focus(taxa, roi):
                epoch = int(time.time()*10000)
                fname = ('tmp/%s_%d.png' % (taxa, epoch))
                logging.debug('cell_snipper(): Writing %s' % fname)
                if thumbs < max_thumbs:
                    try:
                        cv2.imwrite(fname, roi)
                        thumbs+=1
                    except:
                        logging.warning('cell_snipper(): Bad ROI...')
                else:
                    logging.info('cell_snipper(): Wrote %d thumbs' % thumbs)
                    sys.exit()

        frame_count+=1
    logging.info('cell_snipper(): Wrote %d thumbs' % thumbs)


def init_snip():
    """
    Main entry point
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('pt5_masked_snipper Initializing')
    clean_tmp()
    args = get_cli_args()
    (frames, good_cons) = process_video_all(args)
    cell_snipper(frames, good_cons)

if __name__ == '__main__':
    init_snip()

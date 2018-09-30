#!/usr/bin/env python
#-*-coding:utf-8-*-
from __future__ import print_function
import sys
sys.path.append('..')
from Classification import feature_match as fm 

root_path = '/home/kenneth/gitstore/datafolder/com'
fm.class_image_slow(root_path)


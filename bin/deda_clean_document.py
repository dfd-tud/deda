#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Remove Yellow Dots From White Areas of Scanned Pages

Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

'''

import cv2, argparse
from libdeda.extract_yd import ImgProcessingMixin


class Cleaner(ImgProcessingMixin):
    _verbosity = 0
    
    def __call__(self,output):
        im = cv2.imread(self.path)
        _, mask = self.processImage(im,workAtDpi=None,halftonesBlur=10,
            ydColourRange=((16,1,214),(42,120,255)),
            paperColourThreshold=225)#233
        im[mask==255] = 255
        cv2.imwrite(output,im)
        
        
class Main(object):

    def argparser(self):
        parser = argparse.ArgumentParser(description=
            'Remove Yellow Dots From White Areas of Scanned Pages')
        parser.add_argument("input", type=str, help='Path to Input File')
        parser.add_argument("output", type=str, help='Path to Output File')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        Cleaner(self.args.input)(self.args.output)
        

main = lambda:Main()()
if __name__ == "__main__":
    main()
    

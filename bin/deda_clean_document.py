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

import os, argparse
from libdeda.privacy import cleanScan

        
class Main(object):

    def argparser(self):
        parser = argparse.ArgumentParser(description=
            'Remove Yellow Dots From White Areas of Scanned Pages')
        parser.add_argument("-g", "--grayscale", default=False, action="store_true", help='Alias for --secure')
        parser.add_argument("-s", "--secure", default=False, action="store_true", help='Convert output to greyscale (secure mode)')
        parser.add_argument("input", type=str, help='Path to Input File')
        parser.add_argument("output", type=str, help='Path to Output File')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        secureMode = self.args.grayscale or self.args.secure
        outformat = os.path.splitext(self.args.output)[1]
        with open(self.args.output, "wb") as fpout:
            with open(self.args.input, "rb") as fpin:
                fpout.write(cleanScan(fpin.read(), secureMode, outformat))
        

main = lambda:Main()()
if __name__ == "__main__":
    main()
    

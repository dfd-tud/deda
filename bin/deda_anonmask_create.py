#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
'''
Creates an anonymisation mask for tracking dots. 
This needs to read a scan of the printed calibration page.

Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
'''

import sys, argparse
from libdeda.privacy import calibrationScan2Anonmask, createCalibrationpage


class Main(object):
    testpage = "testpage.pdf"

    def argparser(self):
        parser = argparse.ArgumentParser(
            description='Create an anonymisation mask for tracking dots')
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("-w", "--write", default=False, action='store_true', help='Write calibration page to "%s"'%self.testpage)
        group.add_argument("-r", "--read", type=str, metavar='FILE', help='Read scanned calibration page and create anon mask')
        parser.add_argument("-c",'--copy', action='store_true', default=False, help='Copy tracking dot pattern instead of anonymising it')
        parser.add_argument("-v",'--verbose', action='count', default=0, help='Fehlerausgabe')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        if self.args.write:
            with open(self.testpage,"wb") as fp:
                fp.write(createCalibrationpage())
            print("%s written."%self.testpage)
            #TODO print me
        if self.args.read:
            with open(self.args.read,"rb") as fp:
                mask = calibrationScan2Anonmask(fp.read(),self.args.copy)
            output = "mask.json"
            print("Writing mask data to '%s'."%output)
            with open(output,"wb") as fp:
                fp.write(mask)
        

main = lambda:Main()()
if __name__ == "__main__":
    main()
    

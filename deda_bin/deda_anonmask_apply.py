#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
'''
Apply the anonymisation mask created by anonmask_create to a page for printing

Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
'''

import sys, argparse
from libdeda.privacy import AnonmaskApplier


class Main(object):

    def argparser(self):
        parser = argparse.ArgumentParser(
            description='Apply an anonymisation mask to a page')
        parser.add_argument("--xoffset", type=float, metavar="INCHES", default=None, help='Manual horizontal TDM offset')
        parser.add_argument("--yoffset", type=float, metavar="INCHES", default=None, help='Manual vertical TDM offset')
        parser.add_argument("--dotradius", type=float, metavar="INCHES", default=None, help='Dot radius, default=%f'%AnonmaskApplier.dotRadius)
        parser.add_argument("--debug", action="store_true", default=False, help='DEBUG: Print magenta dots instead of yellow ones')

        parser.add_argument("mask", type=str, metavar="MASKFILE", help='Path to mask by anonmask_create')
        parser.add_argument("page", type=str, metavar="PDF_DOCUMENT", help='Path to page that shall be printed')
        parser.add_argument("-v",'--verbose', action='count', default=0, help='Fehlerausgabe')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        with open(self.args.mask) as fp:
            aa = AnonmaskApplier(fp.read(), self.args.dotradius, 
                self.args.xoffset, self.args.yoffset, self.args.debug)
        OUTFILE = "masked.pdf"

        print("Parameters:")
        print("\tTDM x offset: \t%f"%aa.xOffset)
        print("\tTDM y offset: \t%f"%aa.yOffset)
        print("\tDot radius: \t%f"%aa.dotRadius)
        print("\tScaling: \t%f"%aa.scale)
        
        with open(OUTFILE,"wb") as pdfout:
            with open(self.args.page, "rb") as pdfin:
                pdfout.write(aa.apply(pdfin.read()))
        print("Document written to '%s'"%OUTFILE)


main = lambda:Main()()
if __name__ == "__main__":
    main()
    

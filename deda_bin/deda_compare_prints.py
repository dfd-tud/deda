#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import argparse, os, sys
from libdeda.print_parser import comparePrints


class Main(object):

    def argparser(self):
        parser = argparse.ArgumentParser(
            description='Compare Tracking Data from Various Printed Pages')
        parser.add_argument("files", metavar="FILE", nargs="+", type=str, help='Path to Input File')
        parser.add_argument("-d",'--dpi', default=0, type=int, 
            help='Input image dpi')
        parser.add_argument("-v",'--verbose', action='count', default=0, help='Fehlerausgabe')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        printers, errors, identical = comparePrints(self.args.files, 
            ppArgs=dict(ydxArgs=dict(inputDpi=self.args.dpi),
                verbose=self.args.verbose))
        if identical: 
            print("The printers of the input have been detected as")
            print("\tIDENTICAL.")
            return
        print("The amount of detected printers from the input is \n\t%d\n"
            %len(printers))

        if len(errors) > 0: 
            print("Errors:")
            for e, f in errors:
                print("%s: %s"%(str(e),f))
            print()
        
        for i, p in enumerate(printers):
                print("Printer #%d: %s"%(i+1,p["manufacturer"]))
                for f in p["files"]:
                    print("%s"%f)
                print()


main = lambda:Main()()
if __name__ == "__main__":
    main()
    

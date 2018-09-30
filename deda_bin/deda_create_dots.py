#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
'''
Creates a custom TDM

Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
'''

import sys, argparse
from libdeda.privacy import AnonmaskApplierTdm
from libdeda.pattern_handler import Pattern4, TDM


class Main(object):

    def argparser(self):
        parser = argparse.ArgumentParser(
            description='Create a custom TDM')
        parser.add_argument("--serial", type=int, metavar="NNNNNN", default=123456)
        parser.add_argument("--manufacturer", type=str, default="Epson", help=', '.join(set(Pattern4.manufacturers.values())))
        parser.add_argument("--year", type=int, metavar="NN", default=18)
        parser.add_argument("--month", type=int, metavar="NN", default=11)
        parser.add_argument("--day", type=int, metavar="NN", default=11)
        parser.add_argument("--hour", type=int, metavar="NN", default=11)
        parser.add_argument("--minutes", type=int, metavar="NN", default=11)
        parser.add_argument("--dotradius", type=float, metavar="INCHES", default=None, help='default=%f'%AnonmaskApplierTdm.dotRadius)

        parser.add_argument("--debug", action="store_true", default=False, help='DEBUG: Print magenta dots instead of yellow ones')

        parser.add_argument("page", type=str, metavar="PDF_DOCUMENT", help='Path to page that the dots shall be added to')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        tdm = TDM(Pattern4, content=dict(
            serial=self.args.serial,
            hour=self.args.hour,
            minutes=self.args.minutes,
            day=self.args.day,
            month=self.args.month,
            year=self.args.year,
            manufacturer=self.args.manufacturer,
        ))
        print(tdm)
        OUTFILE = "new_dots.pdf"
        aa = AnonmaskApplierTdm(tdm,dotRadius=self.args.dotradius,debug=self.args.debug)
        with open(OUTFILE,"wb") as pdfout:
            with open(self.args.page,"rb") as pdfin:
                pdfout.write(aa.apply(pdfin.read()))
        print("Document written to '%s'"%OUTFILE)


main = lambda:Main()()
if __name__ == "__main__":
    main()
    

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

import sys, argparse, json
import numpy as np; import cv2
from io import BytesIO
from PyPDF2 import PdfFileReader, PdfFileWriter
from reportlab.pdfgen import canvas
from libdeda.cmyk_to_rgb import BLACK, YELLOW, CYAN


DOTRADIUS = 0.004 #in


class Main(object):

    def argparser(self):
        parser = argparse.ArgumentParser(
            description='Apply an anonymisation mask to a page')
        parser.add_argument("--xoffset", type=float, metavar="INCHES", default=None, help='Manual horizontal TDM offset')
        parser.add_argument("--yoffset", type=float, metavar="INCHES", default=None, help='Manual vertical TDM offset')
        parser.add_argument("--dotradius", type=float, metavar="INCHES", default=DOTRADIUS, help='Dot radius, default=%f'%DOTRADIUS)
        parser.add_argument("--black", action="store_true", default=False, help='DEBUG: Print dots black instead of yellow')

        parser.add_argument("mask", type=str, metavar="MASK", help='Path to mask by anonmask_create')
        parser.add_argument("page", type=str, metavar="PAGE", help='Path to page that shall be printed')
        parser.add_argument("-v",'--verbose', action='count', default=0, help='Fehlerausgabe')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        OUTFILE = "masked.pdf"

        with open(self.args.mask) as fp: 
            d = json.load(fp)
        proto = d["proto"]
        self.hps = d["hps"]
        self.vps = d["vps"]
        xOffset = self.args.xoffset or d["x_offset"]
        yOffset = self.args.yoffset or d["y_offset"]
        self.proto = [(xDot+xOffset,yDot+yOffset) for xDot, yDot in proto]
        
        self.dotRadius = self.args.dotradius
        print("Parameters:")
        print("\tTDM x offset: \t%f"%xOffset)
        print("\tTDM y offset: \t%f"%yOffset)
        print("\tDot radius: \t%f"%self.dotRadius)
        
        pdfWatermark(self.args.page, self.createMask, OUTFILE)
        print("Document written to '%s'"%OUTFILE)

    def createMask(self, w, h):
        w = float(w)
        h = float(h)
        colour = BLACK if self.args.black else YELLOW
        io = BytesIO()
        c = canvas.Canvas(io, pagesize=(w,h) )
        c.setStrokeColorRGB(*colour)
        c.setFillColorRGB(*colour)
        for x_ in range(int(w/self.hps/72+1)):
            for y_ in range(int(h/self.vps/72+1)):
                for xDot, yDot in self.proto:
                    x = (x_*self.hps+xDot%self.hps)*72
                    y = h - (y_*self.vps+yDot%self.vps)*72
                    if x > w or y > h: continue
                    c.circle(x,y,self.dotRadius*72,stroke=0,fill=1)
        c.showPage()
        c.save()
        return io


def pdfWatermark(infile, maskCreator, outfile):
    """
    For each page from infile, maskCreator will be called with its dimensions
    and shall return a single page PDF to be put on top. The result is
    written to outfile.
    @infile String,
    @maskCreator Function,
    @outfile String
    """
    output = PdfFileWriter()
    input_ = PdfFileReader(open(infile,"rb"))
    pageBoxes = [input_.getPage(p_nr).mediaBox
        for p_nr in range(input_.getNumPages())
    ]
    maxWidth = max([b.getWidth() for b in pageBoxes])
    maxHeight = max([b.getHeight() for b in pageBoxes])
    maskPdf = maskCreator(maxWidth, maxHeight)

    for p_nr in range(input_.getNumPages()):
        page = input_.getPage(p_nr)
        box = page.mediaBox
        #maskPdf = maskCreator(box.getWidth(),box.getHeight() )
        maskPage = PdfFileReader(maskPdf).getPage(0)
        page.mergePage(maskPage)
        output.addPage(page)
        
    with open(outfile,"wb") as f:
        output.write(f)


main = lambda:Main()()
if __name__ == "__main__":
    main()
    

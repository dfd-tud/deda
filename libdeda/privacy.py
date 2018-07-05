# -*- coding: utf-8 -*- 
"""
Privacy and Anonymisation functions

Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import sys, argparse, json
import numpy as np; import cv2
from io import BytesIO
from PyPDF2 import PdfFileReader, PdfFileWriter
from reportlab.pdfgen import canvas
from libdeda.cmyk_to_rgb import BLACK, YELLOW, CYAN


DOTRADIUS = 0.004 #in


class AnonmaskApplier(object):

    colour = YELLOW

    def __init__(self, mask, dotRadius=DOTRADIUS, xoffset=None, 
                yoffset=None, black=False):
        d = json.loads(mask)
        if d.get("format_ver",0) < 2:
            raise Exception(("Incompatible format version by mask '%s'. "
                "Please generate AND print a new calibration page (see "
                "deda_anonmask_create).")%self.args.mask)
        proto = d["proto"]
        self.hps = d["hps"]
        self.vps = d["vps"]
        xOffset = xoffset or d["x_offset"]
        yOffset = yoffset or d["y_offset"]
        self.proto = [(xDot+xOffset,yDot+yOffset) for xDot, yDot in proto]
        self.dotRadius = dotRadius
        self.xOffset = xOffset
        self.yOffset = yOffset
        if black: self.colour = BLACK

    def apply(self, inPdf):
        return self.pdfWatermark(inPdf, self._createMask)
  
    def _createMask(self, w, h):
        w = float(w)
        h = float(h)
        io = BytesIO()
        c = canvas.Canvas(io, pagesize=(w,h) )
        c.setStrokeColorRGB(*self.colour)
        c.setFillColorRGB(*self.colour)
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

    @staticmethod
    def pdfWatermark(pdfin, maskCreator):
        """
        For each page from infile, maskCreator will be called with its dimensions
        and shall return a single page PDF to be put on top. The result is
        written to outfile.
        @pdfin PDF binary or file path
        @maskCreator Function that creates the watermark for each page given 
            arguments width and height,
        Returns: Pdf binary
        
        Example:
            def func(w,h): ...
            with open("output.pdf","wb") as fp_out:
                with open("input.pdf","rb") as fp_in:
                    fp_out.write(pdfWatermark(pf_in.read(), func))
        """
        output = PdfFileWriter()
        if not isinstance(pdfin, bytes):
            with open(pdfin,"rb") as fp: pdfin = fp.read()
        input_ = PdfFileReader(BytesIO(pdfin))
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
        
        outIO = BytesIO()
        output.write(outIO)
        outIO.seek(0)
        return outIO.read()

        

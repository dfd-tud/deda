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

import sys, argparse, json, os
import numpy as np; import cv2
from libdeda.psfile import PSFile
from libdeda.cmyk_to_rgb import BLACK, YELLOW, CYAN


PS_PLACE_EPS = """
/placeEPS
{
    userdict begin

    /__be_state     save            def
    /__ds_count     countdictstack  def
    /__os_count     count 5 sub     def
    /showpage       {}              def

    initgraphics
    translate
    dup scale
    run

    count           __os_count sub  { pop } repeat
    countdictstack  __ds_count sub  { end } repeat
                    __be_state              restore
    end
} bind def
"""

class ApplyMask(object):

    def __init__(self,page,mask,x,y,dotradius):
        self.mask = mask
        self.xoffset = x
        self.yoffset = y
        self.dotradius = dotradius
        self.page = page


    def __call__(self):
        A4 = (8.3, 11.7)
        pagename, pageext = os.path.splitext(self.page)
        OUTFILE = pagename + "_masked.ps"
        dpi = 72
        with open(self.mask) as fp:
            d = json.load(fp)
        proto = d["proto"]
        hps = d["hps"]
        vps = d["vps"]
        xOffset = self.xoffset or d["x_offset"]
        yOffset = self.yoffset or d["y_offset"]
        print("TDM offset: %f, %f"%(xOffset, yOffset))
        proto = [(xDot+xOffset,yDot+yOffset) for xDot, yDot in proto]

        dotRadius = self.dotradius
        print("Dot radius: %f"%dotRadius)

        ps = PSFile(OUTFILE, margin=0, paper="A4")
        ps.append(PS_PLACE_EPS)
        ps.append("%d %d %d setrgbcolor"%YELLOW)
        for x_ in range(int(ps.width/hps/dpi+1)):
            for y_ in range(int(ps.height/vps/dpi+1)):
                for xDot, yDot in proto:
                    x = (x_*hps+xDot%hps)*dpi
                    y = ps.height - (y_*vps+yDot%vps)*dpi
                    if x > ps.width or y > ps.height: continue
                    ps.append("%f %f %f 0 360 arc"%(x,y,dotRadius*dpi))
                    ps.append("fill")
        #ps.append("(%s) run"%self.args.page)
        #ps.append("(%s) 1 100 100 placeEPS"%self.args.page)
        #ps.append("BeginEPSF")
        with open(self.page) as fp:
            ps.append(fp.read())
        #ps.append("EndEPSF")
        ps.close()
        print("Document written to '%s'"%OUTFILE)
        return(xOffset, yOffset, dotRadius)

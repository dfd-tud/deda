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

import sys, cv2, argparse, json, random
import numpy as np
from colorsys import rgb_to_hsv
from libdeda.psfile import PSFile
from libdeda.print_parser import PrintParser, patternDict
from libdeda.extract_yd import rotateImage, matrix2str
try:
    import Image
except ImportError:
    from PIL import Image
from libdeda.cmyk_to_rgb import CYAN, MAGENTA, BLACK
    

TESTPAGE_SIZE = (8.3, 11.7) # A4, inches
EDGE_COLOUR = MAGENTA
EDGE_MARGIN = 2.0/6 # inches
MARKER_SIZE = 5/72


class Main(object):
    testpage = "testpage.ps"

    def argparser(self):
        parser = argparse.ArgumentParser(
            description='Create an anonymisation mask for tracking dots')
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("-w", "--write", default=False, action='store_true', help='Write calibration page to "%s"'%self.testpage)
        group.add_argument("-r", "--read", type=str, metavar='FILE', help='Read scanned calibration page and create anon mask')
        parser.add_argument("-v",'--verbose', action='count', default=0, help='Fehlerausgabe')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    def __call__(self):
        if self.args.write: 
            TestpageCreator(self.testpage)()
        if self.args.read:
            TestpageParser(self.args.read)()
        

class TestpageCreator(PSFile):
    dpi=72
    imdpi=600
    
    def __init__(self,filepath):
        w = TESTPAGE_SIZE[0]*self.dpi
        h = TESTPAGE_SIZE[1]*self.dpi
        super(TestpageCreator,self).__init__(filepath,paper=(w,h),margin=0)
        self.im = np.zeros((int(TESTPAGE_SIZE[1]*self.imdpi),
            int(TESTPAGE_SIZE[0]*self.imdpi),3))
        self.im[:] = 255
        
    def __call__(self):
        #self.append("72 %d div dup scale"%self.dpi)
        self.append("%d %d %d setrgbcolor"%EDGE_COLOUR)
        for x in [0+EDGE_MARGIN*self.dpi, self.width-EDGE_MARGIN*self.dpi]:
            for y in [0+EDGE_MARGIN*self.dpi, self.height-EDGE_MARGIN*self.dpi]:
                self.append("%f %f %f %f rectfill"
                    %(x,y,MARKER_SIZE*self.dpi,MARKER_SIZE*self.dpi))
        for x in [0+EDGE_MARGIN*self.imdpi, (TESTPAGE_SIZE[0]-EDGE_MARGIN)*self.imdpi]:
            for y in [0+EDGE_MARGIN*self.imdpi, (TESTPAGE_SIZE[1]-EDGE_MARGIN)*self.imdpi]:
                self.im[int(y-MARKER_SIZE*self.imdpi)+1:int(y)+1,
                    int(x):int(x+MARKER_SIZE*self.imdpi)] = tuple(reversed(EDGE_COLOUR))
        
        self.append("%d %d %d setrgbcolor"%CYAN)
        self.append("%f %f %f %f rectfill"%
            ((EDGE_MARGIN+MARKER_SIZE)*self.dpi,EDGE_MARGIN*self.dpi,
            MARKER_SIZE*self.dpi,MARKER_SIZE*self.dpi))
        self.im[int(-EDGE_MARGIN*self.imdpi-MARKER_SIZE*self.imdpi)+1:int(-EDGE_MARGIN*self.imdpi)+1,
            int(EDGE_MARGIN*self.imdpi+MARKER_SIZE*self.imdpi):int(EDGE_MARGIN*self.imdpi+2*MARKER_SIZE*self.imdpi)] = tuple(reversed(CYAN))
        self.close()
        cv2.imwrite("testpage.png",self.im)
        #TODO print me
        

class TestpageParser(object):
    """
    Parse scanned test page and create prototype of anonymisation mask
    that can be applied on pages of any size
    """

    def __init__(self,path):
        """
        @path: Path to scanned test page image
        """
        self.im = cv2.imread(path)
        try:
                dpi = [n.numerator for n in Image.open(path).info["dpi"]]
                self.dpi = np.average(dpi)
                if self.dpi == 72: raise ValueError("assuming invalid dpi")
        except (KeyError, ValueError):
            self.dpi = round(max(self.im.shape)/max(TESTPAGE_SIZE),-2)
            sys.stderr.write("Warning: Cannot determine dpi of input. "
                +"Assuming %f dpi.\n"%self.dpi)

    def __call__(self):
        self.restoreOrientation()
        #cv2.imwrite("orientation.png",self.im)
        #cv2.imwrite("magenta.png",self._selectColour(MAGENTA))
        self.restorePerspective() #FIXME: rotation statt perspective?
        maskdata = self.createMask()
        output = "mask.json"
        print("Writing mask data to '%s'."%output)
        with open(output,"w") as f:
            json.dump(maskdata, f)
        
    @property
    def centre(self):
        return (self.im.shape[0]/2, self.im.shape[1]/2)
            
    def _selectColour(self,colour):
        edgeHue = int(rgb_to_hsv(*colour)[0]*180)
        TOLERANCE = 30
        edgeHue1 = (edgeHue-TOLERANCE)%181
        edgeHue2 = (edgeHue+TOLERANCE)%181
        hsv = cv2.cvtColor(self.im, cv2.COLOR_BGR2HSV)
        if edgeHue2 < edgeHue1: 
            raise Exception("Implementation change necessary: "
                +"Trying to select range that includes 180..0.")
        im = cv2.inRange(hsv, (edgeHue1,100,100),(edgeHue2,255,255))
        return im

    def _findContours(self, colour):
        im = self._selectColour(colour)
        contours = np.array(cv2.findContours(
            im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2],
            dtype=np.object)
        area = np.array([(np.max(contour[:,0])-np.min(contour[:,0]))
                *(np.max(contour[:,1])-np.min(contour[:,1]))
            for e in contours for contour in [e[:,0]]])/self.dpi**2
        return contours[(area > MARKER_SIZE**2/2)*(area < MARKER_SIZE**2*2)]
    
    def restoreOrientation(self):
        contours = self._findContours(CYAN)
        cEdges = np.array([c[0] for e in contours for c in e])
        dot = np.min(cEdges[:,0]), np.min(cEdges[:,1])
        angles = {False: {False: 3, True: 0}, True: {False: 2, True: 1}}
        angle = angles[dot[0]>self.centre[0]][dot[1]>self.centre[1]]
        print("Orientation correction: rotating input by %d°"%(angle*90))
        self.im = np.rot90(self.im, angle)
    
    def _getMagentaMarkers(self):
        contours = self._findContours(EDGE_COLOUR)
        if len(contours) == 0: 
                raise Exception("No contour found for colour EDGE_COLOUR")
        cEdges = np.array([c[0] for e in contours for c in e])
        #"""
        def getEdge(comp1, comp2):
            # get all contours in the quarter of the sheet selected by
            # comp1 and comp2
            area = cEdges[getattr(cEdges[:,0],comp1)(self.centre[0])
                *getattr(cEdges[:,1],comp2)(self.centre[1])]
            return np.min(area[:,0]), np.max(area[:,1])
        comparators = ['__le__','__ge__']
        # list of coords with the bottom left edges of the four magenta
        # squares. order: topleft, bottomleft, topright, bottomright
        edges = [getEdge(c1,c2) for c1 in comparators for c2 in comparators]
        inputPoints = np.array(edges,dtype=np.float32)
        return inputPoints
        
    def restorePerspective(self):
        _,_, angle = cv2.minAreaRect(self._getMagentaMarkers())
        angle = angle%90 if angle%90<45 else angle%90-90
        print("Skew correction: rotating by %+f°"%angle)
        self.im = rotateImage(self.im, angle)
        
        inputPoints = self._getMagentaMarkers()
        outputPoints = np.array([(x*self.dpi, y*self.dpi) for x,y in [
            (0+EDGE_MARGIN, 0+EDGE_MARGIN),
            (0+EDGE_MARGIN, TESTPAGE_SIZE[1]-EDGE_MARGIN),
            (TESTPAGE_SIZE[0]-EDGE_MARGIN, 0+EDGE_MARGIN),
            (TESTPAGE_SIZE[0]-EDGE_MARGIN, TESTPAGE_SIZE[1]-EDGE_MARGIN),
        ]],dtype=np.float32)
        """
        inputPoints = np.array([
            (np.min(cEdges[:,0]),np.min(cEdges[:,1])),
            (np.min(cEdges[:,0]),np.max(cEdges[:,1])),
            (np.max(cEdges[:,0]),np.min(cEdges[:,1])),
            (np.max(cEdges[:,0]),np.max(cEdges[:,1])),
        ], dtype=np.float32)
        outputPoints = np.array([(x*self.dpi, y*self.dpi) for x,y in [
            (0+EDGE_MARGIN, 0+EDGE_MARGIN-MARKER_SIZE),
            (0+EDGE_MARGIN, TESTPAGE_SIZE[1]-EDGE_MARGIN),
            (TESTPAGE_SIZE[0]-EDGE_MARGIN+MARKER_SIZE, 0+EDGE_MARGIN-MARKER_SIZE),
            (TESTPAGE_SIZE[0]-EDGE_MARGIN+MARKER_SIZE, TESTPAGE_SIZE[1]-EDGE_MARGIN),
        ]],dtype=np.float32)
        """
        l = cv2.getPerspectiveTransform(inputPoints,outputPoints)
        print("Perspective Transform")
        for (x1,y1),(x2,y2) in zip(inputPoints,outputPoints):
            print("\tMapping %d,%d -> %d,%d"%(x1,y1,x2,y2))
        testpageSizePx = (
            int(TESTPAGE_SIZE[0]*self.dpi), int(TESTPAGE_SIZE[1]*self.dpi))
        self.im = cv2.warpPerspective(self.im,l,testpageSizePx)
        
    def createMask(self):
        """
        Create a mask prototype
        """
        print("Extracting tracking dots... ")
        #cv2.imwrite("perspective.png",self.im)
        # get tracking dot matrices
        pp = PrintParser(self.im,ydxArgs=dict(inputDpi=self.dpi))
        tdms = list(pp.getAllValidTdms())
        print("\t%d valid matrices"%len(tdms))
        
        # select tdm
        #pp.tdm = random.choice(tdms)
        width = pp.yd.im.shape[1]*pp.yd.imgDpi
        height = pp.yd.im.shape[0]*pp.yd.imgDpi
        edges = [(0,0),(width,0),(0,height),(width,height)]
        tdms = [
            tdms[np.argmin(
                [abs(e1-tdm.atX)+abs(e2-tdm.atY) for tdm in tdms]
            )] for e1, e2 in edges]
        
        #m = pp.tdm.undoTransformation()
        m = pp.tdm.createMask().undoTransformation()
        print("\tTracking dots pattern found:")
        print("\tx=%d, y=%d, trans=%s"%(pp.tdm.atX,pp.tdm.atY,pp.tdm.trans))
        ni,nj,di,dj = patternDict[pp.pattern]
        hps = ni*di
        vps = nj*dj
        #tdm = pp.tdm
        #tdms = sorted(tdms,key=lambda e:e.atY+e.atX)
        """
        dots_references = [(x*di+tdm.atX/pp.yd.imgDpi, y*dj+tdm.atY/pp.yd.imgDpi)
            for tdm in tdms
            for m in [tdm.undoTransformation()]
            for x in range(m.shape[0]) for y in range(m.shape[1])
            if m[x,y] == 1]
        im = np.zeros(self.im.shape)
        for x,y in dots_references:
            im[int(y*self.dpi),int(x*self.dpi)] = 255
        cv2.imwrite("perspective_layer.png", im)
        """
        
        # create mask
        tdmCorrectionx = pp.tdm.trans["x"]*di
        tdmCorrectiony = pp.tdm.trans["y"]*dj
        dots_proto = [((x*di+tdmCorrectionx), (y*dj+tdmCorrectiony))
            for x in range(m.shape[0]) for y in range(m.shape[1]) 
            if m[x,y] == 1]
        """
        dots_proto = [
            ((x*di+pp.tdm.atX/self.dpi),(y*dj+pp.tdm.atY/self.dpi))
            for x in range(m.shape[0]) for y in range(m.shape[1]) 
            if m[x,y] == 1]
        """
        # FIXME: dont use tdm.trans. instead internal tdm function,
        # consider rotation/flips
        hps2 = hps/2 if pp.pattern in (1,5,6) else hps
        vps2 = vps/2 if pp.pattern in (1,5,6) else vps
        
        func = np.median #np.median #np.average
        xOffsets = [(tdm.atX/pp.yd.imgDpi-tdm.trans["x"]*di)%hps2
            for tdm in tdms]
        xOffsets = [e if e>hps2/2 else e+hps2 for e in xOffsets]
        xOffset = func(xOffsets)%hps2
        yOffsets = [(tdm.atY/pp.yd.imgDpi-tdm.trans["y"]*dj)%vps2
            for tdm in tdms]
        yOffsets = [e if e>vps2/2 else e+vps2 for e in yOffsets]
        yOffset = func(yOffsets)%vps2
        #print(hps,vps)
        #print((np.array(xOffsets)%hps2).tolist(), 
        #    (np.array(yOffsets)%vps2).tolist())
        #xOffset = (pp.tdm.atX/pp.yd.imgDpi-pp.tdm.trans["x"]*di)%hps
        #yOffset = (pp.tdm.atY/pp.yd.imgDpi-pp.tdm.trans["y"]*dj)%vps
        
        #print(xOffset,yOffset)
        #xOffset, yOffset = 0.619892%hps, 0.5479907%vps #ricoh
        #xOffset, yOffset = 0.3, 0.233333 #ricoh
        #xOffset, yOffset = 0.408218954%hps, 0.695310458%vps # best for 2.png
        print("TDM offset: %f, %f"%(xOffset,yOffset))

        return dict(proto=dots_proto, hps=hps, vps=vps, x_offset=xOffset,
            y_offset=yOffset)
        

if __name__ == "__main__":
    Main()()
    

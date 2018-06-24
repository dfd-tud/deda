#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
'''
This module helps extracting a yellow dots matrix from a scanned print.
See classes YellowDotsXposer and YellowDotsXposerBasic.

Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
'''

import sys, argparse, os, math, time, subprocess
import numpy as np
import cv2
from scipy.signal import argrelextrema, argrelmin, argrelmax
try:
    import Image
except ImportError:
    from PIL import Image
try:
    import matplotlib.pyplot as plt
except ImportError: pass
if sys.version_info[0] >= 3:
    xrange = range
    

WORK_AT_DPI = 300 #300 # resize input to
HALFTONES_BLUR = 20 #70 # blur factor to get rid of halftones
COLOUR_THRESHOLD = 233 # transform input to back & white to see printed spots
YELLOW_GREEDY = ((18,3,214),(39,120,255)) # [Hue, Saturation, Value] range of yellow dots #((16,), [42,67,...])
YELLOW_STRICT = ((16,10,214),(42,67,255))
YELLOW2 = ((29,11,214),(33,82,255))#((16,20,214),(42,82,255))#((16,10,214),(42,67,255))
YELLOW_STRICTHUE = ((25,10,214),(35,120,255))
YELLOW_STRICTHUE_GREEDYSAT = ((25,1,214),(35,120,255))
YELLOW = YELLOW_STRICTHUE_GREEDYSAT #YELLOW_GREEDY
BLUE = ([99,40,249],[118,72,255])#([137,40,249],[156,72,255]) # [Hue, Saturation, Value] range of blue dots TODO: schreiben
GRID_ANGLE=90/2.0 # angle to neighbour cells in degrees. Max. rotation correction = ±GRID_ANGLE/2
ANALYSING_SQUARE_WIDTH = 3.0 #6.0 #2.3 #inches
MIN_AMOUNT_DOTS = 0.08*300 #.242 # per inch²; for a page to have valid yellow dots
MAX_AMOUNT_DOTS = 11000 #3*300 # per inch²


class Main(object):

    def argparser(self):
        parser = argparse.ArgumentParser(
            description='Extract Yellow Dots From Printed Pages')
        parser.add_argument("file", type=str, help='Path to Input File')
        parser.add_argument("-m",'--mask', type=str,
            help='If file arg is monochrome mask of dots, specify inked area mask')
        parser.add_argument("-e",'--expose', default=False, 
            action='store_true', help='Only output exposed dots')
        parser.add_argument('--no-crop', default=False, 
            action='store_true', help='Do not crop the input')
        parser.add_argument("-c",'--code', type=str,
            help='Translation code')
        parser.add_argument("-d",'--dpi', default=0, type=int, 
            help='Input image dpi')
        parser.add_argument('--debug', default=False, 
            action='store_true', help='Write images while processing to yd*.png')
        parser.add_argument("-v",'--verbose', action='count', default=0, help='Fehlerausgabe')
        #parser.add_argument("-o",'--destination', type=str, default=".", help='Example Parameter')
        self.args = parser.parse_args()

    def __init__(self):
        self.argparser()
    
    @classmethod
    def _parseTransCode(self, code):
        types = [int,int,float,float]
        if code is None:
            transCode = tuple([None]*len(types))
        else:
            try:
                transCode = tuple([types[i](e) if e!='?' else None for i,e in enumerate(code.replace(" ","").split(","))])
            except Exception as e: raise Exception("%s. Argument -c must be %s."%(repr(e),repr(types)))
        return transCode
    
    def __call__(self):
        transCode = self._parseTransCode(self.args.code)
        yd = YellowDotsXposer(
            self.args.file,inputIsMask=self.args.mask is not None,
            inkedPath=self.args.mask,transCode=transCode,
            inputDpi=self.args.dpi, verbose=self.args.verbose,
            imDebugOutput=self.args.debug, onlyExpose=self.args.expose,
            noCrop=self.args.no_crop)
        #print(yd.matrix)
        

class PrintingClass(object):

    def __init__(self, verbose=0):
        self._verbosity = verbose
    
    def _print(self, verbosity, *args, **xargs):
        if not self._verbosity>=verbosity: return
        sys.stderr.write(*args,**xargs)
        sys.stderr.flush()
        

class Common(PrintingClass): 

    def _distances(self,l):
        """
        l: List [0,1,...,i-1,i,i+1,...,n-1,n]
        @returns: List [1-0, ..., i-(i-1), (i+1)-i, ..., n-(n-1)]
        """
        m = sorted(l)
        return [(m[i+1]-m[i]) for i in xrange(len(m)-1)]

    def _raise(self, cls, *args):
        e = cls(*args)
        e.yd = self
        raise e
        

class CommonImageFunctions(Common):
    
    def __init__(self, path, imDebugOutput=False, inputDpi=None):
        #super(CommonImageFunctions,self).__init__()
        self.path = path
        self._imDebugOutput = imDebugOutput
        self.imgDpi = inputDpi if inputDpi else self.getImgDpi(path)

    def getImgDpi(self, im):
        if not isinstance(im,np.ndarray):
            with Image.open(im) as pilimg:
                try:
                    dpi = [n.numerator for n in pilimg.info["dpi"]]
                except KeyError: pass
                else: return np.average(dpi)
                shape = pilimg.size
        else: shape = im.shape
        # assume DIN A4
        imgDpi = round(max(shape)/11.7,-2)
        sys.stderr.write("Error detecting input image DPI. "
                    +"Assuming %f dpi. "%imgDpi
                    +"Specify parameter inputDpi!\n")
        return imgDpi

    def _imwrite(self, filename, im):
        if not self._imDebugOutput: return
        orig = os.path.splitext(os.path.basename(self.path))[0]
        out = "yd_%s_%s.png"%(orig,filename)
        self._print(0,"Writing %s"%out)
        cv2.imwrite(out,im)
        
    def _show(self, im):
        cv2.imshow('cv2 image',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class RotationMixin(CommonImageFunctions,Common):
        
    def getAngle(self, dots):
        """ returns current rotation of @dots in degrees """
        dots = dots[:500]
        """
        pitches = [1.0*(dots[j,1]-dots[i,1])/(dots[j,0]-dots[i,0]) for i in xrange(dots.shape[0]) for j in xrange(i) if dots[j,0]>dots[i,0] and dots[j,1]>=dots[i,1]]
        pitches = [-1.0/a if a<0 else a for a in pitches]
        div = 1000 #FIXME: go independent from @div. smoothing?
        pitchesInt = np.array([a*div for a in pitches],dtype=np.int32)
        pitch = 1.0*np.argmax(np.bincount(pitchesInt))/div
        #pitch = pitches[np.argmin(np.fft.fft(pitches))]
        angle = math.degrees(math.atan(pitch))%GRID_ANGLE
        """
        angles = [self._angle(*tuple(dots[i])+tuple(dots[j]))%GRID_ANGLE for i in xrange(dots.shape[0]) for j in xrange(i)]
        div=13 #FIXME: go independent from @div!
        angle = (1.0*np.argmax(np.bincount(np.array((np.array(angles)*div).round(),dtype=np.int32) ))+.5-.5)/div#+div/2.0
        #sorted(Counter([round(x,3) for x in smooth(np.array(angles),19)]).items(),key=lambda e:e[1],reverse=True)[0:10]
        #div = 100
        #angle = np.argmax(np.bincount(np.array(smooth(np.array(angles),19)*div,dtype=np.int32)))/1.0/div
        #self._print(3,"rotation: (angle*%d, #occurrence) = %s\n"%(div,sorted(zip(np.bincount(anglesInt).nonzero()[0],np.bincount(anglesInt)[np.bincount(anglesInt).nonzero()[0]] ),key=lambda e:e[1])[-7:]))
        #print(angle,math.degrees(math.atan(angle)),math.tan(math.radians(angle)))
        if False: #plotting new
            angles = np.array(angles)
            plt.scatter(dots[:,0],dots[:,1]*-1)
            dots_ = np.array([[dots[i],dots[j]] for i in xrange(dots.shape[0]) for j in xrange(i)])
            dots_ = dots_[(angles>angle-1.0/div) * (angles<angle+1.0/div)]
            dots_ = np.array([d for dd in dots_ for d in dd])
            plt.scatter(dots_[:,1],dots_[:,0]*-1,color="r")
            plt.show()
        if False: # plotting
            angles = np.array(angles)
            mask2 = (angles > angle-1.0/div) * (angles < angle+1.0/div)
            #mask2 = np.ma.mask_or(mask,(angles > 74.6)*(angles < 74.8))
            #show = np.concatenate((A[mask2],B[mask2]))
            plt.scatter(A[:,1], A[:,0]*-1);
            plt.scatter(A[mask2][:,1], A[mask2][:,0]*-1,color="r");
            plt.scatter(B[mask2][:,1], B[mask2][:,0]*-1,color="g");
            plt.show()
        if angle>GRID_ANGLE/2.0: angle = angle-GRID_ANGLE
        #self._print(2,"angles = \n%s, angle = %f, median= %f \n"%(str(sorted(angles)),angle,np.median(angles)))
        return angle

    def _angle(self, x1, y1, x2, y2):
        """ returns the angle in radians given two points """
        r = math.degrees(math.atan2(y2-y1,x2-x1))
        if r < 0: return r+360
        return r
        

class ImgProcessingMixin(RotationMixin,CommonImageFunctions,Common):

    def processImage(self,im,halftonesBlur=HALFTONES_BLUR,
                paperColourThreshold=COLOUR_THRESHOLD,workAtDpi=WORK_AT_DPI,
                ydColourRange=YELLOW
            ):
            #CommonImageFunctions.__init__(self,*args,**xargs)
            self._print(2,"Image processing... ")
            self._print(2,"Input image dimensions: %d x %d\n"
                %tuple(reversed(im.shape[:2])))
            if workAtDpi and workAtDpi < self.imgDpi:
                im = self._resize(im,workAtDpi)
            makeBlur = lambda v: self.imgDpi/300.0*v
            mask = self._makeMask(
                im,makeBlur(halftonesBlur),paperColourThreshold)
            im = self._exposeDots(im,ydColourRange)
            im[mask==0] = 0
            self._print(2,"\n")
            #self._show(im)
            self._imwrite("dots",im)
            return im, mask
    
    def getDots(self, im):
        #return np.transpose(np.array(np.nonzero(im)))
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        contours = cv2.findContours(
            im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        #[plt.plot(contour[:,:,1], contour[:,:,0], linewidth=2) for contour in contours];plt.show()
        X = []
        Y = []
        for cnt in contours:
            x1 = min(min(cnt[:,:,0]))
            x2 = max(max(cnt[:,:,0]))
            y1 = min(min(cnt[:,:,1]))
            y2 = max(max(cnt[:,:,1]))
            x = int(x1+(x2-x1)/2.0)
            y = int(y1+(y2-y1)/2.0)
            X.append(x)
            Y.append(y)
        dots = np.transpose(np.array([X,Y]))
        return dots
        
    def crop(self, dots, h, w):
        """ separate the page into big squares and 
        choose the square with the most dots
        
        @h: acual image height
        """
        self._print(2,"Cropping... ")
        sqW = float(ANALYSING_SQUARE_WIDTH)*self.imgDpi
        if not ANALYSING_SQUARE_WIDTH:
            self._print(2,"No cropping. ANALYSING_SQUARE_WIDTH not set.\n")
            return 0,w,0,h
        if w <= sqW and h <= sqW: 
            self._print(3,"No cropping. Image is too small.\n")
            return 0, w, 0, h
        squaresX = int(math.ceil(w/sqW))
        #sqW = float(w)/squaresX
        squaresY = int(math.ceil(h/sqW))
        #sCoords = [(x*sqW,y*sqW) 
        #    for x in xrange(squaresX) for y in xrange(squaresY)]
        #sCount = [0]*len(sCoords)
        def calcSquares(deltaX,deltaY):
            sCount = np.zeros((squaresX,squaresY),dtype=np.int32)
            for x,y in dots:
                sx = int(math.floor((x+deltaX)/sqW))
                sy = int(math.floor((y+deltaY)/sqW))
                sCount[sx,sy] += 1
            xyMax = np.argmax(sCount)
            xMax = int(xyMax / sCount.shape[1])
            yMax = xyMax % sCount.shape[1]
            cropX = max(0,xMax*sqW-deltaX)
            cropY = max(0,yMax*sqW-deltaY)
            return np.max(sCount), cropX, cropY, xMax, yMax
        a = calcSquares(sqW-w%sqW, sqW-h%sqW)
        b = calcSquares(0,0)
        amount, cropX, cropY, squareX, squareY = a if a[0]>b[0] else b
        self._print(2,"at square %dx%d of %dx%d.\n"%(squareX+1,squareY+1,squaresX,squaresY))
        return int(cropX), int(cropX+sqW), int(cropY), int(cropY+sqW)
        
    def _resize(self,im,dpi):
        """ Resizes an image to @dpi dpi and return new cv2 image. """
        if dpi == self.imgDpi: return im
        factor = 1.0*dpi/self.imgDpi
        newDpi = self.imgDpi*factor
        newSize = tuple(reversed([int(p*factor) for p in im.shape[:2]]))
        self._print(2,("Resizing image from %d dpi to %d dpi, factor %f."
            +" Dimensions: %d x %d\n")%((self.imgDpi,newDpi,factor)+newSize))
        self.imgDpi = newDpi
        im = cv2.resize(im,newSize,interpolation=cv2.INTER_CUBIC)
        self._imwrite("resize",im)
        return im
    
    def _makeMask(self,im,blur,threshold):
        self._print(2,"blurring... ")
        blur = int(round(blur))
        if blur%2==0: blur += 1
        mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(mask,(blur,blur),0)
        self._print(2,"thresholding... ")
        ret,mask = cv2.threshold(mask,threshold,255,cv2.THRESH_BINARY)
        self._imwrite("mask",mask)
        return mask

    def _exposeDots(self,im,ydColourRange):
        self._print(2,"Exposing dots... ")
        # method joost
        #im2 = np.array(~((np.minimum(im[:,:,2],im[:,:,1])-im[:,:,0]<20) \
        #    * (im[:,:,2]>240) * (im[:,:,1]>240)),dtype=np.uint8)*255
        
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im = cv2.inRange(hsv, *ydColourRange)
        #kernel = np.ones((6,6),np.uint8)
        #im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
        return im
    

class RasterMixin(Common):

    def _getDotDistance(self, dots, axis):
        """ calculates dot distance from dist. between all neighbouring dots
        """
        # create distance matrix distM of dots
        z = np.array([complex(d[0],d[1]) for d in dots])
        #z = dots[:,axis] # TODO: test, ob alles auch hiermit funktionieren würde
        try:
            distM = abs(z[..., np.newaxis] - z)
        except MemoryError:
            self._raise(TooManyDotsException,self.ydPerInch)
        del z
        distM[distM==0] = np.max(distM)+1
        neighbours = np.argmin(distM, axis=0)
        #distances = distM.flat
        #distances = [abs(dots[i,axis]-dots[j,axis]) for i,j in enumerate(neighbours) if dots[i,axis]!=dots[j,axis]]
        distances = np.abs(dots[:,axis]-dots[neighbours,axis])
        distances = distances[distances>0]
        
        f = np.bincount(distances)
        #print(f[:15])
        #return np.argmax(f)
        self._print(3,"\n%s\n"%f[:30])
        maxima = np.array(argrelmax(f.argsort().argsort())[0])
        #maxima = [m for m in maxima if np.argmax(f)%m==0]
        MIN_DOT_DISTANCE = 0.01 # inches
        minDist = int(MIN_DOT_DISTANCE*self.imgDpi)
        if len(f[minDist:]) == 0: 
            self._raise(YDExtractingException,"Dot distances below minimum")
        maximum = np.argmax(f[minDist:])+minDist
        distance = maximum/2 if maximum/2.0 in maxima and float(maximum)/self.imgDpi==0.04 else maximum # because of Ricoh's marking dot
        """
        maxima_vals = f[maxima]
        threshold = np.percentile(f,93) #TODO schreiben TODO: oder percentile(f,...)?
        maxima = maxima[maxima_vals>threshold]
        maxima = [x for x in maxima if x<2 or x>len(f)-1-2 or not(f[x-2]<f[x] and f[x+2]>f[x] or f[x-2]>f[x] and f[x+2]<f[x])] #exclude hills on the way to maximum  #TODO: schreiben removed
        # a=f[x-2]-f[x]; b=f[x+2]-f[x]; a<0 and b>0 or a>0 and b<0 gdw a*b<0
        # pitch1 = 
        distance = maxima[1]
        """
        return distance

        """
        # calculate dmin from distribution of $distances 
        # distances for each dot to its closest one
        distances = np.min(distM, axis=0)
        del distM
        #div=100.0 #FIXME: depends on scaling? (hier: float2int)
        div = 1
        distances = np.array((distances*div).round(),dtype=np.int32)
        count = np.bincount(distances)
        maxima = argrelmax(count.argsort().argsort())[0]
        minima = argrelmin(count.argsort().argsort())[0]
        #dMin = minima[0]
        MIN_DOT_DISTANCE = 4/300.0 # inches # TODO: automatic detection of dublicate dots instead
        minDist = int(MIN_DOT_DISTANCE*self.imgDpi*div)
        import ipdb;ipdb.set_trace()
        distance = (np.argmax(count[minDist:])+minDist)/div
        return distance
        return maxima[1]
        """

    def _getGridDelta(self,Axis,distance):
            count = np.bincount(Axis)
            sep0 = (np.argmax(count)-1.0*distance/2)%distance
            #return int(math.floor(sep0))
            return sep0

    def _grid(self, dots, inked, distanceX, distanceY):
        """
        Notice: grid_offset_x, grid_offset_y are typically negative numbers,
            given in pixels.
        Input: dots, mask of inked area
        Output: grid_offset_x, grid_offset_y, matrix with shape <= dots.shape
        """
        if len(dots)==0: raise Exception("Argument dots must not be empty.")
        self._print(2,"Raster detection... \n")
        sepX0 = self._getGridDelta(dots[:,0],distanceX)
        sepY0 = self._getGridDelta(dots[:,1],distanceY)
        shape = (int(math.floor(float(inked.shape[1]+distanceX-sepX0)/distanceX))+1,
            int(math.floor(float(inked.shape[0]+distanceY-sepY0)/distanceY))+1)
        matrix = np.zeros(shape=shape,dtype=np.float16)
        self._print(2,"Output dimensions: %d x %d "%matrix.shape)
        self._print(2,"Writing matrix...")
        self._print(4,"X-Separators: %d, %d, %d, ..."%(sepX0,sepX0+distanceX,sepX0+2*distanceX))
        
        for xdot,ydot in dots:
            xcell = int(math.floor(float(xdot+distanceX-sepX0)/distanceX))
            ycell = int(math.floor(float(ydot+distanceY-sepY0)/distanceY))
            matrix[xcell,ycell] = 1 #if inked[ydot,xdot] != 0 else .5
        
        matrixInked = cv2.resize(inked.T,(matrix.shape[1],matrix.shape[0]))
        matrix[matrixInked<.5] = .5
        """
        m1 = dots2matrix(dots)
        inked_ = dots2matrix(inked)
        for i in xrange(sepX.size-1): for j in xrange(sepY.size-1):
            matrix[i,j] = (np.any(m1[sepY[j]:sepY[j+1], sepX[i]:sepX[i+1]])
                if 0 not in inked_[sepY[j]:sepY[j+1], sepX[i]:sepX[i+1]]
                else .5)
        """
        self._print(2,"\n")
        
        if self._verbosity>=5: # plot
            matrixplt=np.array([(x,y) for x in xrange(matrix.shape[0]) for y in xrange(matrix.shape[1]) if matrix[x,y]==1])
            matrixpltinked=np.array([(x,y) for x in xrange(matrix.shape[0]) for y in xrange(matrix.shape[1]) if matrix[x,y]==.5])
            plt.scatter(matrixplt[:,0],matrixplt[:,1]*-1)
            if len(matrixpltinked)>0:
                plt.scatter(matrixpltinked[:,0],matrixpltinked[:,1]*-1,color="r")
            plt.show()
        return sepX0-distanceX, sepY0-distanceY, matrix


class MatrixTools(object):

    @classmethod
    def commonRolling(self, m, mRef): #TODO schreiben? not used outside overlap()
        """ Roll matrix @m to the same spot as @mRef """
        return self._shifted(m,mRef)

    @classmethod
    def getShifts(self, m, x1=0, x2=None, y1=0, y2=None):
        """ returns all rollings for matrix @m given the deltas x1,x2,y1,y2,
        default: all """
        x2 = x2 or m.shape[0]-1
        y2 = y2 or m.shape[1]-1
        rx=xrange(x1,x2+1)#[-1,0,1] # xrange(-2,3)
        ry=xrange(y1,y2+1)
        return [(i,j,np.roll(m,(i,j),(0,1))) for i in rx for j in ry]
        
    @classmethod
    def _shifted(self, m, mReference, *args):
        #return m
        shifts = self.getShifts(m,*args)
        diffs = [self._arrayDiff(mReference,m_) for i,j,m_ in shifts]
        i,j,m_ = shifts[np.argmin(diffs)]
        #self._print(3,"Matrix shifted by %d x %d\n"%(i,j))
        return m_

    @classmethod
    def matrixSubsets(self, m, len0,len1):
        """ 
        Returns a subset of input matrix @m where each subset
        has the dimensions @len0 x @len1 and all party of @m are covered. 
        @returns: [(x_coord_in_m, y_coord_in_m, m_crop)]
        """
        mList = [
            (x0,y0,m[x0:(x0+len0),y0:(y0+len1)])
            for x0 in [x*len0 for x in range(int(math.floor(m.shape[0]/len0)))]+[m.shape[0]-len0]
            for y0 in [y*len1 for y in range(int(math.floor(m.shape[1]/len1)))]+[m.shape[1]-len1]
        ]
        return mList
        

class RepetitionDetectorMixin(Common,MatrixTools):
    
    @classmethod
    def _arrayDiff(self, a1, a2):
        #relative distribution of ones in a1:
        if np.sum(a1==1)==0 or np.sum(a2==1)==0: return np.nan
        mask = (a1!=.5) * (a2!=.5)
        diff = np.sum(a1[mask]!=a2[mask])
        total = a1[mask].size
        if total == 0: return np.nan
        #diff, same = np.bincount(col1[mask]==col2[mask])
        return 1.0*diff/total

    def findPatternLen(self, m):
        distM = np.zeros(shape=(m.shape[0],m.shape[0]))
        z = m.copy()
        z[z==0] = -1
        z[z==.5] = 0
        for i in xrange(m.shape[0]):
            a = z[i]*z
            total = np.sum(a!=0,axis=1)
            a[a==1] = 0
            distM[i] = np.abs(np.sum(a,axis=1))/total
        # remove cols without ones:
        ones = np.sum(z==1,axis=1)
        remove = ones<=0
        distM[:,remove] = np.nan
        distM[remove,:] = np.nan
        
        div=50.0
        f = np.bincount(np.array((distM[np.isnan(distM)==False]*div).round(),dtype=np.int32).flat)
        minima = argrelmin(f.argsort().argsort())[0]
        if len(minima) > 1:
            threshold = minima[0]/div
        elif len(f)>0:
            threshold = f.argmax()/2/div
        else: threshold = 0
        self._print(3,"threshold=%f\n"%threshold)

        X1, X2 = np.nonzero(distM<=threshold)
        #func = lambda arr: np.argmax(np.bincount(arr))
        func = np.median
        #func = np.average
        #func = lambda x: x
        distances = [
            func(self._distances(X1[X2==i]))
            for i in xrange(m.shape[0])
            if len(X1[X2==i])>1
            #and 1 in m[i]
            ]
        distFreq = np.bincount(np.array(distances,dtype=np.int32))
        self._print(3,"Pattern length frequencies: \n%s\n"%repr(distFreq))
        MIN_MATRIX_PATTERN_LENGTH = 5 #4
        MAX_MATRIX_PATTERN_LENGTH = None # 30
        distFreqWindow = distFreq[MIN_MATRIX_PATTERN_LENGTH:MAX_MATRIX_PATTERN_LENGTH]
        if len(distFreqWindow)==0:
            self._print(0,"WARNING: Cannot detect pattern length.\n")
            return -1
        period = np.argmax(distFreqWindow)+MIN_MATRIX_PATTERN_LENGTH
        return period
    
    def _splitFullMatrix(self, fullMatrix, xLen, yLen, sort=True):
        mList = [(x,y,m) for x,y,m in MatrixTools.matrixSubsets(fullMatrix,xLen,yLen)
            if np.sum(m==1) > 0]
        if sort:
            #dotCount = np.array([np.sum(m==1) for m in mList])
            #mList = list(reversed(mList[dotCount.argsort()]))
            sortfunc = lambda e: np.sum(e[-1])==1
            mList = list(sorted(mList, key=sortfunc, reverse=True))
        return mList
        
    def overlap(self, matrices, prob1=0.5, considerRepetitions=9):
        mBase = matrices[0][-1]
        mList = matrices[:considerRepetitions]
        mList = np.array([self._shifted(m,mBase,-2,2,-2,2) 
            for x,y,m in mList])

        m2 = np.zeros(mBase.shape)
        for x in xrange(m2.shape[0]):
            for y in xrange(m2.shape[1]):
                div = np.sum(mList[:,x,y]!=.5)
                if div==0:
                    self._print(0,("WARNING: overlap() cannot determine cell "
                        +"%dx%d. Try increasing @considerRepetitions.\n"
                        )%(x,y))
                else:
                    prob_ = 1.0*np.sum(mList[:,x,y]==1)/div
                    m2[x,y] = 1 if prob_ >= prob1 else 0
        return m2


class YDExtractingException(Exception): pass


class TooManyDotsException(Exception): 

    def __init__(self, ydPerInch):
        super(TooManyDotsException,self).__init__("Too many dots found (%f yellow dots per inch). Verify your mask for inked areas or adjust MAX_AMOUNT_DOTS."%ydPerInch)
        
        
class YellowDotsXposerBasic(
        RasterMixin,ImgProcessingMixin,RepetitionDetectorMixin):

    def __init__(self, path, mask=None,
            inputDpi=None, imDebugOutput=False,
            verbose=0, *args, **xargs):
        """
        Read scanned print.
        @path: str or np.array, input image file
        @mask: str, path to inked mask or None for auto/none.
        @inputDpi: float, DPI of input image. None for auto detection

        @imDebugOutput: bool, Writes an image file each step if True
        @verbose: int, from -1 for silent to 3 for high verbosity
        """
        inkedPath=mask
        inputIsMask = inkedPath is not None
        #assert(not (
        #    inputIsMask and not inputDpi
        #))
        PrintingClass.__init__(self, verbose)
        CommonImageFunctions.__init__(
            self,path,imDebugOutput=imDebugOutput,inputDpi=inputDpi)
        self.inkedPath = inkedPath
        im = path if isinstance(path,np.ndarray) else cv2.imread(path)

        self.hasDots = True
        if inputIsMask:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if not os.path.exists(inkedPath):
                sys.stderr.write(("WARNING: Mask %s not found. "
                    +"Assuming empty mask.\n")%inkedPath)
                inked = np.zeros(im.shape, np.uint8)
                inked[:] = 255
            else:
                inked = cv2.cvtColor(
                        cv2.imread(inkedPath), cv2.COLOR_BGR2GRAY)
                newShape=(
                        min(inked.shape[0],im.shape[0]),
                        min(inked.shape[1],im.shape[1])
                )
                if newShape != im.shape or newShape != inked.shape:
                    self._print(-1,"WARNING: Dimensions of image and mask"
                            +" differ.\n")
                inked = inked[0:newShape[0],0:newShape[1]]
                im = im[0:newShape[0],0:newShape[1]]
        else:
            im,inked = self.processImage(im,*args,**xargs)
        im[inked==0] = 0

        imNeg = 255-im
        dots, dots2 = self.getDots(im), self.getDots(imNeg)
        if len(dots)<len(dots2):
                im = imNeg
                dots = dots2

        printedPixels = np.sum(inked==0)
        self.ydPerInch = float(len(dots))/(im.shape[0]*im.shape[1]
            -printedPixels)*self.imgDpi**2 #TODO: schreiben
        self._print(2,"%f yellow dots per inch\n"%self.ydPerInch)
        if self.ydPerInch < MIN_AMOUNT_DOTS or len(dots)<3: 
            self._print(0,("No tracking dot pattern found :-) "
                +"(%f yellow dots per inch)\n")%self.ydPerInch)
            self.hasDots = False
            return
        if self.ydPerInch > MAX_AMOUNT_DOTS and MAX_AMOUNT_DOTS>0:
            self._raise(TooManyDotsException,self.ydPerInch)
        self.dots = dots
        self.im = im
        self.inked = inked

    def cleanDotPositions(self,rotation=True,
            crop=True):
        """
        Unrotate the input and crop a convenient area
        """
        if rotation:
                self._print(2,"rotate ")
                self.rotation = -self.getAngle(self.dots)
                self._print(2,"%+f° "%self.rotation)
                self._print(2,"\n")
                if self.rotation != 0:
                    self.im = rotateImage(self.im,-self.rotation)
                    self.inked = rotateImage(self.inked,-self.rotation)
                    self._imwrite("rotation_im",self.im)
                    self._imwrite("rotation_inked",self.inked)
                    self.dots = self.getDots(self.im)
        if crop:
                x1, x2, y1, y2 = self.crop(self.dots, *self.im.shape)
                self.im = self.im[y1:y2,x1:x2]
                self.inked = self.inked[y1:y2,x1:x2]
                self._print(2,"Cropped to %d x %d\n"%(
                    self.im.shape[1],self.im.shape[0]))
                self._imwrite("crop_im",self.im)
                self._imwrite("crop_inked",self.inked)
                self.dots = self.getDots(self.im)
                #dots = np.array([(y,x) for y,x in dots 
                #    if x>=squareX and x<squareX+sqW
                #    and y>=squareY and y<squareY+sqW
                #    ])
        #plt.scatter(dots[:,0],dots[:,1]*-1);plt.show()
        return self.dots
        
    def grid(self, dx=None, dy=None):
        """
        Make a grid and set self.fullmatrix.
        dx, dy are the cell distances in inches.
        Returns grid_offset_x, grid_offset_y, fullMatrix
            where grid_offset_x, grid_offset_y are given in pixels.
        """
        if None in (dx,dy):
            dx,dy = self.getDotDistances()
        self.gridDeltax, self.gridDeltay, self.fullMatrix = self._grid(
            self.dots,self.inked,dx*self.imgDpi,dy*self.imgDpi)
        return self.gridDeltax, self.gridDeltay, self.fullMatrix
        
    def getDotDistances(self):
        """ Calculate dx,dy automatically and return them in inches """
        self._print(2,"Find dot distance... ")
        distX = self._getDotDistance(self.dots,0)/self.imgDpi
        distY = self._getDotDistance(self.dots,1)/self.imgDpi
        self._print(2,"%f, %f\n"%(distX,distY))
        return distX,distY
            
    def getPatternShape(self):
        """
        Calculate the unique matrix' size automatically.
        Returns a tuple (n_x,n_y) where n_x and n_y are amounts of 
        matrix cells. 
        """
        self._print(2,"Finding pattern...\n")
        nx = self.findPatternLen(self.fullMatrix)
        ny = self.findPatternLen(self.fullMatrix.T)
        self._print(2,"Matrix dimensions: %d x %d\n"%(nx,ny))
        return nx,ny
    
    def separateMatrixRepetitions(self, nx=None, ny=None):
        if None in (nx,ny):
            nx,ny = self.getPatternShape()
        self.matrices = self._splitFullMatrix(self.fullMatrix, nx, ny)
        return self.matrices

    def getAllFullMatrices(self,dx,dy,cutLength=ANALYSING_SQUARE_WIDTH):
        """ 
        Returns list fullMatrices for all possible crops. 
        See also doc for getAllMatrices() 
        @returns: [(crop_x, crop_y, fullMatrix)]
        """
        w = float(cutLength*self.imgDpi)
        matrices = []
        im = self.im
        inked = self.inked
        dots = self.dots
        # TODO: if None in (dx,dy,nx,ny): once calc all for the first cut
        for x1 in ([int(x*w) for x in range(int(math.floor(im.shape[1]/w)))]+[max(0,int(im.shape[1]-w))]):
            for y1 in ([int(y*w) for y in range(int(math.floor(im.shape[0]/w)))]+[max(0,int(im.shape[0]-w))]):
                x2 = min(int(x1+w),im.shape[1])
                y2 = min(int(y1+w),im.shape[0])
                self.im = im[y1:y2,x1:x2]
                self.inked = inked[y1:y2,x1:x2]
                # set self.dots, notice: self.getDots() is expensive
                mask = (dots>=(x1,y1)).all(axis=1)*(dots<(x2,y2)).all(axis=1)
                if not np.any(mask): continue
                self.dots = dots[mask]-(x1,y1)
                #self.dots = np.array([(x-x1,y-y1) for x,y in dots 
                #    if x>=x1 and x<x2 and y>=y1 and y<y2])
                #self.cleanDotPositions(crop=False) #optional
                deltax, deltay, fullMatrix = self.grid(dx,dy)
                matrices.append((x1+deltax,y1+deltay,fullMatrix))
        self.im = im
        self.inked = inked
        self.dots = dots
        return matrices
    
    def getAllMatrices(self,nx,ny,dx,dy,**xargs):
        """
        Does cropping, puts the grid, separates the matrices and returns the
        union of the matrix lists for all squares.
        The advantage of cropping is to prevent overlapping of the grid.

        IMPORTANT: When calling this, cleanDotPositions must not have been
            called before with parameter crop=True (default).

        @length Square side length in inches
        @returns: matrix list
        """
        fullMatrices = self.getAllFullMatrices(dx,dy,**xargs)
        mList = [
            (xCrop+xCell*dx*self.imgDpi,yCrop+yCell*dy*self.imgDpi,m)
            for xCrop,yCrop,fullMatrix in fullMatrices
            for xCell,yCell,m in 
                self._splitFullMatrix(fullMatrix,nx,ny,sort=False)]
        _,idx = np.unique([m for x,y,m in mList],axis=0,return_index=True)
        mList = np.array(mList,dtype=np.object)[idx].tolist()
        #dotCount = np.array([np.sum(m==1) for m in mList])
        #mList = list(reversed(mList[dotCount.argsort()]))
        sortfunc = lambda e: np.sum(e[-1])==1
        mList = list(sorted(mList, key=sortfunc, reverse=True))
        self.matrices = mList
        return mList
    
    def __str__(self):
        if not self.hasDots: return "<Page without yellow dots>"
        return matrix2str(self.matrix)

    def __repr__(self):
        return "<YellowDotsXposer: path='%s', transCode=%s>"%(
            self.path,str(self.transCode))

YDX = YellowDotsXposerBasic


class YellowDotsXposer(YellowDotsXposerBasic):

    def __init__(self, path, transCode=(None,None,None,None), 
            inputDpi=None, imDebugOutput=False, inputIsMask=False,
            onlyPrintRotation=False, noRotation=False, noCrop=False,
            onlyExpose=False, inkedPath=None, verbose=0, *args, **xargs):
        """
        Find yellow dot matrix by scanned print.
        @path: str, input file
        @inputDpi: float, DPI of input image. None for auto detection
        @transCode: tuple (int matrix width, int matrix height,
            float dot distance X, float dot distance Y) in inches. 
            Default: auto detection
        @inputIsMask: bool, True if input is a monochrome mask
        @inkedPath: str, path to inked mask or None for auto/none.

        @noCrop: bool, Do not crop the image to the best case spot
        @noRotation: bool, Do not correct rotation of the input if True
        @onlyExpose: bool, Writes monochrome mask and returns
        @onlyPrintRotation: bool, Calculate self.rotation and stop

        @imDebugOutput: bool, Writes an image file each step if True
        @verbose: int, from -1 for silent to 3 for high verbosity
        """
        assert(not (
            onlyPrintRotation and noRotation or
            onlyPrintRotation and onlyExpose or
            inputIsMask and not inputDpi or
            inkedPath and not inputIsMask or
            inputIsMask and onlyExpose
        ))
        assert(not onlyPrintRotation)
        xLen, yLen, distX, distY = transCode
        self.transCode = transCode
        super(YellowDotsXposer,self).__init__(
            path,imDebugOutput=imDebugOutput or onlyExpose,inputDpi=inputDpi,
            mask=inkedPath,verbose=verbose,*args,**xargs)
        if onlyExpose: return
        self.cleanDotPositions(rotation=not noRotation,crop=not noCrop)
        if None in [distX,distY]: distX,distY = self.getDotDistances()
        self.grid(distX,distY)
        if None in [xLen,yLen]: xLen,yLen = self.getPatternShape()
        transCode = (xLen,yLen,distX,distY)
        self._print(0,"Detected tracking dot pattern (%d, %d, %f, %f)\n"%transCode)
        self.transCode = transCode
        if xLen < 0 or yLen < 0: return
        self.matrices = self._splitFullMatrix(self.fullMatrix,xLen,yLen)
        self.matrix = self.overlap(self.matrices)
        self._print(1,str(self))
        
        #matrix=self.matrix;matrixplt=np.array([(x,y) for x in xrange(matrix.shape[0]) for y in xrange(matrix.shape[1]) if matrix[x,y]==1]);plt.scatter(matrixplt[:,0],matrixplt[:,1]*-1);plt.show()
        #import ipdb;ipdb.set_trace()
        
        
def matrix2str(matrix):
        """ Create yellow dots string representation """
        out = "\n\n"
        out += "_"+"".join(["|%d"%(i%10) for i in xrange(matrix.shape[0])])+"\n"
        for j in xrange(matrix.shape[1]):
            out += "%d|"%(j%10)
            for i in xrange(matrix.shape[0]):
                if matrix[i,j] == 1: out += (". ")
                else: out += ("  ")
            out += ("\n")
        out += "\t%d dots.\n"%np.sum(matrix==1)
        out += ("\n\n")
        return out

def array2str(a):
    return "".join([str(int(digit)) for digit in a.flatten()])        

def rotateImage(image, angle):
    """ source: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point """
    row,col = image.shape[:2]
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image
  

if __name__ == "__main__":
    Main()()


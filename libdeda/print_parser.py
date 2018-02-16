# -*- coding: utf-8 -*- 
"""
PrintParser extracts the yellow dots from a page according to one of the
known patterns. This includes redundance checks etc.

See documentations of PrintParser and pattern_handler.

Example:
    pp = PrintParser("1.tiff")
    print(pp.tdm)


Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os, sys, random, math
import numpy as np
from libdeda.extract_yd import YDX, TooManyDotsException, matrix2str, \
    YELLOW_STRICT, YELLOW2, YELLOW_STRICTHUE, YELLOW_GREEDY, \
    YELLOW_STRICTHUE_GREEDYSAT
from libdeda.pattern_handler import patterns, TDM


knownPatterns = [
    #patternId, ni, nj, di, dj, min. score for successful detection (-1=max)
    (1, 32,32,.02,.02,  -1),
    (2, 18,23,.03,.03,  1),
    (3, 23,18,.03,.03,  1),
    (4, 24,48,.02,.02,  1),
    (5, 16,32,.04,.04,  1),
    (6, 32,16,.04,.04,  1),
    (41,48,24,.02,.02,  1),
]
"""
    hps x vps
    1:64x64
    2:54x69
    4:48x96
    5:64x128
"""

patternDict = {e[0]:e[1:5] for e in knownPatterns}
patternLimits = {e[0]:e[-1] for e in knownPatterns}


class YD_Parsing_Error(Exception): 

    def __init__(self, msg, yd):
        super(YD_Parsing_Error,self).__init__(msg)
        self.yd=yd


def parsePrint(*args, **xargs):
    return PrintParser(*args,**xargs)
    
    
class PrintParser(object):
  """
  Attributes:
  validMatrices: List of numpy matrices that passed the redundance check,
  pattern: detected pattern id,
  actions: pattern handler object, see doc for 
    pattern_handler._MatrixParserInterface
  """
  
  def __init__(self, path, ydxArgs={}, verbose=False):
    ydxArgs["path"] = path
    self._verbose = verbose
    self._ydxArgs = ydxArgs
    self.validMatrices = []
    self.allTDMs = []
    self.pattern = None
    self.tdm = None
    exceptions = []
    
    #colourProfiles = [YELLOW_STRICTHUE,YELLOW_GREEDY,YELLOW_STRICT,YELLOW_STRICTHUE_GREEDYSAT]
    colourProfiles = [YELLOW_GREEDY,YELLOW_STRICTHUE_GREEDYSAT,YELLOW_STRICTHUE]
    for i,cp in enumerate(colourProfiles):
        self.colourProfile = i
        self._print("Trying colour profile #%d... "%i)
        try:
            self._createYD_instance(ydColourRange=cp)
            self.pattern = self._calcPattern()
        except YD_Parsing_Error as e: 
            exceptions.append(e)
            self._print("%s\n"%repr(e))
        else:
            self.tdm = self.validMatrices[0]
            self._print("\n")
            break
    if self.pattern is None: 
        raise exceptions[0]

  def _createYD_instance(self,**xargs):
    self._print("Extracting dots...\n")
    ydxArgs = self._ydxArgs.copy()
    ydxArgs.update(**xargs)
    try:
        yd = YDX(**ydxArgs)
    except TooManyDotsException as e:
        raise YD_Parsing_Error("Too many dots found.",None)
    if not yd.hasDots: raise YD_Parsing_Error("No dots found.",yd)
    yd.cleanDotPositions(crop=False)
    self.yd = yd
    
  def _print(self, s):
    if not self._verbose: return
    sys.stderr.write(s)
    sys.stderr.flush()
  
  def _calcPattern(self):
    """ Returns the automatically detected pattern id """
    self._print("Getting candidates")
    mmLists = {}
    for p,patternParams in patternDict.items():
        self._print(".")
        mm = self.yd.getAllMatrices(*patternParams)#,cutLength=2.3)
        mm = [(x,y,m) for x,y,m in mm if .5 not in m and 1 in m]
        mmLists[p] = mm
    self._print(" %s"%", ".join(["p%d: %d"
        %(p,len(mm)) for p,mm in mmLists.items()]))
    self._print("\n")
    candidates = {} #patternId -> score
    validMatrices = {} #patternId -> mList
    i = 0
    while True:
        mm = {p:mm[i] for p,mm in mmLists.items() if len(mm)>i}
        if len(mm)==0: break
        valid = [(p,tdm) for p,m in mm.items() for tdm in self._getTdms(p,*m)]
        for p,tdm in valid:
            validMatrices[p] = validMatrices.get(p,[])+[tdm]
            candidates[p] = candidates.get(p,0)+1
        ci = sorted(candidates.items(),
            key=lambda e:e[1]*10+e[0],reverse=True)
        self._print(("\rTesting matrix #%d. "%i)+
            ", ".join(["p%d: %d"%(p,amount) for p,amount in ci]))
        for p,ni,nj,di,dj,s in knownPatterns:
            if s>0 and candidates.get(p,-1) >= s:
                self._print("\n")
                self.validMatrices = validMatrices[p]
                self.allTDMs = mmLists[p]
                self.tdm = validMatrices[p][0]
                self.pattern = p
                return p
        i += 1
    self._print("\n")
    if len(ci)>0 and patternLimits[ci[0][0]] == -1:
        p = ci[0][0]
        self.validMatrices = validMatrices[p]
        self.allTDMs = mmLists[p]
        # self.tdm = most common element from self.validMatrices
        self.tdm = max(set(self.validMatrices), key=self.validMatrices.count)
        self.pattern = p
        return p
    raise YD_Parsing_Error("Cannot detect pattern.",self.yd)
  
  def _get_dxy(self, x,y, pattern):
        #return 0,0
        ni,nj,di,dj = patternDict[pattern]
        di = di*self.yd.imgDpi
        dj = dj*self.yd.imgDpi
        #return di/2, dj/2
        hps = ni*di
        vps = nj*dj
        dots = self.yd.dots.T
        coords = dots[:,(dots[0] >= x) * (dots[0] < x+hps) *
            (dots[1] >= y) * (dots[1] < y+vps)]
        dx = np.average((coords[0]-x)%di)
        dy = np.average((coords[1]-y)%dj)
        return dx, dy

  def _getTdms(self, p,x,y,m):
        x_,y_=self._get_dxy(x,y,p)
        for t in patterns[p].getTransformations(m):
            tdm = TDM(pattern=p,m=m,trans=t,
                atX=x+x_,atY=y+y_)
            if tdm.check() != True: continue
            yield tdm
            #[tdm for t in patterns[p].getTransformations(m) 
            #for tdm in [TDM(pattern=p,m=m,trans=t,atX=x,atY=y)] 
            #if tdm.check() == True]
   
  def getAllValidTdms(self):
    for m in self.allTDMs:
        for tdm in self._getTdms(self.pattern,*m):
            yield tdm
    
  def getValidMatrixFromSheet(self):
    """
    Returns the valid TDM from list
    """
    return self.tdm


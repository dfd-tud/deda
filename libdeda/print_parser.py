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
    YELLOW_STRICTHUE_GREEDYSAT, YELLOW_PURE
from libdeda.pattern_handler import patterns, TDM


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
  
  def __init__(self, image, ydxArgs={}, verbose=False):
    ydxArgs["image"] = image
    self._verbose = verbose
    self._ydxArgs = ydxArgs
    self.validMatrices = []
    self.allTDMs = []
    self.pattern = None
    self.tdm = None
    exceptions = []
    
    #colourProfiles = [YELLOW_STRICTHUE,YELLOW_GREEDY,YELLOW_STRICT,YELLOW_STRICTHUE_GREEDYSAT]
    colourProfiles = [YELLOW_GREEDY,YELLOW_STRICTHUE_GREEDYSAT,YELLOW_STRICTHUE,YELLOW_PURE]
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
            self._print("\n")
            self.colourProfile = cp
            self.colourProfileId = i
            break
    if self.pattern is None: 
        raise exceptions[0]

  def _createYD_instance(self,**xargs):
    self._print("Extracting dots...\n")
    ydxArgs = self._ydxArgs.copy()
    ydxArgs.update(**xargs)
    rotation = ydxArgs.pop("rotation",True)
    try:
        yd = YDX(**ydxArgs)
    except TooManyDotsException as e:
        raise YD_Parsing_Error("Too many dots found.",None)
    if not yd.hasDots: raise YD_Parsing_Error("No dots found.",yd)
    yd.cleanDotPositions(crop=False,rotation=rotation)
    self.yd = yd
    
  def _print(self, s):
    if not self._verbose: return
    sys.stderr.write(s)
    sys.stderr.flush()
  
  def _calcPattern(self):
    """ Returns the automatically detected pattern id """
    self._print("Getting candidates")

    # make mmList: interpretation of a sheet under any pattern
    mmLists = {}
    for pattern in patterns.values():
        self._print(".")
        mm = self.yd.getAllMatrices(
            pattern.n_i, pattern.n_j, pattern.d_i, pattern.d_j
        )#,cutLength=(ni*di,nj*dj))
        mm = [(meta,m) for meta,m in mm if .5 not in m and 1 in m]
        mmLists[pattern] = mm
    self._print(" %s"%", ".join(["p%d: %d"
        %(p,len(mm)) for p,mm in mmLists.items()]))
    self._print("\n")

    candidates = {} #patternId -> score
    validMatrices = {} #patternId -> mList
    i = 0
    while True:
        mm = {p:mm[i] for p,mm in mmLists.items() if len(mm)>i}
        if len(mm)==0: break
        valid = [(p,tdm) for p,(meta,m) in mm.items() 
            for tdm in self._getTdms(p,meta,m) if tdm.check() == True]
        for p,tdm in valid:
            validMatrices[p] = validMatrices.get(p,[])+[tdm]
            candidates[p] = candidates.get(p,0)+1
        ci = sorted(candidates.items(),
            key=lambda e:e[1]*10+int(e[0]),reverse=True)
        self._print(("\rTesting matrix #%d. "%i)+
            ", ".join(["p%d: %d"%(p,amount) for p,amount in ci]))
        for p in patterns.values():
            if p.minCount>0 and candidates.get(p,-1) >= p.minCount:
                self._print("\n")
                self.validMatrices = validMatrices[p]
                self.allTDMs = mmLists[p]
                self.tdm = validMatrices[p][0]
                self.pattern = p
                return p
        i += 1
    self._print("\n")
    if len(ci)>0 and ci[0][0].minCount == -1:
        p = ci[0][0]
        self.validMatrices = validMatrices[p]
        self.allTDMs = mmLists[p]
        # self.tdm = most common element from self.validMatrices
        self.tdm = max(set(self.validMatrices), key=self.validMatrices.count)
        self.pattern = p
        return p
    raise YD_Parsing_Error("Cannot detect pattern.",self.yd)
  
  def _get_dxy(self, x,y, pattern):
        """ Returns avg distance between cell edge and dot """
        #return 0,0
        di = pattern.d_i*self.yd.imgDpi
        dj = pattern.d_j*self.yd.imgDpi
        #return di/2, dj/2
        hps = pattern.n_i*di
        vps = pattern.n_j*dj
        dots = self.yd.dots.T
        coords = dots[:,(dots[0] >= x) * (dots[0] < x+hps) *
            (dots[1] >= y) * (dots[1] < y+vps)]
        dx = np.average((coords[0]-x)%di)
        dy = np.average((coords[1]-y)%dj)
        return dx, dy

  def _getTdms(self, p,meta,m):
        cropx, cropy, gridx, gridy, cellx, celly = meta
        x = cropx+gridx+cellx
        y = cropy+gridy+celly
        x_,y_=self._get_dxy(x,y,p)
        for t in p.getTransformations(m):
            tdm = TDM(pattern=p,m=m,trans=t,
                atX=x+x_,atY=y+y_)
            tdm.meta_position = meta+(x_,y_)
            yield tdm
            #[tdm for t in p.getTransformations(m) 
            #for tdm in [TDM(pattern=p,m=m,trans=t,atX=x,atY=y)] 
            #if tdm.check() == True]
   
  def getAllValidTdms(self):
    """ TDMs where redundance check passed """
    for tdm in self.getAllTdms():
        if tdm.check() == True: yield tdm
    
  def getAllTdms(self):
    """ TDMs where redundance check passed or failed """
    for meta,m in self.allTDMs:
        for tdm in self._getTdms(self.pattern,meta,m):
            yield tdm
  
  def getAllCorrectTdms(self):
    """ TDMs where redundance check passed and content is valid """
    for tdm in self.getAllValidTdms():
        if tdm.decode()["raw"] == self.tdm.decode()["raw"]: yield tdm
  
  def getValidMatrixFromSheet(self):
    """
    Returns the valid TDM from list
    """
    return self.tdm


def comparePrints(files,ppArgs=None):
        """
        Compare the origin of a list of scanned documents
        
        files: List of Strings,
        ppArgs: Dict, args for PrintParser init,
        Returns: (printers list, errors list, identical bool)
            where printers is a list
            [{manufacturer: string, files: subset of files}]
            and errors is a list of the form [(Exception, element of files)]
            and identical is True if all files come from the same printer.
        
        """
        ppArgs = ppArgs or {}
        printers = {}
        errors = []
        for f in files:
            try:
                pp = PrintParser(f, **ppArgs)
            except Exception as e:
                errors.append((e,f))
            else:
                info = pp.tdm.decode()
                p = info["printer"]
                printers[p] = printers.get(p,info)
                printers[p]["files"] = printers[p].get("files",[])+[f]
        
        return (printers.values(), errors, 
            len(printers) == 1 and len(errors) == 0)
        

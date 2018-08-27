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

import os, sys, random, math, itertools
import numpy as np
from scipy.stats import circmean
from libdeda.extract_yd import YDX, TooManyDotsException, matrix2str, \
    YELLOW_STRICT, YELLOW2, YELLOW_STRICTHUE, YELLOW_GREEDY, \
    YELLOW_STRICTHUE_GREEDYSAT, YELLOW_PURE
from libdeda.pattern_handler import patterns


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
    self.dpi = None
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
    ydxArgs = dict(verbose=-1, inputDpi=self.dpi)
    ydxArgs.update(self._ydxArgs)
    ydxArgs.update(**xargs)
    rotation = ydxArgs.pop("rotation",True)
    try:
        yd = YDX(**ydxArgs)
    except TooManyDotsException as e:
        self.dpi = e.yd.imgDpi
        raise YD_Parsing_Error("Too many dots found.",None)
    self.dpi = yd.imgDpi
    if not yd.hasDots: raise YD_Parsing_Error("No dots found.",yd)
    yd.cleanDotPositions(crop=False,rotation=rotation)
    self.yd = yd
    
  def _print(self, s):
    if not self._verbose: return
    sys.stderr.write(s)
    sys.stderr.flush()
  
  def _calcPattern(self):
    """ Returns the automatically detected pattern id """
    # interpretation of a sheet under any of the patterns
    self._print("Getting candidates")
    tdmIterators = {p:itertools.tee(self._getTdms(p), 2)
        for p in patterns.values() if self._print(".") or True}
    tdmIteratorsFresh = {p:e[0] for p,e in tdmIterators.items()}
    tdmIterators = [(p,e[1]) for p,e in tdmIterators.items()]
    self._print("\n")

    patternScore = {p:0 for p in patterns.values()} #patternId -> score
    patternScoreSorted = list(patternScore.items()) # will stay sorted
    validMatrices = {p:[] for p in patterns.values()} #patternId -> mList
    i = 0
    while len(tdmIterators) > 0:
        self._print("\rTesting matrix #%d. "%i)
        i += 1
        p,it = tdmIterators.pop(0)
        try: tdm = next(it)
        except StopIteration: continue
        tdmIterators.append((p,it))
        if tdm.check() == False: continue
        validMatrices[p] += [tdm]
        patternScore[p] += 1
        patternScoreSorted = sorted(list(patternScore.items()),
            key=lambda e:e[1],reverse=True)
        self._print(", ".join(["p%s: %d"%(p,amount) 
                       for p,amount in patternScoreSorted]))
        if p.minCount>0 and patternScore[p] >= p.minCount:
            self._print("\n")
            self.validMatrices = validMatrices[p]
            self.allTDMs = tdmIteratorsFresh[p]
            self.tdm = validMatrices[p][0]
            self.pattern = p
            return p
    self._print("\n")
    if patternScoreSorted[0][1]>0 and patternScoreSorted[0][0].minCount==-1:
        p = patternScoreSorted[0][0]
        self.validMatrices = validMatrices[p]
        self.allTDMs = tdmIteratorsFresh[p]
        # self.tdm = most common element from self.validMatrices
        self.tdm = max(set(self.validMatrices), key=self.validMatrices.count)
        self.pattern = p
        return p
    raise YD_Parsing_Error("Cannot detect pattern.",self.yd)
  
  def _get_dxy(self, p, meta, m):
        """ Returns avg distance between cell edge and dot """
        #return 0,0
        #return di/2, dj/2
        #cropx, cropy, gridx, gridy, cellx, celly = meta
        x = sum(meta[slice(0,len(meta),2)])
        y = sum(meta[slice(1,len(meta),2)])
        dots = self.yd.dots.T/self.yd.imgDpi
        coords = dots[:,(dots[0] >= x) * (dots[0] < x+p.d_i*m.shape[0]) *
            (dots[1] >= y) * (dots[1] < y+p.d_j*m.shape[1])]
        dx = np.average((coords[0]-x)%p.d_i)
        dy = np.average((coords[1]-y)%p.d_j)
        #dx = circmean((coords[0]-x)%p.d_i, high=p.d_i)
        #dy = circmean((coords[1]-y)%p.d_j, high=p.d_j)
        return dx, dy
  
  def _getTdms(self, p):
    for meta, m in p.getAllMatricesFromYDX(self.yd):
        meta = tuple([e/self.yd.imgDpi for e in list(meta)])
        meta += self._get_dxy(p, meta, m)
        # add half a pixel because of pixel-inch inaccuracy
        meta += (.5/self.yd.imgDpi, .5/self.yd.imgDpi) 
        for tdm in p.getAlignedTDMs(meta,m): yield tdm
    
  def getAllValidTdms(self):
    """ TDMs where redundance check passed """
    for tdm in self.getAllTdms():
        if tdm.check() == True: yield tdm
    
  def getAllTdms(self):
    """ TDMs where redundance check passed or failed """
    for tdm in self.allTDMs:
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
        

# -*- coding: utf-8 -*- 
'''
The patterns and their specific functions
Align all matrices for each translation and do redundance checks

Example: 
    from extract_yd import YDX
    from pattern_handler import TDM, patterns
    
    matrices = YDX("input.jpg").getAllMatrices(
        patterns[2].n_i,patterns[2].n_j,patterns[2].d_i,patterns[2].d_j)
    for m in matrices:
        transformations = patterns[2].getTransformations(m)
        for t in transformations:
            tdm = TDM(m=m, pattern=patterns[2], trans=t)
            if tdm.check(): 
                print(tdm)
                print(tdm.decode())
                exit()
            else: print("Invalid matrix, trying other...")


Copyright 2018 Timo Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
'''

import argparse, os, sys, math, random
import numpy as np
from libdeda.extract_yd import MatrixTools, matrix2str, array2str


class _MatrixParserInterface(object):

    """
    Interface
    """
    
    def checkUnaligned(self,unaligned):
        """
        Optional function for performance reasons
        Do quick verification on original matrix that has been rotated and
        flipped but not rolled. If this is implemented,
        a pattern can be excluded without completely aligning it.
        Output: bool
        """
        pass

    def check(self,aligned): 
        """
        Verify a matrix' validity and do redundance checks
        Input: aligned matrix
        Output: bool
        """
        pass
    
    def decode(self,aligned):
        """
        Transform a matrix to human readable information
        Input: aligned matrix
        Output: dict
        """
        pass
        
    def createMask(self, aligned):
        """
        Returns anonymisation mask. United with @m it will contain
            ambiguous information.
        Input: aligned matrix
        Output: matrix of same shape as m
        """
        

class _SetPatternId(type):

    def __new__(cls, clsname, superclasses, attributedict):
        c = type.__new__(cls, clsname, superclasses, attributedict)
        c.pid = c.__int__()
        return c
        
        
class _AbstractMatrixParser(_MatrixParserInterface,
                            metaclass=_SetPatternId):

    # Size
    n_i = -1
    n_j = -1
    d_i = -1
    d_j = -1
    
    empty=[] # list of coordinate tuples where matrix should not have dots
    markers=[]
    codebits=[]
    allowFlip=False # analyse the horizontally flipped matrix as well
    allowUpsideDown=True
    allowRotation = True # If true, finds a pattern rotated by 90 degree
    
    # When reading, do a majority decision for @minCount TDMs. At least this
    # amount of valid TDMs has to be detected for a certain pattern. -1=all
    minCount = 1

    # List of (x,y) coordinates from where (0+i,0+j) shall be mapped to
    # (x+i, y+j) for all i,j
    repetitions = []
    
    # For offset patterns: Define distance between markers, default: n_i, n_j
    n_i_prototype = property(lambda self:self.n_i)
    n_j_prototype = property(lambda self:self.n_j)
    
    hps = property(lambda self:self.n_i*self.d_i)
    vps = property(lambda self:self.n_j*self.d_j)
    
    def __init__(self, m=None):
        dtype = np.uint8
        shape = (self.n_i_prototype, self.n_j_prototype)
        self.aligned = np.zeros(shape, dtype)
        for x,y in self.markers: self.aligned[x,y] = 1
        if m is not None:
            for x,y in self.codebits: self.aligned[x,y] = m[x,y]
    
    def __call__(self, *args, **xargs):
        """ Create a new instance, object acts as class constructor """
        return self.__class__(*args,**xargs)
        
    @classmethod
    def __int__(self):
        pid = ''.join(list(filter(str.isdigit, self.__name__)))
        if len(pid) > 0: return int(pid)

    def __str__(self):
        return self.__class__.__name__.replace("Pattern","")
    
    def __hash__(self):
        return hash(self.__class__.__name__)
    
    def checkUnaligned(self,unaligned):
        return True

    @classmethod
    def getAllMatricesFromYDX(pattern, ydx):
        mm = ydx.getAllMatrices(
            pattern.n_i, pattern.n_j, pattern.d_i, pattern.d_j
        )#,cutLength=(ni*di,nj*dj))
        if pattern.allowRotation: mm += ydx.getAllMatrices(
            pattern.n_j, pattern.n_i, pattern.d_j, pattern.d_i)
        return mm

    def getAlignedTDMs(self, unaligned):
        for t in self.getTransformations(unaligned):
            aligned = self.applyTransformation(unaligned,t)
            yield TDM(pattern=self(aligned), trans=t)

    def applyTransformation(self, unaligned, t):
        """ Align matrix @unaligned according to transformation @t """
        m = unaligned
        if t.get("flip"): m=np.fliplr(m)
        if t.get("rot"): m=np.rot90(m,t["rot"])
        m = np.roll(m,(t["x"],t["y"]),(0,1))
        m = m[0:self.n_i_prototype, 0:self.n_j_prototype]
        return m
    
    def undoTransformation(self, aligned, t):
        """ Unalign matrix @aligned according to transformation @t """
        m = aligned
        m2 = np.zeros((self.n_i, self.n_j),dtype=np.uint8)
        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                for rx,ry in self.repetitions+[(0,0)]:
                    m2[(rx+x)%self.n_i, (ry+y)%self.n_j] = m[x,y]
        m = m2
        #m = np.roll(m,(-d["x"],-d["y"]),(0,1))
        if t.get("rot"): m=np.rot90(m,4-t["rot"])
        if t.get("flip"): m=np.fliplr(m)
        return m
        
    def getTransformations(self,unaligned):
        """
        Get list of possible geometrical transformations to align a matrix
        @unaligned numpy matrix,
        @returns the list of the transformations that fit best @markers and 
                @empty. Format:
            [dict(
                x=int rolling_axis_0,
                y=int rolling_axis_1,
                rot=int rotated_x_times_90_deg,
                flip=bool
            )]
        """
        l = []
        rot = []
        if self.allowRotation:
            if self.n_i == unaligned.shape[0] \
            and self.n_j == unaligned.shape[1]:
                rot+=[0]
            if self.n_i == unaligned.shape[1] \
            and self.n_j == unaligned.shape[0]:
                rot+=[1]
        else: rot = [0]
        if hasattr(self,"rot"): rot = self.rot

        if self.allowUpsideDown: rot = [rud for r in rot for rud in [r,r+2]]
        for rot_ in rot:
          for flip in set([False, self.allowFlip]):
            m = unaligned.copy()
            if flip: m = np.fliplr(m)
            if rot_ != 0: m = np.rot90(m,rot_)
            if not self.checkUnaligned(m): continue
            dd = [dict(x=-x,y=-y,rot=rot_,flip=flip)
                for x in range(m.shape[0])
                for y in range(m.shape[1])
                if all([m[(dotx+x)%m.shape[0],(doty+y)%m.shape[1]] for dotx,doty in self.markers]) and not any([m[(dotx+x)%m.shape[0],(doty+y)%m.shape[1]] for dotx,doty in self.empty])
            ]
            l.extend(dd)
        return l
        

class _Pattern1(_AbstractMatrixParser):
  
  s = None # Pattern specific s parameter
  n_i = 32
  n_j = 32
  d_i = .02
  d_j = .02
  n_i_prototype = 16
  n_j_prototype = 16
  repetitions = [(16,16)]
  allowRotation = False
  minCount = -1
  markers=[(0,0),(1,0)]
  empty=None
  allowUpsideDown=False #FIXME
  rot=[0,3]
  
  C = property(lambda self: range(self.s,16,2))
  R = range(1,16,2)
  
  @property
  def codebits(self):
    return [(c,r) for c in self.C for r in self.R]
  
  def checkUnaligned(self,m):
    dots = np.sum(m,axis=0)
    return 2 in dots and sum([1 for s in dots if s%2==0]) >= 8
    #return 2 in dots and all([s%2==0 for s in dots])

  def check(self,aligned):
    m = np.array([aligned[x,y] for x,y in self.codebits]).reshape(7,8)
    sums = np.sum(m,axis=0)
    return np.sum(m)>3 and all([bool(1-int(s%2)) for s in sums])
    
  def decode(self,aligned):
    m = np.array([aligned[x,y] for x,y in self.codebits]).reshape(7,8)
    info = list(reversed(m[1:,:].T.flatten()))
    words = list(map(''.join,zip(*[iter(map(str,info))]*4)))
    decoded = "".join([str(int(b,2)) for b in words])
    snr = decoded[-12:-1]
    trans = {'0':'W','9':'P'}
    snr2 = "%s%s%s%s"%((trans.get(snr[0]) or '?'),snr[1:4],
        (trans.get(snr[4]) or '?'),snr[5:11])
    return dict(raw=decoded, serial="%s or %s"%(snr,snr2), 
        manufacturer="Ricoh/Lanier/Savin/NRG", printer=decoded)

  def createMask(self,m, allOnes=False):
    anon = np.zeros(m.shape,dtype=np.uint8)
    if allOnes:
        for x,y in self.codebits: anon[x,y] = 1
    for r in self.R:
        amountNewDots = 3 if np.sum(m[self.C,r]) == 0 else 1
        C_empty = [c for c in self.C if m[c,r] == 0]
        C_fill = random.sample(C_empty,amountNewDots)
        anon[C_fill,r] = 1
    return anon
    

class Pattern1s2(_Pattern1):
    s = 2
    empty=[(x,y) for x in range(0,16) for y in (2,4,6,8,10,12,14)]\
                +[(x,y) for x in (3,5,7,9,11,13,15) for y in range(1,16)]


class Pattern1s3(_Pattern1):
    s = 3
    empty=[(x,y) for x in range(0,16) for y in (2,4,6,8,10,12,14)]\
                +[(x,y) for x in (0,2,4,6,8,10,12,14) for y in range(1,16)]
        

class Pattern2(_AbstractMatrixParser):

  n_i = 18
  n_j = 23
  d_i = .03
  d_j = .03
  empty=[(0,0),(2,1)]
  markers=[(1,1),(1,0),(0,1)]
  allowFlip=True
  manufacturers = {"3210": "Okidata", "3021": "HP", "2310": "Ricoh", 
        "0132": "Ricoh", "0213": "Lexmark", "0123": "Kyocera"}
  blocks = [[
            [(9*x+1+2*col+1-(5*y+2+row)%2,5*y+2+row) for col in range(4)]
            for row in range(5)]
        for y in range(4) for x in range(2)]

  @property
  def codebits(self):
    a = np.array(np.array(self.blocks).flat)
    return a.reshape((int(len(a)/2),2))
    
  def check(self,aligned):
    m = np.array([[[aligned[x,y] for x,y in word] for word in words] 
        for words in self.blocks])
    return ( 
        # one hot encoding check
        all([(np.sum(words[0:4], axis=1)==1).all() for words in m])
        # parity check
        and all([(np.sum(words,axis=0)%2==1).all() for words in m]) 
    )
    
  def decode(self,aligned):
    m = np.array([[[aligned[x,y] for x,y in word] for word in words] 
        for words in self.blocks])
    trans = {'0001':0, '0010':1, '0100':2, '1000':3}
    blocks = ["".join([str(trans[array2str(w)]) for w in words[0:4]]) 
        for words in m]
    raw = "-".join(blocks)
    return dict(raw=raw, printer=raw,
        manufacturer=self.manufacturers.get(blocks[0]))
  
  def createMask(self, m):
    """ Full mask """
    anon = np.zeros(m.shape,dtype=np.uint8)
    for x,y in [z for x in self.blocks for y in x for z in y]:
        anon[x,y] = 1
    anon[m==1] = 0
    return anon
  
  def createMaskStrategic(self, m):
    oneHotCodes = [w for words in self.blocks for w in words[0:4]]
    parity = [cell for words in self.blocks for cell in words[4]]
    anon = np.zeros(m.shape,dtype=np.uint8)
    for b in oneHotCodes: # fill at random
        empty = [(x,y) for x,y in b if m[x,y] == 0]
        anon[random.choice(empty)] = 1
    empty = [(x,y) for x,y in parity if m[x,y] == 0]
    for x,y in empty: anon[x,y] = 1
    
    # blocks A and B
    anon[1:18,2:7] = 0
    trans = {str(v):k for k,v in {'0001':0, '0010':1, '0100':2, '1000':3}.items()}
    manuStr = self.decode(m)["manufacturer"]
    otherManufacturers = [k for k,s in self.manufacturers.items() 
        if s!=manuStr]
    manu2 = random.choice(otherManufacturers)
    for r,v in enumerate(manu2):
        cy = r+2
        cx = 1
        code = trans[v] # e.g. "0001"
        col2x = lambda x: cx+2*x+1-cy%2 # column index to x coordinate
        for c,v in enumerate(code):
            if v == "0": continue # or m[col2x(c),cy] == 1
            anon[col2x(c),cy] = 1
    anon[m==1] == 0
    anon[10:18,2:7] = anon[1:9,2:7] # Block B := block A
    return anon
    

class Pattern3(_AbstractMatrixParser):    

  n_i = 24
  n_j = 48
  d_i = .02
  d_j = .02
  n_i_prototype = 24
  n_j_prototype = 16
  #alignment = dict(markers = [(0,4+1),(0,1+1),(2,1+1)])
  markers = [(0,4),(0,1),(2,1)]
  blocks = [[((bx*4+by+x)%24,(3*by+1+y)%16) for y in range(3) for x in range(2)]
        for by in range(5) for bx in range(6) if bx+by>=2]
  repetitions = [(8,16),(16,32)]
  
  @property
  def codebits(self):
    a = np.array(np.array(self.blocks).flat)
    return a.reshape((len(a)/2,2))

  def checkUnaligned(self, m):
    return 30 in [np.sum(m[:,0:16]),np.sum(m[:,16:32]),np.sum(m[:,32:48])]

  def check(self,aligned):
    m = aligned[:,0:16]
    if np.sum(m) != 30: return False
    m = np.array([[m[x,y] for x,y in b] for b in self.blocks])
    return all([np.sum(block)==1 for block in m])
    
  def decode(self,aligned):
    m = aligned[:,0:16]
    m = np.array([[m[x,y] for x,y in b] for b in self.blocks])
    
    trans = {'000001':0, '000010':1, '000100':2, '001000':3, '010000':4,
        '100000':5}
    blocks = [str(trans[array2str(block)]) for block in m]
    raw = "--%s -%s %s %s %s"%(
        "".join(blocks[0:4]),"".join(blocks[4:9]),"".join(blocks[9:15]),
        "".join(blocks[15:21]),"".join(blocks[21:27]))
    return dict(raw=raw, printer=raw, manufacturer="Konica Minolta/Epson")
  
  def createMask(self, m):
    """ Full mask """
    anon = np.zeros(m.shape,dtype=np.uint8)
    for x,y in [y for x in self.blocks for y in x]:
        anon[x,y] = 1
    anon[m==1] = 0
    return anon
  
  def createMaskStrategic(self,m):
    anon = np.zeros(m.shape,dtype=np.uint8)
    for b in self.blocks:
        empty = [(x,y) for x,y in b if m[x,y]==0]
        anon[random.choice(empty)] = 1
    return anon


class Pattern4(_AbstractMatrixParser):

  n_i = 16
  n_j = 32
  d_i = .04
  d_j = .04
  n_i_prototype = 8
  n_j_prototype = 16
  empty = [(x,y) for x in range(8,16) for y in range(0,17)]
  allowFlip=True
  #markers = [(x,6) for x in range(1,4)] #1,8
  manufacturers = {0:"Xerox", 3:"Epson", 20: "Dell", 4: "Xerox"}
  codebits = [(x,y) for x in range(8) for y in range(1,16)]
  repetitions = [(8,16)]
    
  def checkUnaligned(self,m):
    # max. two rows and cols with even amount of dots
    rows = np.sum(m,axis=0)
    cols = np.sum(m,axis=1)
    return len([1 for r in rows if r%2==1]) >= 14 \
        and len([1 for c in cols if c%2==1]) >= 7

  def check(self,aligned):
    m = aligned[0:8,1:16]
    r = (all([bool(int(s%2)) for s in np.sum(m[:,:14],axis=0)])
        and all([bool(int(s%2)) for s in np.sum(m[1:,:],axis=1)])
        and np.sum(m[1:4,8])==0 and np.sum(m[1:3,9:11])==0
    )
    return bool(r)
  
  def decode(self,aligned):
    m = aligned[0:8,1:16]
    m = np.rot90(m) # same shape as EFF
    format = dict(
        minutes=1, hour=4, day=5, month=6, year=7, serial=(12,11,10),
        unknown1=2, manufacturer=3, unknown3=8, unknown4=13, unknown5=14,
        printer=(13,12,11,10))
    decoded = {}
    for key,cols in format.items():
        if not isinstance(cols,tuple): cols = (cols,)
        decoded[key] = "".join(
            ["%02d"%int(array2str(m[col,1:]),2) for col in cols])
    decoded["raw"] = "".join(["%02d"%int(array2str(m[col+1,1:]),2) 
        for col in range(m.shape[0]-1)])
    if decoded["manufacturer"].isdigit() \
            and int(decoded["manufacturer"]) in self.manufacturers: 
        decoded["manufacturer"] = self.manufacturers.get(
            int(decoded["manufacturer"]))
    else: decoded["manufacturer"] = "Xerox/Dell/Epson"
    if decoded["manufacturer"] == "Dell": decoded["serial"] = None
    else: decoded["serial"] = "-%s-"%decoded["serial"]
    return decoded
    
  def createMask(self,m):
    anon = np.zeros(m.shape,dtype=np.uint8)
    anon[0:8,15] = 1
    anon[(7,6,5,3),12] = 1
    anon[3:8,8] = 1
    anon[4:8,9] = 1
    anon[3:8,10] = 1
    for y in range(1,6):
        empty = [x for x in range(0,8) if m[x,y] == 0]
        add = max(1, 4-int(np.sum(m[:,y])))
        anon[random.sample(empty,add),y] = 1
    anon[m==1] == 0
    return anon


patterns = {cls.pid:cls()
    for name, cls in globals().items() 
    if name.startswith("Pattern")}


class TDM(object):
    """
    An aligned Tracking Dots Matrix
    """
    
    def __init__(self, pattern, trans, atX=-1, atY=-1):
        """
        One of @m or @aligned is required. @aligned must be @m transformed
            according to @trans.
        
        pattern _MatrixParserInterface pattern object,
        trans dict transformation dict,
        atX int position on paper in pixels,
        atY int,
        m np.array untransformed matrix,
        aligned np.array transformed matrix,
        """
        self.pattern = pattern
        self.aligned = pattern.aligned
        self.trans = trans
        self.atX = atX
        self.atY = atY
        
    def __getattr__(self, name):
        return getattr(self.pattern,name)(self.m)
        
    def decode(self):
        return self.pattern.decode(self.aligned)
        
    def check(self):
        return self.pattern.check(self.aligned)
        
    def createMask(self, addTdm=True):
        mask = self.pattern.createMask(self.aligned)
        if addTdm: mask = np.array((self.aligned+mask)>0,dtype=np.uint8)
        return TDM(
            atX=self.atX,atY=self.atY,
            pattern=self.pattern(mask),trans=self.trans)
        
    def __str__(self):
        return matrix2str(self.aligned)
        
    def __repr__(self):
        return "<TDM of Pattern %s at %d x %d pixels>"%(
            self.pattern,self.atX,self.atY)
        
    def undoTransformation(self):
        return self.pattern.undoTransformation(self.aligned,self.trans)
        
    def __hash__(self):
        return hash("".join([str(int(x)) for x in self.aligned.flat]))
        
    def __eq__(self, o):
        return hash(self) == hash(o)
        
        

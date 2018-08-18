#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import unittest
from io import BytesIO
from wand.image import Image as WandImage
from libdeda.pattern_handler import Pattern4, TDM
from libdeda.print_parser import PrintParser, comparePrints
from libdeda.privacy import cleanScan, createCalibrationpage, AnonmaskApplierTdm, calibrationScan2Anonmask, AnonmaskApplier


def createTdm(serial=123456, hour=11):
    return TDM(Pattern4,trans=dict(rot=3,flip=True),content=dict(
        serial = serial,
        manufacturer = "Epson",
        hour = hour,
        minutes = 11,
        day = 11,
        month = 11,
        year = 18,
    ))
    

def tdm2png(tdm, xoffset=0, yoffset=0, pdf=None):
    aa = AnonmaskApplierTdm(tdm,xoffset=xoffset,yoffset=yoffset)
    pdf = aa.apply(pdf)
    with WandImage(file=BytesIO(pdf),format="pdf",resolution=300) as wim:
        imbin = wim.make_blob("png")
    return imbin


class TestTDM(unittest.TestCase):

    def test_tdm(self):
        createTdm()
        
    def test_check(self):
        self.assertTrue(createTdm().check() == True)
        
    def test_different(self):
        self.assertTrue(createTdm(hour=1) != createTdm(hour=2))
        
    def test_equal(self):
        self.assertTrue(createTdm() == createTdm())
        

class TestPrivacy(unittest.TestCase):

    def setUp(self):
        self.tdm = createTdm()
        
    def test_cleanScan(self):
        png = tdm2png(self.tdm)
        self.assertTrue(len(cleanScan(png)) > 0)
        
    def test_anonapplyTdm(self):
        self.assertTrue(len(AnonmaskApplierTdm(self.tdm).apply(None)) > 0)
    
    def test_calibrationpage(self):
        self.assertTrue(len(createCalibrationpage()) > 0)
        
    def test_calibrationscan2anonmaskCopy(self):
        self.assertTrue(len(self.createMask(True)) > 0)
        
    def test_calibrationscan2anonmask(self):
        self.assertTrue(len(self.createMask(False)) > 0)
        
    def createMask(self, copy):
        return calibrationScan2Anonmask(
            tdm2png(self.tdm,pdf=createCalibrationpage()),copy)
        
    def test_anonapply(self):
        self.assertTrue(len(self.createMasked(True)) > 0)
        
    def createMasked(self, copy):
        mask = self.createMask(copy)
        masked = AnonmaskApplier(mask).apply(None)
        with WandImage(file=BytesIO(masked),format="pdf",resolution=300) as wim:
            maskedpng = wim.make_blob("png")
        return maskedpng

    def test_parseCopied(self):
        png = self.createMasked(True)
        self.assertTrue(PrintParser(png).tdm == self.tdm)
        
    def test_parseMasked(self):
        png = self.createMasked(False)
        try: pp = PrintParser(png)
        except: return
        else: self.assertTrue(pp.tdm != self.tdm)


class TestPrintParser(unittest.TestCase):

    def setUp(self):
        self.tdm = createTdm(serial=123456)
        self.xoffset, self.yoffset = self.tdm.hps_prototype/2, self.tdm.vps_prototype/2
        self.png = tdm2png(
            self.tdm,xoffset=self.xoffset, yoffset=self.yoffset)
        self.tdm2 = createTdm(serial=123457, hour=1)
        self.tdm3 = createTdm(serial=123457, hour=2)
        self.png2 = tdm2png(self.tdm2)
        self.png3 = tdm2png(self.tdm3)
        

    def test_parsing(self):
        # verify
        pp = PrintParser(self.png,ydxArgs=dict(inputDpi=300))
        self.assertTrue(pp.tdm == self.tdm)
        self.assertTrue(pp.pattern == self.tdm.pattern)

    def test_offset(self):
        pp = PrintParser(self.png,ydxArgs=dict(inputDpi=300))
        self.assertTrue(round(pp.tdm.xoffset,3) == round(self.xoffset,3))
        self.assertTrue(round(pp.tdm.yoffset,3) == round(self.yoffset,3))
    
    def test_compareDifferent(self):
        printers, errors, identical = comparePrints(
            [self.png, self.png2], ppArgs=dict(ydxArgs=dict(inputDpi=300)))
        self.assertTrue(len(printers) == 2)
        self.assertTrue(identical == False)
        self.assertTrue(len(errors)==0)

    def test_compareIdentical(self):
        printers, errors, identical = comparePrints(
            [self.png2, self.png3], ppArgs=dict(ydxArgs=dict(inputDpi=300)))
        self.assertTrue(len(printers) == 1)
        self.assertTrue(list(printers)[0]["manufacturer"]
                        == self.tdm2["manufacturer"])
        self.assertTrue(identical == True)
        self.assertTrue(len(errors)==0)
    
    
        
if __name__ == "__main__":
    unittest.main()
    

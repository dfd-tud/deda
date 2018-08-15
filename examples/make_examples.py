#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys
from wand.image import Image as WandImage
from libdeda.pattern_handler import Pattern4, TDM
from libdeda.privacy import AnonmaskCreator, AnonmaskApplierClass, \
                            createCalibrationpage, calibrationScan2Anonmask, \
                            AnonmaskApplier
from libdeda.print_parser import PrintParser

calibrationPage = createCalibrationpage()

tdm = Pattern4()
tdm["serial"] = 123456
tdm["manufacturer"] = "Epson"
tdm["hour"] = 11
tdm["minutes"] = 11
tdm["day"] = 11
tdm["month"] = 11
tdm["year"] = 18
tdm = TDM(tdm)
print(tdm)
print(tdm.decode())

def tdm2page(tdm,page=None):
    proto = AnonmaskCreator.tdm2mask(tdm, True)
    aa = AnonmaskApplierClass(
        proto, tdm.hps, tdm.vps, xoffset=-2, yoffset= -1)
    if page is None: return aa._createMask()
    else: return aa.apply(page)
    
if __name__ == "__main__":
    calibrationPageDotsPdf = tdm2page(tdm,calibrationPage)
    with WandImage(file=calibrationPageDotsPdf,format="pdf",resolution=300
    ) as wim:
        calibrationPageDotsPng = wim.make_blob("png")
    with open("calibrationpage-printed.png","wb") as fp:
        fp.write(calibrationPageDotsPng)
    with open("calibrationpage.pdf","wb" as fp:
        fp.write(calibrationPage)
    
    mask = calibrationScan2Anonmask(calibrationPageDotsPng)
    masked = AnonmaskApplier(mask).apply(calibrationPageDotsPdf)
    with WandImage(file=masked,format="pdf",resolution=300) as wim:
        maskedpng = wim.make_blob("png")
    with open("calibrationpage-masked.png","wb") as fp: fp.write(maskedpng)
    

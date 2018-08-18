#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys
from io import BytesIO
from wand.image import Image as WandImage
from libdeda.pattern_handler import Pattern4, TDM
from libdeda.privacy import AnonmaskApplierTdm, AnonmaskApplier, \
                            createCalibrationpage, calibrationScan2Anonmask
from libdeda.print_parser import PrintParser

calibrationPage = createCalibrationpage()

tdm = TDM(Pattern4,trans=dict(rot=3,flip=True),content=dict(
    serial = 123456
    manufacturer = "Epson"
    hour = 11
    minutes = 11
    day = 11
    month = 11,
    year = 18
))
print(tdm)
print(tdm.decode())

xoffset, yoffset = tdm.hps_prototype/2, tdm.vps_prototype/2
#print(xoffset, yoffset)
aa = AnonmaskApplierTdm(tdm,xoffset=xoffset,yoffset=yoffset)
calibrationPageDotsPdf = aa.apply(calibrationPage)
with WandImage(file=BytesIO(calibrationPageDotsPdf),format="pdf",
               resolution=300) as wim:
    calibrationPageDotsPng = wim.make_blob("png")


if __name__ == "__main__":
    with open("calibrationpage-printed.png","wb") as fp:
        fp.write(calibrationPageDotsPng)
    with open("calibrationpage.pdf","wb") as fp:
        fp.write(calibrationPage)
    
    mask = calibrationScan2Anonmask(calibrationPageDotsPng)
    masked = AnonmaskApplier(mask).apply(calibrationPageDotsPdf)
    with WandImage(file=BytesIO(masked),format="pdf",resolution=300) as wim:
        maskedpng = wim.make_blob("png")
    with open("calibrationpage-masked.png","wb") as fp: fp.write(maskedpng)
    

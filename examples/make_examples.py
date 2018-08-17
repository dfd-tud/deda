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

tdm = TDM(Pattern4,trans=dict(rot=0))
tdm["serial"] = 123456
tdm["manufacturer"] = "Epson"
tdm["hour"] = 11
tdm["minutes"] = 11
tdm["day"] = 11
tdm["month"] = 11
tdm["year"] = 18
print(tdm)
print(tdm.decode())


if __name__ == "__main__":
    aa = AnonmaskApplierTdm(tdm,xoffset=-2,yoffset=-1)
    calibrationPageDotsPdf = aa.apply(calibrationPage)
    with WandImage(file=BytesIO(calibrationPageDotsPdf),format="pdf",
                   resolution=300) as wim:
        calibrationPageDotsPng = wim.make_blob("png")
    with open("calibrationpage-printed.png","wb") as fp:
        fp.write(calibrationPageDotsPng)
    with open("calibrationpage.pdf","wb") as fp:
        fp.write(calibrationPage)
    
    mask = calibrationScan2Anonmask(calibrationPageDotsPng)
    masked = AnonmaskApplier(mask).apply(calibrationPageDotsPdf)
    with WandImage(file=BytesIO(masked),format="pdf",resolution=300) as wim:
        maskedpng = wim.make_blob("png")
    with open("calibrationpage-masked.png","wb") as fp: fp.write(maskedpng)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys
from wand.image import Image as WandImage
from libdeda.pattern_handler import Pattern4, TDM
from libdeda.privacy import AnonmaskCreator, AnonmaskApplierClass
from libdeda.print_parser import PrintParser

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
assert(tdm.check() == True)

def tdm2pdf(tdm):
    proto = AnonmaskCreator.tdm2mask(tdm, True)
    aa = AnonmaskApplierClass(
        proto, tdm.hps, tdm.vps, xoffset=-2, yoffset= -1)
    return aa._createMask()
    
if __name__ == "__main__":
    pdf = tdm2pdf(tdm)
    with WandImage(file=pdf,format="pdf",resolution=300) as wim:
        imbin = wim.make_blob("png")
    # verify
    pp = PrintParser(imbin,ydxArgs=dict(inputDpi=300))
    print(pp.tdm)
    print(pp.tdm.decode())
    assert(pp.tdm == tdm)
    


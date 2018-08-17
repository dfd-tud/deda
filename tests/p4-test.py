#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys
from io import BytesIO
from wand.image import Image as WandImage
from libdeda.pattern_handler import Pattern4, TDM
from libdeda.privacy import AnonmaskApplierTdm
from libdeda.print_parser import PrintParser

tdm = TDM(Pattern4,trans=dict(rot=1))
tdm["serial"] = 123456
tdm["manufacturer"] = "Epson"
tdm["hour"] = 11
tdm["minutes"] = 11
tdm["day"] = 11
tdm["month"] = 11
tdm["year"] = 18
print(tdm)
print(tdm.decode())
assert(tdm.check() == True)

if __name__ == "__main__":
    aa = AnonmaskApplierTdm(tdm,xoffset=-2,yoffset=-1)
    pdf = aa.apply(None)
    with WandImage(file=BytesIO(pdf),format="pdf",resolution=300) as wim:
        imbin = wim.make_blob("png")
    # verify
    pp = PrintParser(imbin,ydxArgs=dict(inputDpi=300))
    print(pp.tdm)
    print(pp.tdm.decode())
    assert(pp.tdm == tdm)
    


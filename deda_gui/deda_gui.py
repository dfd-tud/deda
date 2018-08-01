#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2018 Stephan Escher

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import cv2
import eel
import os, sys
import numpy as np
from libdeda.print_parser import *
from libdeda.extract_yd import ImgProcessingMixin
from libdeda.privacy import AnonmaskApplier, calibrationScan2Anonmask


def main():
    eel.init(os.path.join(os.path.dirname(os.path.realpath(__file__)),'web'))

    #---
    #--- Extract and Decode section
    #---
    @eel.expose
    def forensic(upload):
        if os.path.isfile(upload):
            filext = os.path.splitext(upload)[1].lower()
            filename = os.path.split(upload)[1]
            if(filext=='.jpeg' or filext=='.jpg' or filext=='.png' or filext=='.tiff' or filext=='.bmp'):
                return forensicAction(upload)
            else:
                return 'Not a valid Image. Currently only jpg, png, tiff or bmp are allowed.'
        else:
            return 'Not a valid Path to an Image'

    def forensicAction(imgfile):
        try:
            pp = PrintParser(imgfile)
        except YD_Parsing_Error as e: pattern = None
        else: pattern = pp.pattern
        print(pattern)
        if pattern is None:
            result = 'No tracking dot pattern detected.'
            return result
        result_pattern = pp.pattern

        tdm = pp.getValidMatrixFromSheet()
        if tdm is None: return 'Tracking Dots detected, but no valid TDM could be extracted.'

        #create table with yd matrix
        result_matrix = '<table class="table table-bordered table-sm table-dark table-hover text-center" id="tdmtable"><tbody>'
        tdmlist = str(tdm).split('\t')[0].split('\n')
        countdots = str(str(tdm).split('\t')[1])

        for i in tdmlist:
            if len(i)>0:
                result_matrix += '<tr><th scope = "row">' + i[0] + '</th>'
                for j in range(2,len(i),2):
                    if i[j] == '.':
                        result_matrix += '<td style="color:yellow;"><i class="far fa-dot-circle"></i></td>'
                    else:
                        result_matrix += '<td>' + i[j] + '</td>'
                result_matrix += '</tr>'
        result_matrix += '</tbody></table>'

        #create table with decoding results
        dec = tdm.decode()
        result_decoding = '<table class="table table-bordered table-sm table-hover" id="resulttable"><tbody>'
        result_decoding += '<tr><th scope="row">Analysed Image</th><td>' + os.path.split(imgfile)[1] + '</td></tr>'
        result_decoding += '<tr><th scope="row">Detected Pattern</th><td>' + str(result_pattern) + '</td></tr>'

        for i in (dec.items()):
            result_decoding += '<tr><th scope="row">' + i[0].title() + '</th><td>' + i[1] + '</td></tr>'
        result_decoding += '<tr><th scope="row">Dot Count per Pattern</th><td>' + countdots + '</td></tr></tbody></table>'

        #send results to javascript
        eel.print_result(result_matrix, result_decoding)
        return ''

    #---
    #--- Compare Documents
    #---
    @eel.expose
    def compare(uploads):
        file_uploads = []
        for upload in uploads:

            #upload is folder path:
            if os.path.isdir(upload):
                files = [os.path.join(upload, s) for s in os.listdir(upload)
                        if not os.path.isdir(os.path.join(upload, s))]
                file_count = 0
                for f in files:
                    filext = os.path.splitext(f)[1].lower()
                    filename = os.path.split(f)[1]
                    if(filext=='.jpeg' or filext=='.jpg' or filext=='.png' or filext=='.tiff' or filext=='.bmp'):
                        file_uploads.append(f)
                        file_count += 1
                if(file_count==0):
                    return (upload + ' doesnt contain a valid Image.')

            #upload is file path:
            elif os.path.isfile(upload):
                filext = os.path.splitext(upload)[1].lower()
                filename = os.path.split(upload)[1]
                if(filext=='.jpeg' or filext=='.jpg' or filext=='.png' or filext=='.tiff' or filext=='.bmp'):
                    file_uploads.append(upload)
                else:
                    return (upload + ' is not a valid Image. Currently only jpg, png, tiff or bmp are allowed.')
            #else
            else:
                return (upload + ' is not valid Path to an image or a folder containing them.')

        if len(file_uploads)<2:
            return 'To compare tracking dots you have to enter at least 2 pictures or a folder containing them.'

        #compare Image Uploads
        return compareAction(file_uploads)


    def compareAction(file_uploads):
        printers = {}
        errors = []
        info = ""
        for f in file_uploads:
            try:
                pp = PrintParser(f)
            except Exception as e:
                errors.append((e,f))
            else:
                info = pp.tdm.decode()
                p = info["printer"]
                printers[p] = printers.get(p,[])+[(f,info)]

        if len(printers) == 1 and len(errors) == 0:
            info = "The tracking dots of all input images have been detected as IDENTICAL."
        elif len(printers) == 1:
            info = str(len(file_uploads)-len(errors)) + ' of ' + str(len(file_uploads)) + ' images were succesfully analysed and have been detected as IDENTICAL. <br> ' + str(len(errors)) + ' images have no tracking dots or could not be extracted.'
        elif len(errors) == 0:
            info = 'The tracking dots of all input images have been detected.<br>' + str(len(file_uploads)) + ' images have been extracted. <br>' + str(len(printers)) + ' different patterns were found.'
        elif len(printers) == 0:
            info = 'No tracking dots found.'
        else:
            info = 'Tracking dots could be extracted in ' + str(len(file_uploads)-len(errors)) + ' of ' + str(len(file_uploads)) + ' images. <br>' + str(len(printers)) + ' different patterns were found. <br>'+ str(len(errors)) + ' images have no tracking dots or could not be extracted.'

        result_compare = '<table class="table table-bordered table-sm table-hover" id="resulttable">'
        result_compare += '<thead><tr><th scope="col">#</th><th scope="col">Printer Information</th><th scope="col">Files</th></tr></thead><tbody>'
        for i, (printerId, filesinfo) in enumerate(printers.items()):
            result_compare += '<tr><th scope="row">' + str(i+1) + '</th>'
            result_compare += '<td>'
            for key,val in filesinfo[0][1].items():
                result_compare += '<i>' + key.title() + '</i>: ' + val + '<br>'
            result_compare += '</td><td>'
            for f, sets in filesinfo:
                result_compare += f + '<hr id="table_hr">'
            result_compare += '</td></tr>'

        if len(errors) > 0:
            result_compare += '<tr><th scope="row">Errors</th>'
            result_compare += '<td>These images have no tracking dots or could not be extracted.</td><td>'
            for e, f in errors:
                result_compare += f + '<hr id="table_hr">'
            result_compare += '</td></tr>'

        result_compare += '</tbody></table>'
        #send output to javascript
        eel.print_result(info, result_compare)
        return ''

    #---
    #--- Anon Scan
    #---
    @eel.expose
    def anonScanAction(upload, savepath):
        if os.path.isfile(upload):
            filext = os.path.splitext(upload)[1].lower()
            path, filename = os.path.split(upload)
            path = path + '/'
            if(filext=='.jpeg' or filext=='.jpg' or filext=='.png' or filext=='.tiff' or filext=='.bmp'):
                improcess = ImgProcessingMixin(upload)
                improcess._verbosity = 0
                im = cv2.imread(upload)
                _, mask = improcess.processImage(im,workAtDpi=None,halftonesBlur=10,ydColourRange=((16,1,214),(42,120,255)),paperColourThreshold=225)
                im[mask==255] = 255
                if(savepath == ''):
                    output = os.path.splitext(os.path.basename(upload))[0] + '_anon' + os.path.splitext(os.path.basename(upload))[1]
                    save = path + output
                    cv2.imwrite(save, im)
                    info = 'Document anonymized and saved.<br>'
                    result = '<table class="table table-bordered table-sm table-dark table-hover text-center" id="resulttable"><tbody>'
                    result += '<tr><th scope="row">Path</th><td>' + path + '</td></tr>'
                    result += '<tr><th scope="row">Image</th><td>' + filename + '</td></tr>'
                    result += '<tr><th scope="row">Anonymized Image</th><td>' + output + '</td></tr></tbody></table>'
                    #send output to javascript
                    eel.print_result(info, result)
                else:
                    #TODO
                    output = savepath
                return ''
            else:
                return 'Not a valid Image. Currently only jpg, png, tiff or bmp are allowed'
        else:
            return 'Not a valid Path'


    #---
    #--- Generate Print Mask
    #---
    @eel.expose
    def generateMask(upload):
        if os.path.isfile(upload):
            filext = os.path.splitext(upload)[1].lower()
            path = os.path.split(upload)[0] + '/'

            if(filext=='.jpeg' or filext=='.jpg' or filext=='.png' or filext=='.tiff' or filext=='.bmp'):
                try:
                    TestpageParser(upload)()
                    return 'Done. Mask was generated and saved as ' + path + 'mask.json'
                except:
                    return 'An Error occoured.'
            else:
                return 'Not a valid Image. Currently only jpg, png, tiff or bmp are allowed'
        else:
            return 'Not a valid Path'

    #---
    #--- Apply Print Mask
    #---
    @eel.expose
    def applyMask(page, mask, x, y, dotradius):
        if os.path.isfile(page):
            if os.path.isfile(mask):
                pagename, pageext = os.path.splitext(page)
                pageext = pageext.lower()
                pagepath = os.path.split(page)[0] + '/'
                maskext = os.path.splitext(mask)[1].lower()
                if(pageext=='.ps'):
                    if(maskext=='.json'):
                        try:
                            if (x==''): x = None
                            else: x = float(x)
                            if (y==''): y = None
                            else: y = float(y)
                            if (dotradius==''): dotradius = 0.004
                            else: dotradius = float(dotradius)
                            xoff, yoff, dotrad = ApplyMask(page,mask,x,y,dotradius)()
                            text_file = open(pagename + "_masked.txt", "w")
                            text_file.write("X-Offset: %s \ny-Offset: %s \nDotradius: %s" % (xoff, yoff, dotrad))
                            text_file.close()
                            info = 'Done'
                            result = '<table class="table table-bordered table-sm table-dark table-hover text-center" id="resulttable"><tbody>'
                            result += '<tr><th scope="row">Masked Document</th><td>' + pagename + '_masked.ps</td></tr>'
                            result += '<tr><th scope="row">Info File</th><td>' + pagename + '_masked.txt</td></tr>'
                            result += '<tr><th scope="row">x-Offset</th><td>' + str(round(xoff,4)) + '</td></tr>'
                            result += '<tr><th scope="row">y-Offset</th><td>' + str(round(yoff,4)) + '</td></tr>'
                            result += '<tr><th scope="row">Dotradius</th><td>' + str(dotrad) + '</td></tr></tbody></table>'
                            #send output to javascript
                            eel.print_result(result)
                            return info
                        except:
                            return 'An Error occoured.'
                    else:
                        return 'Not a valid Mask File (json).'
                else:
                    return 'Not a valid Document File - please provide a PostScript Document.'
            else:
                return 'Not a valid Path to Mask File.'
        else:
            return 'Not a valid Path to Document File.'


    #---
    #--- Start GUI
    #---
    web_app_options = {
	    'mode': "", #"chrome", #or "chrome-app"
	    'port': 8080,
	    'chromeFlags': ["--size=(1024, 800)"]
    }

    eel.start('index.html', options=web_app_options)


if __name__ == '__main__':
    main()
    

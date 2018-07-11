"""
Speed boost for PageObject.mergePage()

Adapted from https://github.com/Averell7
"""

from PyPDF2 import pdf
from PyPDF2.pdf import *


class PageObject(pdf.PageObject):

    def _addCode(contents, pdf, code, endCode = ""):

        stream = ContentStream(contents, pdf)
        #stream.operations.insert(0, [[], code])
        stream.operations.append([[], code])
        stream.operations.append([[], endCode])
        return stream
    _addCode = staticmethod(_addCode)
    
    # Variant of the mergePage function.
    # Merges the content streams of several pages and code strings into one page.
    # Resource references (i.e. fonts) are maintained from all pages.
    # The parameter ar_data is an array containing code strings and PageObjects.
    # ContentStream is called only if necessary because it calls ParseContentStream
    # which is slox. Otherwise the Content is directly extracted and added to the code.

    def mergePage(self, ar_data ):

        newResources = DictionaryObject()
        rename = {}
        originalResources = self["/Resources"].getObject()       
        code_s = b""
       
        if isinstance(ar_data, PageObject) :
            ar_data = [ar_data]
        strType = type("x")
        for data in ar_data :
            if isinstance(data, PageObject) :       

                # Now we work on merging the resource dictionaries.  This allows us
                # to find out what symbols in the content streams we might need to
                # rename.
                pagexResources = data["/Resources"].getObject()

                for res in "/ExtGState", "/Font", "/XObject", "/ColorSpace", "/Pattern", "/Shading":
                    new, newrename = PageObject._mergeResources(originalResources, pagexResources, res)
                    if new:
                        newResources[NameObject(res)] = new
                        rename.update(newrename)

                # Combine /Resources sets.
                originalResources.update(newResources)

                # Combine /ProcSet sets.
                newResources[NameObject("/ProcSet")] = ArrayObject(
                    frozenset(originalResources.get("/ProcSet", ArrayObject()).getObject()).union(
                        frozenset(pagexResources.get("/ProcSet", ArrayObject()).getObject())
                    )
                )

                if len(rename) > 0 :
                    pagexContent = data['/Contents'].getObject()
                    pagexContent = PageObject._contentStreamRename(pagexContent, rename, self.pdf)
                    code_s += pagexContent.getData() + b"\n"
                else :
                    page_keys = data.keys()
                    if "/Contents" in page_keys :            # if page is not blank
                        code_s += self.extractContent(data["/Contents"]) + b"\n"


            else :
                code_s += data + b"\n"


        originalContent = self.get("/Contents",
            ContentStream(ArrayObject(), self.pdf)
        ).getObject()
        outputContent = PageObject._addCode(originalContent, self.pdf, code_s)

        self[NameObject('/Contents')] = outputContent
        self[NameObject('/Resources')] = originalResources

    def setContent(self, data ):


        newResources = DictionaryObject()
        rename = {}
        #originalResources = self["/Resources"].getObject()
        originalContent = self["/Contents"].getObject()

        stream = ContentStream(originalContent, self.pdf)
        stream.operations = []
        stream.operations.append([[], data])


        self[NameObject('/Contents')] = stream
        #self[NameObject('/Resources')] = originalResources

    def extractContent(self,data) :
        code_s = b""
        pageContent = data.getObject()
        if isinstance(pageContent, ArrayObject) :
            for data2 in pageContent :
                code_s += self.extractContent(data2)
        else :
            if isinstance(data, TextStringObject) :
                code_s += data
            else :
                    decodedData = filters.decodeStreamData(pageContent)
                    code_s += decodedData

        return code_s

    def _mergeResources(res1, res2, resource):
        newRes = DictionaryObject()
        newRes.update(res1.get(resource, DictionaryObject()).getObject())
        page2Res = res2.get(resource, DictionaryObject()).getObject()
        renameRes = {}
        for key in page2Res.keys():
            if key in newRes and ( newRes[key] != page2Res[key]
                                         or resource == "/XObject" ) :
                i = 1
                while True :
                    if (key + "renamed" + str(i)) in newRes:
                        i = i + 1
                    else :
                        newname = NameObject(key + "renamed" + str(i))
                        break

                renameRes[key] = newname
                newRes[newname] = page2Res[key]
            elif not key in newRes:
                newRes[key] = page2Res.raw_get(key)
        return newRes, renameRes
    _mergeResources = staticmethod(_mergeResources)


pdf.PageObject = PageObject


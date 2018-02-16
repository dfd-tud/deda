DEDA Tracking Dots Extraction, Decoding and Anonymisation Toolkit
=================================================================


#### 1. Reading tracking data   

Tracking data can be read and sometimes be decoded from a scanned image.   
`$ ./deda_parse_print.py INPUTFILE`


#### 2. Find a divergent printer in a set of scanned documents   

`$ ./deda_compare_prints.py INPUT1 INPUT2 [INPUT3] ...`


#### 3. Analysing an unknown tracking pattern

New patterns might not be recognised by parse_print. The dots can be extracted
for further analysis.      
`$ ./libdeda/extract_yd.py INPUTFILE`


#### 4. Anonymise a scanned image

This (mostly) removes tracking data from a scan:   
`$ ./deda_clean_document.py INPUTFILE OUTPUTFILE`


#### 5. Anonymise a document for printing

* Save your document as a PS file and call it DOCUMENT.PS. 
PDFs can be converted using pdf2ps:   
`$ pdf2ps INPUT.PDF OUTPUT.PS`  

* Print the testpage.ps file created by    
`$ ./deda_anonmask_create.py -w`   
without any page margin.

* Scan the document and pass the file to   
`$ ./deda_anonmask_create.py -r INPUTFILE`   
This creates 'mask.json', the individual printer's anonymisation mask.   

* Now apply the anonymisation mask:   
`$ ./deda_anonmask_apply.py mask.json DOCUMENT.PS`   
This creates 'masked.ps', the anonymised document. It may be printed with a
zero page margin setting.

Check whether a masked page covers your printer's tracking dots by using a 
microscope. The mask's dot radius, x and y offsets can be customised and 
passed to ./deda_anonmask_apply.py as parameters.


DEDA - tracking Dots Extraction, Decoding and Anonymisation toolkit
=================================================================

Document Colour Tracking Dots, or yellow dots, are small systematic dots which encode information about the printer and/or the printout itself. This process is integrated in almost every commercial colour laser printer. This means that almost every printout contains coded information about the source device, such as the serial number. 

On the one hand, this tool gives the possibility to read out and decode these forensic features and on the other hand, it allows anonymisation to prevent arbitrary tracking.

If you use this software, please cite the paper:
Timo Richter, Stephan Escher, Dagmar Schönfeld, and Thorsten Strufe. 2018. Forensic Analysis and Anonymisation of Printed Documents. In Proceedings of the 6th ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec '18). ACM, New York, NY, USA, 127-138. DOI: https://doi.org/10.1145/3206004.3206019


#### 0. Install

* From PyPI:
`$ pip3 install deda`
* From current directory:
`$ pip3 install .`


#### 1. Reading tracking data   

Tracking data can be read and sometimes be decoded from a scanned image. For good results the input shall use a lossless compression and 300 dpi.
`$ deda_parse_print INPUTFILE`


#### 2. Find a divergent printer in a set of scanned documents   

`$ deda_compare_prints INPUT1 INPUT2 [INPUT3] ...`


#### 3. Analysing an unknown tracking pattern

New patterns might not be recognised by parse_print. The dots can be extracted
for further analysis.      
`$ libdeda/extract_yd.py INPUTFILE`


#### 4. Anonymise a scanned image

This (mostly) removes tracking data from a scan:   
`$ deda_clean_document INPUTFILE OUTPUTFILE`


#### 5. Anonymise a document for printing

* Save your document as a PS file and call it DOCUMENT.PS. 
PDFs can be converted using pdf2ps:   
`$ pdf2ps INPUT.PDF OUTPUT.PS`  

* Print the testpage.ps file created by    
`$ deda_anonmask_create -w`   
without any page margin.

* Scan the document (300 dpi) and pass the lossless file to   
`$ deda_anonmask_create -r INPUTFILE`   
This creates 'mask.json', the individual printer's anonymisation mask.   

* Now apply the anonymisation mask:   
`$ deda_anonmask_apply mask.json DOCUMENT.PS`   
This creates 'masked.ps', the anonymised document. It may be printed with a
zero page margin setting.

Check whether a masked page covers your printer's tracking dots by using a 
microscope. The mask's dot radius, x and y offsets can be customised and 
passed to `deda_anonmask_apply` as parameters.


DEDA - tracking Dots Extraction, Decoding and Anonymisation toolkit
=================================================================

Document Colour Tracking Dots, or yellow dots, are small systematic dots which encode information about the printer and/or the printout itself. This process is integrated in almost every commercial colour laser printer. This means that almost every printout contains coded information about the source device, such as the serial number. 

On the one hand, this tool gives the possibility to read out and decode these forensic features and on the other hand, it allows anonymisation to prevent arbitrary tracking.

If you use this software, please cite the paper:
Timo Richter, Stephan Escher, Dagmar Schönfeld, and Thorsten Strufe. 2018. Forensic Analysis and Anonymisation of Printed Documents. In Proceedings of the 6th ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec '18). ACM, New York, NY, USA, 127-138. DOI: https://doi.org/10.1145/3206004.3206019


----
#### Installation

* Install Python 3
* Install Deda

From PyPI:
`$ pip3 install --user deda`

Or from current directory:
`$ pip3 install --user .`
* Optional requirement by deda_anonmask_apply (Unix and GNU/Linux only):
`$ pip3 install --user wand`

Without Wand, pages containing white areas on images cannot be anonymised.

----
#### Graphical User Interface

* To open the GUI type:
`$ deda_gui`


----
#### Terminal Application


##### 1. Reading tracking data   

Tracking data can be read and sometimes be decoded from a scanned image. For good results the input shall use a lossless compression (e.g. png) and 300 dpi. Make sure to set a neutral contrast 
`$ deda_parse_print INPUTFILE`


##### 2. Find a divergent printer in a set of scanned documents   

`$ deda_compare_prints INPUT1 INPUT2 [INPUT3] ...`


##### 3. Analysing an unknown tracking pattern

New patterns might not be recognised by parse_print. The dots can be extracted
for further analysis.      
`$ deda_extract_yd INPUTFILE`


##### 4. Create your own tracking dots

If you want to create your own tracking dots matrix and add it to a pdf
document, pass the contents as parameters (see `deda_create_dots -h`).
`$ deda_create_dots PDFINPUT`

The calibration page (`$ deda_anonmask_create -w`) may be used as an input.


##### 5. Anonymise a scanned image

This (mostly) removes tracking data from a scan:   
`$ deda_clean_document INPUTFILE OUTPUTFILE`


##### 6. Anonymise a document for printing

* Save your document as a PDF file and call it DOCUMENT.PDF.

* Print the testpage.pdf file created by    
`$ deda_anonmask_create -w`   
without any page margin.

* Scan the document (300 dpi) and pass the lossless file to   
`$ deda_anonmask_create -r INPUTFILE`   
This creates 'mask.json', the individual printer's anonymisation mask.   

* Now apply the anonymisation mask:   
`$ deda_anonmask_apply mask.json DOCUMENT.PDF`
This creates 'masked.pdf', the anonymised document. It may be printed with a
zero page margin setting.

Check whether a masked page covers your printer's tracking dots by using a 
microscope. The mask's dot radius, x and y offsets can be customised and 
passed to `deda_anonmask_apply` as parameters.

Note that if DOCUMENT.PDF contains graphics with white or light coloured parts, these can only be masked if "wand" is installed (see above).


----
#### Troubleshooting

##### deda_parse_print: command not found

Possible solutions:
* Install deda accordig to chapter 0
* Execute
`$ export PATH="$PATH:$(python -c 'import site,os; print(os.path.join(site.USER_BASE, "bin"))')"`


##### Deda does not recognise my tracking dots

Set up your scan program so that it does not eliminate the paper structure nor tracking dots by some threshold and check again. Remember that monochrome pages as well as inkjet prints might not contain tracking dots.


##### My printer does not print tracking dots. Can I hide this fact?

If there are really no tracking dots, you can either create your own ones (`deda_create_dots`) or print the calibration page (`deda_anonmask_create -w`) with another printer and use the mask for your own printer. You can use the anonymised version of the tracking dots or just copy them (`deda_anonmask_create --copy`). See chapters "Anonymise a document for printing" and "Create your own tracking dots".


##### Install Error: command 'x86_64-linux-gnu-gcc' failed with exit status 1

This may be caused by the eel dependency which is needed for the GUI. Try
`$ sudo apt-get install build-essential autoconf libtool pkg-config python3.6-dev gcc && pip3 install --user eel`


##### wand.exceptions.PolicyError: attempt to perform an operation not allowed by the security policy PDF' @ error/constitute.c/IsCoderAuthorized/408

This is being caused by ImageMagick. Either remove Wand (`pip3 uninstall wand`) or add `<policy domain="coder" rights="read | write" pattern="PDF" />` just before `</policymap>` in /etc/ImageMagick-*/policy.xml. See also https://stackoverflow.com/questions/52998331/imagemagick-security-policy-pdf-blocking-conversion.



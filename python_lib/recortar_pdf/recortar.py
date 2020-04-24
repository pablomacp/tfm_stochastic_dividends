# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:30:43 2020

@author: pablo
"""

from PyPDF2 import PdfFileWriter, PdfFileReader

with open("input.pdf", "rb") as in_f:
       input1 = PdfFileReader(in_f)
       output = PdfFileWriter()

       numPages = input1.getNumPages()

#       for i in range(numPages):
#           page = input1.getPage(i)
#           print page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y()
#           page.trimBox.lowerLeft = (25, 25)
#           page.trimBox.upperRight = (225, 225)
#           page.cropBox.lowerLeft = (50, 50)
#           page.cropBox.upperRight = (200, 200)
#           output.addPage(page)

       page = input1.getPage(0)
       print(page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y())
       x0=200
       y0=767
       x1=398
       y1=808
       page.trimBox.lowerLeft = (x0, y0)
       page.trimBox.upperRight = (x1, y1)
       page.cropBox.lowerLeft = (x0, y0)
       page.cropBox.upperRight = (x1, y1)
       output.addPage(page)

       with open("out.pdf", "wb") as out_f:
           output.write(out_f)

# psfile.py - generate encapsulated PostScript files
#
# Copyright (C) 2009  Jochen Voss <voss@seehuhn.de>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""Generate encapsulated PostScript files.

This module helps you to create PostScript files from Python
by creating the required wrappers to set up the page geometry.
You will need to provide the PostScript code to actually
the figure or page you want to create.

Exported classes:

    EPSFile - Encapsulated PostScript figures
    PSFile - Stand-alone PostScript files
"""

import time
try:
    import StringIO
except ImportError:
    import io as StringIO
try: basestring
except NameError: basestring = str

__version__ = '0.9'
__all__ = [ 'EPSFile', 'PSFile', 'paper_sizes' ]

paper_sizes = {
    "A3": (842, 1190),
    "A4": (595, 842),
    "letter": (612, 792),
}

class PSBase(object):

    """Abstract base class for PSFile and EPSFile."""

    def __init__(self, fname, ps_type="",
                 title=None, creator=None,
                 margin=72, margin_top=None, margin_right=None,
                 margin_bottom=None, margin_left=None,
                 padding=0, padding_top=None, padding_right=None,
                 padding_bottom=None, padding_left=None,
                 paper="A4"):
        # PostScript DSC sections
        self.comments = []
        self.prolog = []
        self.setup = []
        self.trailer = []

        self.definitions = []
        self.dict_space = 0     # extra space in the dictionary
        self.body = StringIO.StringIO()

        self.closed = False

        paper_width, paper_height = self._decode_paper_size(paper)
        self.paper_width = paper_width
        self.paper_height = paper_height

        margin_top = margin_top if margin_top is not None else margin
        margin_right = margin_right if margin_right is not None else margin
        margin_bottom = margin_bottom if margin_bottom is not None else margin
        margin_left = margin_left if margin_left is not None else margin

        padding_top = padding_top if padding_top is not None else padding
        padding_right = padding_right if padding_right is not None else padding
        padding_bottom = padding_bottom if padding_bottom is not None else padding
        padding_left = padding_left if padding_left is not None else padding

        if self.xbase is None or self.xbase < margin_left:
            self.xbase = margin_left
        if self.ybase is None or self.ybase < margin_bottom:
            self.ybase = margin_bottom
        if self.width is None:
            self.width = paper_width - margin_left - margin_right
        if self.height is None:
            self.height = paper_height - margin_bottom - margin_top
        bbox = ( self.xbase-padding_left,
                 self.ybase-padding_bottom,
                 self.xbase+self.width+padding_right,
                 self.ybase+self.height+padding_top)
        self.add_dsc_comment("BoundingBox", "%d %d %d %d\n"%bbox)
        self.setup.append("%d %d translate" % (self.xbase, self.ybase))

        if creator is not None:
            self.add_dsc_comment("Creator", creator)
        if title is not None:
            self.add_dsc_comment("Title", title)
        self.add_dsc_comment("CreationDate",
                             time.strftime("%Y-%m-%d %H:%M:%S"))

        # put a claim on the file by opening it
        self.fd = fd = open(fname, "w")
        fd.write("%!PS-Adobe-3.0"+ps_type+"\n")
        fd.flush()

    def _decode_paper_size(self, paper):
        if isinstance(paper, basestring) and paper.endswith("*"):
            flip = True
            paper = paper[:-1]
        else:
            flip = False

        if paper in paper_sizes:
            paper_width, paper_height = paper_sizes[paper]
        elif len(paper) == 2:
            paper_width, paper_height = paper
        else:
            raise ValueError("invalid paper size %s"%repr(paper))
        if flip:
            paper_width, paper_height = paper_height, paper_width
        return paper_width, paper_height

    def add_dsc_comment(self, key, value):
        """Add a PostScript DSC comment to the file header."""
        self.comments.append((key, value.strip()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def define(self, name, body):
        """Define a PostScript macro.

        This adds a PostScript macro to the header of the generated
        file, making `name` an abbreviation for `body`.  This can be
        used to shrink the size of the generated PostScript file.
        """
        if self.closed:
            return ValueError("PostScript file already closed")
        self.definitions.append((name,body))

    def write(self, text):
        """Append `text` to the body of the PostScript file.

        This method appends the string `text` to the body
        of the PostScript file unchanged.
        """
        if self.closed:
            return ValueError("PostScript file already closed")
        self.body.write(text)

    def append(self, text):
        """Append a block of text to the PostScript file.

        This method sanitises white space in `text` (removes
        leading and trailing empty lines, expands tabs, removes
        indentation, adds a trailing newline character as needed),
        and then appends the result to the body of the PostScript
        file.
        """
        if self.closed:
            return ValueError("PostScript file already closed")

        ll = [l.rstrip().expandtabs() for l in text.splitlines()]

        first_is_special = True
        while ll and ll[0] == "":
            del ll[0]
            first_is_special = False
        while ll and ll[-1] == "":
            del ll[-1]
        if not ll:
            return
        if len(ll) == 1:
            first_is_special = False

        if first_is_special and ll[0][0] != " ":
            l1 = ll[1]
            n1 = len(l1)-len(l1.lstrip())
            ll[0] = " "*n1+ll[0]
        n = min([len(l)-len(l.lstrip()) for l in ll if l!=""])
        for l in ll:
            if not l:
                continue
            self.body.write(l[n:].rstrip()+"\n")

    def _close_hook(self):
        pass

    def _write_comments(self):
        fd = self.fd
        for key, value in self.comments:
            fd.write("%%"+key+": "+value+"\n")
        fd.write("%%EndComments\n")

    def _write_prolog(self):
        if not self.prolog:
            return
        fd = self.fd
        fd.write("%%BeginProlog\n")
        for l in self.prolog:
            fd.write(l.strip()+"\n")
        fd.write("%%EndProlog\n")

    def _write_setup(self):
        if not self.setup:
            return
        fd = self.fd
        fd.write("%%BeginSetup\n")
        for l in self.setup:
            fd.write(l.strip()+"\n")
        fd.write("%%EndSetup\n")

    def _write_definitions(self):
        dictlen = len(self.definitions) + self.dict_space
        if dictlen == 0:
            return
        fd = self.fd
        fd.write("%d dict begin\n"%dictlen)
        self.trailer.insert(0, "end")
        for name, body in self.definitions:
            ll = [l.rstrip().expandtabs() for l in body.splitlines()]
            first_is_special = True
            while ll and ll[0] == "":
                del ll[0]
                first_is_special = False
            while ll and ll[-1] == "":
                del ll[-1]
            if len(ll) > 1:
                fd.write("/%s {\n"%name)
                if first_is_special and ll[0][0] != " ":
                    l1 = ll[1]
                    n1 = len(l1)-len(l1.lstrip())
                    ll[0] = " "*n1+ll[0]
                n = min([len(l)-len(l.lstrip()) for l in ll if l!=""])
                for l in ll:
                    fd.write("  "+l[n:].rstrip()+"\n")
                fd.write("} bind def\n")
            else:
                fd.write("/%s { %s } bind def\n"%(name, body.strip()))

    def _write_body(self):
        self.fd.write(self.body.getvalue())
        self.fd.write("showpage\n")

    def _write_trailer(self):
        if not self.trailer:
            return
        fd = self.fd
        fd.write("%%BeginTrailer\n")
        for l in self.trailer:
            fd.write(l.strip()+"\n")
        self.fd.write("%%EOF\n")

    def close(self):
        """Close the PostScript file.

        This method writes the PostScript code to the output file
        and then closes the PostScript file.  A closed PostScript file
        cannot be written to any more.
        """
        if self.closed:
            return
        self._close_hook()
        self.closed = True

        self._write_comments()
        self._write_prolog()
        self._write_setup()
        self._write_definitions()
        self._write_body()
        self._write_trailer()
        self.fd.close()

        self.body.close()

class PSFile(PSBase):

    """Stand-alone PostScript files.

    Instances of this class are file-like objects.  Headers and
    trailers are automatically generated, you only need to add the
    PostScript code to put the marks onto the page.
    """

    def __init__(self, fname,
                 title=None, creator=None,
                 margin=72, margin_top=None, margin_right=None,
                 margin_bottom=None, margin_left=None,
                 paper="A4"):
        """Create a new PostScript document.

        The plotting region is a page of the given dimensions (A4
        paper by default), minus the given margins.  The size of the
        margin is determined by the arguments `margin_top`,
        `margin_right`, `margin_bottom` and `margin_left`, or by
        `margin` if the individual values are not set.  (All lengths
        are given in PostScript points, i.e. 72 units correspond to
        one inch.)  The coordinate systems is adjusted so that the
        lower left corner of the drawing area has coordinates (0,0).

        The arguments `title` and `creator` can be used to set the
        corresponding PostScript header comments.
        """

        self.xbase = margin_left
        self.ybase = margin_bottom
        self.width = None
        self.height = None

        PSBase.__init__(self, fname, title=title, creator=creator,
                        margin=margin, margin_top=margin_top,
                        margin_right=margin_right, margin_bottom=margin_bottom,
                        margin_left=margin_left,
                        paper=paper)

        w = self.paper_width
        h = self.paper_height
        for name, dim in paper_sizes.items():
            if dim == (w,h):
                paper = name
                orientation = "Portrait"
                break
            elif dim == (h,w):
                w, h = h, w
                paper = name
                orientation = "Landscape"
                break
        else:
            paper = "custom"
            orientation = None
        self.add_dsc_comment("DocumentMedia",
                             "%s %d %d 0 ( ) ( )"%(name, w, h))
        if orientation is not None:
            self.add_dsc_comment("Orientation", orientation)
        self.setup.insert(0, "<< /PageSize [ %d %d ] >> setpagedevice"%(w,h))
        if orientation == "Landscape":
            self.setup.insert(1, "%d 0 translate 90 rotate"%w)

        self.dict_space += 1    # space for pgsave in `write_body`

    def _close_hook(self):
        self.add_dsc_comment("Pages", "1")

    def _write_body(self):
        fd = self.fd
        fd.write("%%Page: 1 1\n")
        fd.write("/pgsave save def\n")
        fd.write(self.body.getvalue())
        fd.write("pgsave restore\n")
        fd.write("showpage\n")

class EPSFile(PSBase):

    """Encapsulated PostScript figures.

    Instances of this class are file-like objects.  Headers and
    trailers are automatically generated, you only need to add the
    PostScript code to put the marks onto the page.
    """

    def __init__(self, fname, width, height,
                 title=None, creator=None,
                 margin=3, margin_top=None, margin_right=None,
                 margin_bottom=None, margin_left=None,
                 paper="A4"):
        """Create a new PostScript Figure for inclusion in other documents.

        The figure's drawing area has width `width` and height
        `height`.  (All lengths are given in PostScript points,
        i.e. 72 units correspond to one inch.)  The coordinate system
        is adjusted so that the lower left corner of the drawing area
        has coordinates (0,0).

        The arguments `title` and `creator` can be used to set the
        corresponding PostScript header comments.

        A bounding box for the drawing area is computed and placed in
        the PostScript file header.  The bounding box covers the
        drawing area and an additional margin.  The size of the margin
        is determined by the arguments `margin_top`, `margin_right`,
        `margin_bottom` and `margin_left`, or by `margin` if the
        individual values are not set.
        """

        paper_width, paper_height = self._decode_paper_size(paper)
        self.xbase = (paper_width - width) // 2
        self.ybase = (paper_height - height) // 2
        self.width = width
        self.height = height

        PSBase.__init__(self, fname, ps_type=" EPSF-3.0",
                        title=title, creator=creator,
                        padding=margin,
                        padding_top=margin_top,
                        padding_right=margin_right,
                        padding_bottom=margin_bottom,
                        padding_left=margin_left,
                        paper=paper)

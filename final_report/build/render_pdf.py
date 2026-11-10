"""PDF renderer (reportlab) — one content stream, academic layout.

Fonts: Times New Roman (body/headings), STIX General (glyph fallback for the
6 math characters Times lacks), DejaVu Sans Mono (code).
Layout: A4, 22 mm side margins (166 mm frame), navy headings, blue accent,
captions below figures / above tables, 2-pass TOC + LOF + LOT with page
numbers, PDF outline bookmarks.
"""
import os
import re
import html
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                               Spacer, PageBreak, Image, Table, TableStyle,
                               Preformatted, KeepTogether, HRFlowable)
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

from blocks import FIGS, fig_caption_text

# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------
NAVY = HexColor("#1F2A44")
ACCENT = HexColor("#4C72B0")
LIGHT = HexColor("#EAF0F8")
CAPGRAY = HexColor("#555555")
GRIDC = HexColor("#AAB4C0")
CODEBG = HexColor("#F5F7FA")

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
_TNR_DIR = "/System/Library/Fonts/Supplemental"
_MPL_TTF = None
try:
    import matplotlib
    _MPL_TTF = os.path.join(os.path.dirname(matplotlib.__file__),
                            "mpl-data", "fonts", "ttf")
except Exception:
    pass


def _register_fonts():
    faces = [
        ("Serif", os.path.join(_TNR_DIR, "Times New Roman.ttf")),
        ("Serif-Bold", os.path.join(_TNR_DIR, "Times New Roman Bold.ttf")),
        ("Serif-Italic", os.path.join(_TNR_DIR, "Times New Roman Italic.ttf")),
        ("Serif-BoldItalic",
         os.path.join(_TNR_DIR, "Times New Roman Bold Italic.ttf")),
        ("Mono", os.path.join(_MPL_TTF, "DejaVuSansMono.ttf")),
        ("Mono-Bold", os.path.join(_MPL_TTF, "DejaVuSansMono-Bold.ttf")),
        ("Mono-Italic", os.path.join(_MPL_TTF, "DejaVuSansMono-Oblique.ttf")),
        ("Mono-BoldItalic",
         os.path.join(_MPL_TTF, "DejaVuSansMono-BoldOblique.ttf")),
        ("STIX", os.path.join(_MPL_TTF, "STIXGeneral.ttf")),
        ("STIX-Bold", os.path.join(_MPL_TTF, "STIXGeneralBol.ttf")),
        ("STIX-Italic", os.path.join(_MPL_TTF, "STIXGeneralItalic.ttf")),
        ("STIX-BoldItalic", os.path.join(_MPL_TTF, "STIXGeneralBolIta.ttf")),
    ]
    for name, path in faces:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(
        "Serif", normal="Serif", bold="Serif-Bold",
        italic="Serif-Italic", boldItalic="Serif-BoldItalic")
    pdfmetrics.registerFontFamily(
        "Mono", normal="Mono", bold="Mono-Bold",
        italic="Mono-Italic", boldItalic="Mono-BoldItalic")
    pdfmetrics.registerFontFamily(
        "STIX", normal="STIX", bold="STIX-Bold",
        italic="STIX-Italic", boldItalic="STIX-BoldItalic")


_register_fonts()

# characters Times New Roman lacks (checked against the content at build time)
_TIMES_CMAP = None
_T_FALLBACK = None


def _fallback_chars():
    global _TIMES_CMAP, _T_FALLBACK
    if _TIMES_CMAP is None:
        from fontTools.ttLib import TTFont as _TT
        f = _TT(os.path.join(_TNR_DIR, "Times New Roman.ttf"), fontNumber=0)
        cmap = set()
        for t in f["cmap"].tables:
            cmap |= set(t.cmap.keys())
        _TIMES_CMAP = cmap
        _T_FALLBACK = set()
    return _TIMES_CMAP


def _stix_face(bold, ital):
    if bold and ital:
        return "STIX-BoldItalic"
    if bold:
        return "STIX-Bold"
    if ital:
        return "STIX-Italic"
    return "STIX"


# --------------------------------------------------------------------------
# Inline markup -> reportlab Paragraph markup
# --------------------------------------------------------------------------
_TAG_RE = re.compile(
    r"<(/?)(b|i|code|sub|sup|br)(/?)\s*>", re.I)
_ENTITY_RE = re.compile(r"&(?!#?[a-zA-Z]+;)")


def _closer(tag):
    return {"b": "</b>", "i": "</i>", "code": "</font>"}[tag]


def _conv_run(text, bold, ital):
    """Escape a literal text run, wrapping Times-missing glyphs in STIX."""
    if not text:
        return ""
    text = _ENTITY_RE.sub("&amp;", text)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    cmap = _fallback_chars()
    out = []
    face = None
    for ch in text:
        if ord(ch) > 127 and ord(ch) not in cmap:
            if face is None:
                face = _stix_face(bold, ital)
                out.append('<font face="%s">' % face)
            out.append(ch)
        else:
            if face is not None:
                out.append("</font>")
                face = None
            out.append(ch)
    if face is not None:
        out.append("</font>")
    return "".join(out)


def inline(text, doc=None):
    """Convert content inline markup to reportlab Paragraph markup.

    Unclosed <b>/<i>/<code> tags are auto-balanced at the end of the run and
    stray closing tags are dropped, so a content typo cannot abort a build.
    """
    if doc is not None:
        text = doc.resolve(text)
    out = []
    stack = []                      # open tags in order
    pos = 0
    for m in _TAG_RE.finditer(text):
        out.append(_conv_run(text[pos:m.start()],
                             "b" in stack, "i" in stack))
        closing, name = m.group(1), m.group(2).lower()
        if name in ("b", "i", "code"):
            if closing:
                while stack and stack[-1] != name:      # rebalance nesting
                    t = stack.pop()
                    out.append(_closer(t))
                if stack:
                    stack.pop()
                    out.append(_closer(name))
            else:
                stack.append(name)
                out.append("<b>" if name == "b" else
                           "<i>" if name == "i" else
                           '<font face="Mono" size="9.5">')
        else:  # sub / sup / br pass through
            out.append(m.group(0))
        pos = m.end()
    out.append(_conv_run(text[pos:], "b" in stack, "i" in stack))
    for tag in reversed(stack):     # auto-close anything still open
        out.append("</b>" if tag == "b" else
                   "</i>" if tag == "i" else "</font>")
    return "".join(out)


def plain(text, doc=None):
    """Tag-free text (for TOC entries, outlines)."""
    if doc is not None:
        text = doc.resolve(text)
    text = _TAG_RE.sub("", text)
    return html.unescape(text)


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------
def _styles():
    S = {}
    S["Body"] = ParagraphStyle(
        "Body", fontName="Serif", fontSize=11, leading=14.8,
        alignment=TA_JUSTIFY, spaceAfter=6, textColor=colors.black)
    S["Abstract"] = ParagraphStyle(
        "Abstract", parent=S["Body"], fontSize=10.5, leading=14.2,
        leftIndent=10, rightIndent=10, spaceAfter=8)
    S["H1"] = ParagraphStyle(
        "H1", fontName="Serif-Bold", fontSize=14.5, leading=18,
        textColor=NAVY, spaceBefore=16, spaceAfter=5, keepWithNext=True)
    S["H2"] = ParagraphStyle(
        "H2", fontName="Serif-Bold", fontSize=12, leading=15.5,
        textColor=NAVY, spaceBefore=13, spaceAfter=4, keepWithNext=True)
    S["H3"] = ParagraphStyle(
        "H3", fontName="Serif-BoldItalic", fontSize=11, leading=14.5,
        textColor=HexColor("#2E5680"), spaceBefore=11,
        spaceAfter=3, keepWithNext=True)
    S["Bullet"] = ParagraphStyle(
        "Bullet", parent=S["Body"], leftIndent=17, bulletIndent=4,
        spaceAfter=3.5)
    S["FigCaption"] = ParagraphStyle(
        "FigCaption", fontName="Serif", fontSize=9.5, leading=12.5,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=12, textColor=CAPGRAY)
    S["FigCaptionR"] = ParagraphStyle(
        "FigCaptionR", parent=S["FigCaption"], fontSize=9, spaceAfter=10)
    S["TabCaption"] = ParagraphStyle(
        "TabCaption", fontName="Serif", fontSize=9.5, leading=12.5,
        alignment=TA_CENTER, spaceBefore=10, spaceAfter=4, textColor=CAPGRAY,
        keepWithNext=True)
    S["ListTabCaption"] = ParagraphStyle(
        "ListTabCaption", parent=S["TabCaption"], fontName="Serif-Italic",
        alignment=TA_LEFT, spaceBefore=8)
    S["Cell"] = ParagraphStyle(
        "Cell", fontName="Serif", fontSize=8.6, leading=11,
        alignment=TA_CENTER, textColor=colors.black)
    S["CellL"] = ParagraphStyle(
        "CellL", parent=S["Cell"], alignment=TA_LEFT)
    S["CellH"] = ParagraphStyle(
        "CellH", parent=S["Cell"], fontName="Serif-Bold", textColor=NAVY)
    S["CellHL"] = ParagraphStyle(
        "CellHL", parent=S["CellH"], alignment=TA_LEFT)
    S["Eq"] = ParagraphStyle(
        "Eq", fontName="Serif-Italic", fontSize=11, leading=15,
        alignment=TA_CENTER)
    S["EqNum"] = ParagraphStyle(
        "EqNum", fontName="Serif", fontSize=11, leading=15,
        alignment=TA_LEFT, textColor=CAPGRAY)
    S["Ref"] = ParagraphStyle(
        "Ref", fontName="Serif", fontSize=9.5, leading=12.6,
        leftIndent=26, firstLineIndent=-26, spaceAfter=3.5, alignment=TA_LEFT)
    S["Subtitle"] = ParagraphStyle(
        "Subtitle", fontName="Serif-Italic", fontSize=13, leading=17,
        alignment=TA_CENTER, textColor=ACCENT, spaceBefore=4, spaceAfter=6)
    S["Authors"] = ParagraphStyle(
        "Authors", fontName="Serif", fontSize=10.5, leading=14,
        alignment=TA_CENTER, textColor=NAVY, spaceAfter=4)
    # cover
    S["CoverTitle"] = ParagraphStyle(
        "CoverTitle", fontName="Serif-Bold", fontSize=27, leading=33,
        alignment=TA_CENTER, textColor=NAVY)
    S["CoverSub"] = ParagraphStyle(
        "CoverSub", fontName="Serif", fontSize=12, leading=16,
        alignment=TA_CENTER, textColor=ACCENT)
    S["CoverAuthors"] = ParagraphStyle(
        "CoverAuthors", fontName="Serif-Bold", fontSize=13.5, leading=18,
        alignment=TA_CENTER, textColor=colors.black)
    S["CoverMeta"] = ParagraphStyle(
        "CoverMeta", fontName="Serif", fontSize=11, leading=15,
        alignment=TA_CENTER, textColor=CAPGRAY)
    # part dividers
    S["PartLabel"] = ParagraphStyle(
        "PartLabel", fontName="Serif", fontSize=13, leading=16,
        alignment=TA_CENTER, textColor=ACCENT, spaceAfter=6)
    S["PartTitle"] = ParagraphStyle(
        "PartTitle", fontName="Serif-Bold", fontSize=25, leading=31,
        alignment=TA_CENTER, textColor=NAVY, spaceAfter=10)
    S["PartSub"] = ParagraphStyle(
        "PartSub", fontName="Serif-Italic", fontSize=12.5, leading=17,
        alignment=TA_CENTER, textColor=CAPGRAY, spaceBefore=8)
    # TOC levels (4: part, h1, h2, h3)
    S["TOC0"] = ParagraphStyle(
        "TOC0", fontName="Serif-Bold", fontSize=11.5, leading=16,
        spaceBefore=6, textColor=NAVY)
    S["TOC1"] = ParagraphStyle(
        "TOC1", fontName="Serif", fontSize=11, leading=15, leftIndent=14)
    S["TOC2"] = ParagraphStyle(
        "TOC2", fontName="Serif", fontSize=10.5, leading=14, leftIndent=30)
    S["TOC3"] = ParagraphStyle(
        "TOC3", fontName="Serif-Italic", fontSize=10, leading=13.5,
        leftIndent=46, textColor=CAPGRAY)
    S["TOCL"] = ParagraphStyle(
        "TOCL", fontName="Serif", fontSize=10, leading=13.5, leftIndent=0)
    return S


STYLES = _styles()

FRAME_W = A4[0] - 44 * mm          # 166 mm usable
FRAME_H = A4[1] - 39 * mm

REPORT_TITLE = ("Classification of Human-Written and AI-Generated Text — "
                "Consolidated Project Report")


# --------------------------------------------------------------------------
# Document template with TOC/LOF/LOT notifications + outline bookmarks
# --------------------------------------------------------------------------
class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        self._sec = 0

    def beforeDocument(self):
        self._sec = 0
        self._seen_part = False

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        txt = flowable.getPlainText()
        if name in ("H1", "H2", "H3", "PartTitle"):
            if name == "PartTitle":
                self._seen_part = True
            toc_lvl = {"PartTitle": 0, "H1": 1, "H2": 2, "H3": 3}[name]
            if self._seen_part:
                olvl = toc_lvl
            else:                      # front matter: h1 starts at 0
                olvl = max(0, toc_lvl - 1)
            self._sec += 1
            key = "sec%d" % self._sec
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(plain(txt)[:110], key,
                                      level=olvl, closed=(olvl == 0))
            self.notify("TOCEntry", (toc_lvl, txt, self.page, key))
        elif name == "FigCaption":
            self.notify("LOFEntry", (0, txt, self.page, None))
        elif name == "TabCaption":
            self.notify("LOTEntry", (0, txt, self.page, None))


# --------------------------------------------------------------------------
# Page decoration
# --------------------------------------------------------------------------
def _decorate(canvas, doc):
    page = canvas.getPageNumber()
    canvas.saveState()
    canvas.setTitle("Classification of Human-Written and AI-Generated Text — "
                    "Consolidated Project Report")
    canvas.setAuthor("Pravesh Gedon Sharma; Ankit Rawat")
    canvas.setSubject("AI-generated text detection; mixture of experts; TF-IDF")
    if page > 1:
        canvas.setFont("Serif-Italic", 8)
        canvas.setFillColor(HexColor("#777777"))
        canvas.drawString(22 * mm, A4[1] - 13 * mm,
                          "Human vs AI Text Classification — Consolidated "
                          "Project Report (NIT Sikkim, 2026)")
        canvas.setStrokeColor(HexColor("#CCCCCC"))
        canvas.setLineWidth(0.4)
        canvas.line(22 * mm, A4[1] - 15 * mm, A4[0] - 22 * mm,
                    A4[1] - 15 * mm)
        canvas.setFont("Serif", 9)
        canvas.drawCentredString(A4[0] / 2.0, 10 * mm, str(page))
        canvas.setStrokeColor(HexColor("#CCCCCC"))
        canvas.line(22 * mm, 14 * mm, A4[0] - 22 * mm, 14 * mm)
    canvas.restoreState()


# --------------------------------------------------------------------------
# Figure / table helpers
# --------------------------------------------------------------------------
def _figure_flowables(doc, fig_id, width_frac, repeat):
    info = FIGS[fig_id]
    path = info["file"]
    iw, ih = PILImage.open(path).size
    w = FRAME_W * width_frac
    h = w * ih / float(iw)
    maxh = 210 * mm
    if h > maxh:
        h = maxh
        w = h * iw / float(ih)
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    img.keepWithNext = True
    n = doc.fig_no[fig_id]
    if repeat:
        cap = fig_caption_text(doc, fig_id, repeat_of=doc.fig_section[fig_id])
        style = "FigCaptionR"
    else:
        cap = fig_caption_text(doc, fig_id)
        style = "FigCaption"
    head, _, tail = cap.partition(":")
    para = Paragraph(
        '<b><font color="#1F2A44">%s:</font></b>%s'
        % (plain(head), inline(tail, doc)), STYLES[style])
    return [img, para]


def _table_flowables(doc, tab_id, caption, header, rows, widths_mm,
                     left_cols, style_cap="TabCaption", numbered=True):
    n = doc.tab_no.get(tab_id)
    if numbered:
        head = '<b><font color="#1F2A44">Table %d:</font></b> %s' % (
            n, inline(caption, doc))
        cap_style = STYLES[style_cap]
    else:
        head = inline(caption, doc)
        cap_style = STYLES["ListTabCaption"]
    flow = [Paragraph(head, cap_style)]
    ncols = len(header)
    data = []
    hdr = []
    for j, cell in enumerate(header):
        st = "CellHL" if j in left_cols else "CellH"
        hdr.append(Paragraph("<b>%s</b>" % inline(str(cell), doc),
                             STYLES[st]))
    data.append(hdr)
    for row in rows:
        r = []
        for j, cell in enumerate(row):
            st = "CellL" if j in left_cols else "Cell"
            r.append(Paragraph(inline(str(cell), doc), STYLES[st]))
        data.append(r)
    total = float(sum(widths_mm))
    if total > 166.0:                       # keep inside the 166 mm frame
        widths_mm = [w * 166.0 / total for w in widths_mm]
    colw = [w * mm for w in widths_mm]
    t = Table(data, colWidths=colw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, GRIDC),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white,
                                              HexColor("#FAFBFC")]),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 4 * mm))
    return flow


_CODE_FS_CACHE = None


def _code_flowables(text):
    text = textwrap.dedent(text).strip("\n")
    text = html.unescape(text)
    lines = text.split("\n")
    maxlen = max((len(l) for l in lines), default=1)
    fs = min(7.2, FRAME_W / max(1, maxlen * 0.602))
    fs = max(5.5, fs)
    style = ParagraphStyle("Code", fontName="Mono", fontSize=fs,
                           leading=fs * 1.32, textColor=colors.black)
    pf = Preformatted(text, style)
    box = Table([[pf]], colWidths=[FRAME_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.4, HexColor("#C8D2DE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [box, Spacer(1, 3 * mm)]


# --------------------------------------------------------------------------
# TOC flowables
# --------------------------------------------------------------------------
def make_tocs():
    toc = TableOfContents(notifyKind="TOCEntry", dotsMinLevel=1,
                          levelStyles=[STYLES["TOC0"], STYLES["TOC1"],
                                       STYLES["TOC2"], STYLES["TOC3"]],
                          rightColumnWidth=36)
    lof = TableOfContents(notifyKind="LOFEntry", dotsMinLevel=0,
                          levelStyles=[STYLES["TOCL"]],
                          rightColumnWidth=36)
    lot = TableOfContents(notifyKind="LOTEntry", dotsMinLevel=0,
                          levelStyles=[STYLES["TOCL"]],
                          rightColumnWidth=36)
    return {"toc": toc, "lof": lof, "lot": lot}


# --------------------------------------------------------------------------
# Story builder
# --------------------------------------------------------------------------
def _cover_story():
    from content_front import TITLE, SUBTITLE, AUTHORS, INSTITUTION, COURSE
    authors = "  &amp;  ".join(
        "<b>%s (%s)</b>" % (n, i) for n, i in AUTHORS)
    s = STYLES
    return [
        Spacer(1, 38 * mm),
        Paragraph(inline(TITLE.replace("\n", "<br/>")), s["CoverTitle"]),
        Spacer(1, 7 * mm),
        HRFlowable(width="55%", thickness=1.2, color=ACCENT, hAlign="CENTER",
                   spaceBefore=2, spaceAfter=7),
        Paragraph(inline(SUBTITLE), s["CoverSub"]),
        Spacer(1, 30 * mm),
        Paragraph(authors, s["CoverAuthors"]),
        Spacer(1, 8 * mm),
        Paragraph(inline(INSTITUTION.replace("\n", "<br/>")), s["CoverMeta"]),
        Spacer(1, 6 * mm),
        Paragraph(inline(COURSE), s["CoverMeta"]),
        Spacer(1, 70 * mm),
        Paragraph("Compiled from the project's verified metrics, result CSVs "
                  "and figures — see Figure_Inventory.md and Quality_Audit.md "
                  "for provenance.", STYLES["CoverMeta"]),
        PageBreak(),
    ]


def _part_story(title, subtitle):
    flow = [PageBreak(), Spacer(1, 52 * mm)]
    flow.append(Paragraph(inline(title), STYLES["PartTitle"]))
    flow.append(HRFlowable(width="50%", thickness=1.1, color=ACCENT,
                           hAlign="CENTER", spaceBefore=4, spaceAfter=2))
    if subtitle:
        flow.append(Paragraph(inline(subtitle), STYLES["PartSub"]))
    flow.append(PageBreak())
    return flow


def build_story(doc, tocs):
    story = []
    eq_no = [0]
    S = STYLES
    for b in doc.blocks:
        kind = b[0]
        if kind == "cover":
            story.extend(_cover_story())
        elif kind == "part":
            story.extend(_part_story(b[1], b[2]))
        elif kind == "h1":
            story.append(Paragraph(inline(b[1], doc), S["H1"]))
            story.append(HRFlowable(width="100%", thickness=0.7,
                                    color=ACCENT, spaceBefore=0,
                                    spaceAfter=6))
        elif kind == "h2":
            story.append(Paragraph(inline(b[1], doc), S["H2"]))
        elif kind == "h3":
            story.append(Paragraph(inline(b[1], doc), S["H3"]))
        elif kind in ("p",):
            story.append(Paragraph(inline(b[1], doc), S["Body"]))
        elif kind == "abstract":
            story.append(Paragraph(inline(b[1], doc), S["Abstract"]))
        elif kind == "bullets":
            for item in b[1]:
                story.append(Paragraph(inline(item, doc), S["Bullet"],
                                       bulletText="\u2022"))
        elif kind == "numbered":
            for i, item in enumerate(b[1], 1):
                story.append(Paragraph(inline(item, doc), S["Bullet"],
                                       bulletText="%d." % i))
        elif kind == "fig":
            story.extend(_figure_flowables(doc, b[1], b[2], b[3]))
        elif kind == "table":
            story.extend(_table_flowables(doc, b[1], b[2], b[3], b[4], b[5],
                                          b[6]))
        elif kind == "listtable":
            story.extend(_table_flowables(doc, None, b[1], b[2], b[3], b[4],
                                          b[5], numbered=False))
        elif kind == "code":
            story.extend(_code_flowables(b[1]))
        elif kind == "eq":
            eq_no[0] += 1
            left = Paragraph(inline(b[1], doc), S["Eq"])
            right = Paragraph("(%d)" % eq_no[0], S["EqNum"])
            t = Table([[left, right]],
                      colWidths=[FRAME_W - 28 * mm, 28 * mm])
            t.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
        elif kind == "refs":
            for item in b[1]:
                story.append(Paragraph(inline(item, doc), S["Ref"]))
        elif kind == "subtitle":
            story.append(Paragraph(inline(b[1], doc), S["Subtitle"]))
        elif kind == "authors":
            story.append(Paragraph(inline(b[1], doc), S["Authors"]))
        elif kind == "pagebreak":
            story.append(PageBreak())
        elif kind == "spacer":
            story.append(Spacer(1, b[1] * mm))
        elif kind == "toc":
            story.append(tocs[b[1]])
        else:
            raise ValueError("unknown block kind: %r" % (b,))
    return story


def render(doc, outpath):
    tocs = make_tocs()
    story = build_story(doc, tocs)
    rdoc = ReportDoc(outpath, pagesize=A4,
                     leftMargin=22 * mm, rightMargin=22 * mm,
                     topMargin=21 * mm, bottomMargin=18 * mm)
    frame = Frame(22 * mm, 18 * mm, FRAME_W, FRAME_H, id="main",
                  leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    rdoc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                        onPage=_decorate)])
    rdoc.multiBuild(story)
    return outpath

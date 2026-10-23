"""
Struct with methods converting converter_json's json into
a form better suited for linguistics and LLM
"""

import logging
import re
import statistics
import unicodedata

logger = logging.getLogger(__name__)


from analysis.extraction.linguistics_conversion.schema import (
    FinalDocument,
    ParagraphBlock,
    ListBlock,
    ListItem,
    WordInfo,
    VisualElement,
    FloatingElements,
    ReferenceSections,
    classify_block_content,
    strip_list_marker,
    AcronymItem,
)

from analysis.extraction.main_extractor import (
    DocumentData,
    extractPDF,
    GeometryClassifier
)
from analysis.extraction.linguistics_conversion.schema import (
    PageArtifact,
    is_acronym,
    is_index,
    is_widow_func,
    is_bekart_func,
    is_szewc_func,
)

from dataclasses import dataclass

@dataclass
class LineGeom:
    """
    Geometric layout properties of a single rendered text line,
    used for paragraph break detection and list continuation heuristics.
    """
    x0: float
    x1: float
    y0: float
    y1: float
    first_word_w: float
    pitch: float | None
    ends_hyphen: bool
    font_size: float | None = None


class PDFMapper:
    USE_GEOMETRY_HEURISTICS = True
    USE_NESTED_LISTS = True

    EQ_NUM_RE = re.compile(r"^\(\s*\d+(?:[.,]\s*\d+)*\s*\)$")
    MATH_FONTS = ("math", "cmmi", "cmr", "cmsy", "cmex", "symbol", "msbm")
    MONOSPACE_FONTS = ("courier", "consolas", "mono", "code", "typewriter")
    IMG_PATTERN = re.compile(
        r"^(rysunek|rys\.|fot\.|schemat|figure|fig\.|image|scheme|diagram)\s*(?:\d+|[IVX]+)",
        re.IGNORECASE,
    )
    TAB_PATTERN = re.compile(
        r"^(tabela|tab\.|table)\s*(?:\d+|[IVX]+)", re.IGNORECASE
    )

    def _use_geom(self) -> bool:
        """
        Determines whether geometry-based heuristics should be applied.
        Disabled within the bibliography section.
        """
        return self.USE_GEOMETRY_HEURISTICS and not getattr(self, "in_bibliography_section", False)
    
    PITCH_TOL = 0.15
    SPACE_W = 3.0
    EARLY_SLACK = 4.0
    
    TOP_MARGIN_THRESH = 70
    BOTTOM_MARGIN_OFFSET = 75
    MARGIN_INDENT_THRESH = 20
    MIN_VERTICAL_GAP = 11.5
    ACRONYM_THRESH = 0.6
    LIST_CONT_MAX_X_DIFF = 10
    TABLE_INTERSECT_THRESH = 0.8
    MIN_VERTICAL_GAP_LIST = 50

    BIB_HEADINGS = (
        "BIBLIOGRAFIA",
        "LITERATURA",
        "REFERENCES",
        "WYKAZ LITERATURY",
        "PIŚMIENNICTWO",
        "ŹRÓDŁA",
        "BIBLIOGRAPHY",
        "WORKS CITED",
        "CITED REFERENCES",
        "SPIS LITERATURY",
        "WYKAZ BIBLIOGRAFICZNY",
        "WYKAZ ŹRÓDEŁ",
        "LITERATURE",
    )
    AFTER_BIB_HEADING_RE = re.compile(
        r"^(DODATEK\b|ZAŁĄCZNIK\b|ZALACZNIK\b|APPENDIX\b|ANEKS\b|SPIS\s|WYKAZ\s|LIST\b|OŚWIADCZENIE\b|OSWIADCZENIE\b|ABSTRACT\b|STRESZCZENIE\b)", re.IGNORECASE
    )
    HEADING_NUM_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.|[IVXLC]+\.)\s+")
    SPLIT_POLISH_SIGNS_RE = re.compile(
        r"(?:(?<=[a-ząćęłńóśźż])|(?<=(?<!\S)[A-ZĄĆĘŁŃÓŚŹŻ])) (?=[ąęĄĘ])"
        r"|(?<=[A-ZĄĆĘŁŃÓŚŹŻ]) (?=[ĄĘ])")

    @staticmethod
    def _strip_heading_number(text: str) -> str:
        """
        Removes a leading section number (e.g. '5. ', '2.1 ', 'A. ') from a heading
        and returns the upper-cased remainder. Falls back to the whole text if nothing remains.
        """
        cleaned = PDFMapper.HEADING_NUM_RE.sub("", text, count=1).strip().upper()
        return cleaned or text.strip().upper()

    @staticmethod
    def _is_bib_heading_line(text: str) -> bool:
        """
        True if a single line is exactly a bibliography heading (optionally numbered,
        e.g. '9. REFERENCES', 'Bibliografia:'). Used for headings that sit in one PDF block
        together with the bibliography entries and are not bold / larger than body text.
        """
        text = text.strip()
        if not text or len(text) >= 80:
            return False
            
        if re.match(r"^\s*\d+\.\d+", text):
            return False
            
        key = PDFMapper._strip_heading_number(text).rstrip(":. ")
        return key in PDFMapper.BIB_HEADINGS

    @staticmethod
    def _is_after_bib_heading_line(text: str) -> bool:
        """
        True if a single line is exactly a heading that ends the bibliography 
        (e.g. 'ABSTRACT', 'SPIS RYSUNKÓW').
        """
        text = text.strip()
        if not text or len(text) >= 80:
            return False
        key = PDFMapper._strip_heading_number(text).rstrip(":. ")
        return bool(PDFMapper.AFTER_BIB_HEADING_RE.match(key))

    @staticmethod
    def _pitch_matches(pitch: float | None, ref_pitch: float | None, tol: float = 0.15) -> bool:
        """
        Checks whether the line pitch matches the reference pitch within tolerance.
        """
        if pitch is None or ref_pitch is None or ref_pitch <= 0:
            return True            
        return abs(pitch - ref_pitch) <= tol * ref_pitch

    @staticmethod
    def _ended_early(prev: LineGeom, next_first_word_w: float, content_right: float) -> bool:
        """
        Checks whether the previous line ended with sufficient room for the next word.
        """
        if prev.ends_hyphen:
            return False
        free = content_right - prev.x1
        return free > next_first_word_w + PDFMapper.SPACE_W + PDFMapper.EARLY_SLACK

    @staticmethod
    def _indent_vs_block(line_x0: float, block_line_x0s: list[float], thresh: float = 10) -> bool:
        """
        Checks whether a line is indented relative to the median left margin of the block.
        """
        others = sorted(block_line_x0s)
        median_x0 = others[len(others)//2]
        return line_x0 - median_x0 > thresh

    @classmethod
    def _is_spans_monospace(cls, spans) -> bool:
        """
        Determines whether any span uses a monospace font family.
        """
        return any(
            any(mono in (getattr(s, "font", "") or "").lower() for mono in cls.MONOSPACE_FONTS)
            for s in spans
        )

    @staticmethod
    def _block_text(b) -> str:
        """
        Extracts concatenated text from all spans across all lines in a text block.
        """
        return " ".join(s.text.strip() for l in b.lines for s in l.spans if s.text.strip())

    def _math_like(self, b) -> bool:
        """
        Heuristically determines whether a text block represents a mathematical formula.
        """
        txt = self._block_text(b)
        if not txt:
            return False
        spans = [s for l in b.lines for s in l.spans if s.text.strip()]
        if not spans:
            return False
        mf = sum(any(m in (getattr(s, "font", "") or "").lower() for m in self.MATH_FONTS) for s in spans) / len(spans)
        letters = sum(c.isalpha() for c in txt)
        words = txt.split()
        short_tokens = sum(len(w) <= 2 for w in words) / len(words) if words else 0.0
        return mf > 0.4 or (len(txt) < 40 and short_tokens > 0.7) or letters / max(len(txt), 1) < 0.35

    def _formula_regions(self, page, content_right: float | None):
        """
        Identifies bounding rectangles of formulas based on equation numbers and adjacent math blocks.
        """
        regions = []
        for b in page.text_blocks:
            t = self._block_text(b)
            is_right_side = (content_right is None) or (b.bbox[0] > content_right - 80) or (b.bbox[2] > content_right - 50)
            if self.EQ_NUM_RE.match(t) and is_right_side:
                y0, y1 = b.bbox[1] - 8, b.bbox[3] + 8
                changed = True
                while changed:
                    changed = False
                    for c in page.text_blocks:
                        if c is b:
                            continue

                        vertically_aligned = (c.bbox[3] >= y0 - 6 and c.bbox[1] <= y1 + 6)
                        if vertically_aligned and (self._math_like(c) or c.bbox[2] < b.bbox[0]):
                            ny0, ny1 = min(y0, c.bbox[1]), max(y1, c.bbox[3])
                            if (ny0, ny1) != (y0, y1):
                                y0, y1, changed = ny0, ny1, True
                regions.append((y0, y1))
        return regions

    def __init__(self):
        # Stan wewnętrzny
        self.logical_blocks = []
        self.paragraph_buffer = []
        self.list_buffer = []
        self.curr_line = 0
        self._last_extracted_y0 = None
        self.last_y1 = None
        self.in_bibliography_section = False
        self._line_is_monospace = False
        self.list_stack = []
        self._block_seq = 0

    def _classify_block_content(self, text: str, active_marker: str = None):
        """
        Classifies block content while taking monospace font and bibliography state into account.
        """
        block_type, marker_type = classify_block_content(
            text,
            active_marker=active_marker,
            in_bibliography=getattr(self, "in_bibliography_section", False),
        )
        if marker_type == "listing" and active_marker != "listing":
            if not getattr(self, "_line_is_monospace", False):
                return "paragraph", None
        return block_type, marker_type

    @staticmethod
    def is_continuation(last_item_bbox: list, current_block_bbox: list) -> bool:
        """
        Checks if a block is a continuation of the previous list item
        based on its horizontal (X-axis) alignment.

        Args:
            last_item_bbox (list): Bounding box of the last list item [x0, y0, x1, y1].
            current_block_bbox (list): Bounding box of the current text block [x0, y0, x1, y1].

        Returns:
            bool: True if the current block aligns with or is indented further than the last item, False otherwise.
        """
        last_x0 = last_item_bbox[0]
        curr_x0 = current_block_bbox[0]
        return (
            abs(last_x0 - curr_x0) < PDFMapper.LIST_CONT_MAX_X_DIFF or curr_x0 > last_x0
        )
        
    TERMINAL = (".", "!", "?", ":", ";", ",")

    def _line_ends_unit(
        self,
        prev_geom,
        prev_text,
        next_first_word_w,
        content_right,
        is_justified,
        next_text="",
    ):
        """
        Determines whether the previous line terminated a grammatical or semantic unit
        based on terminal punctuation and early right-margin ending.
        """
        if prev_geom.ends_hyphen:
            return False
        first_clean_char = next((c for c in next_text.strip() if c.isalnum()), "")
        if first_clean_char and first_clean_char.islower():
            return False
        ended_early = self._ended_early(prev_geom, next_first_word_w, content_right)
        ends_punct = prev_text.rstrip().endswith(self.TERMINAL)
        return ended_early and ends_punct

    def _is_item_continuation(
        self,
        geom,
        prev_geom,
        item,
        ref_pitch,
        content_right,
        line_text,
        prev_text="",
        is_justified=False,
    ):
        """
        Determines whether the current line continues the active list item
        using geometric layout, indent alignment, pitch, and punctuation checks.
        """
        if not self._use_geom():
            return PDFMapper.is_continuation(item.bbox, [geom.x0, geom.y0, geom.x1, geom.y1])

        first_clean_char = next((c for c in line_text.strip() if c.isalnum()), "")
        is_lowercase_continuation = bool(first_clean_char and first_clean_char.islower())

        if line_text.strip().isdigit():
            result = False
        elif self._classify_block_content(line_text, None)[0] == "list":
            result = False
        elif item.words and getattr(geom, "font_size", None) and (geom.font_size < statistics.median(w.size for w in item.words) - 1.2):
            result = False
        elif prev_geom and prev_geom.ends_hyphen:
            result = True
        elif not any(
            abs(geom.x0 - cx) <= 8
            for cx in ([item.bbox[0]] + ([item.words[0].bbox[0]] if item.words else []) + ([item.words[1].bbox[0]] if len(item.words) > 1 else []))
        ):
            result = False
        elif (
            ref_pitch is not None
            and not self._pitch_matches(geom.pitch, ref_pitch, self.PITCH_TOL)
            and not is_lowercase_continuation
        ):
            result = False
        elif prev_geom and self._line_ends_unit(
            prev_geom,
            prev_text or item.text,
            geom.first_word_w,
            content_right,
            is_justified,
            next_text=line_text,
        ):
            result = False
        elif prev_geom is None and (prev_text or item.text).rstrip().endswith((".", "!", "?")):
            first_raw = line_text.strip()
            if re.match(r"^\([a-z0-9]+\)", first_raw):
                result = False
            else:
                first_clean_char = next((c for c in first_raw if c.isalnum()), "")
                if not (first_clean_char and first_clean_char.islower()):
                    result = False
                else:
                    result = True
        else:
            result = True

        logger.info(
            "CONT? line=%r x0=%.1f prev_end_early=%s justified=%s -> %s",
            line_text[:40],
            geom.x0,
            self._ended_early(prev_geom, geom.first_word_w, content_right) if prev_geom else None,
            is_justified,
            result,
        )
        return result


    @staticmethod
    def _adjust_item_word_positions(item_words, raw_text=None, cleaned_text=None):
        """
        Adjusts the character position indices of words within a list item
        by removing the offset caused by the list marker (bullet point, number etc).

        Args:
            item_words (list[WordInfo]): Words belonging to a single list item.
            raw_text (str): Item text including the list marker.
            cleaned_text (str): Item text with the list marker stripped.
        """

        marker_offset = None
        if raw_text and cleaned_text:
            word_position = raw_text.find(cleaned_text)
            if word_position >= 0:
                marker_offset = word_position
        if marker_offset is None:
            if len(item_words) <= 1:
                return
            marker_offset = item_words[1].start_char

        for word in item_words:
            word.start_char -= marker_offset
            word.end_char -= marker_offset

        for word in item_words:
            if word.start_char < 0:
                word.start_char = 0
            if word.end_char < 0:
                word.end_char = 0


    @staticmethod
    def is_header(words: list[WordInfo]) -> bool:
        """
        Determines if a given sequence of words constitutes a heading based on
        typographical features (bold, italic, or font size).

        Args:
            words (list[WordInfo]): A list of WordInfo objects to evaluate.

        Returns:
            bool: True if the words format matches a heading (fully bold/italic and short,
            or has an unusually large average size), False otherwise.
        """
        if not words:
            return False
        is_bold = all(w.bold for w in words)
        is_italic = all(w.italic for w in words)
        avg_size = sum(w.size for w in words) / len(words)
        return (
            (is_bold and len(words) < 15)
            or (is_italic and len(words) < 15)
            or avg_size > 12.5
        )

    @staticmethod
    def is_keywords(words: list[WordInfo]) -> bool:
        """
        Checks if a given sequence of words represents a 'keywords' or 'abbreviations' section
        by analyzing the beginning of the constructed string.

        Args:
            words (list[WordInfo]): A list of WordInfo objects to evaluate.

        Returns:
            bool: True if the text begins with recognized keyword/abbreviation identifiers
            (in Polish or English), False otherwise.
        """
        if not words:
            return False
        full_text = " ".join(w.text for w in words).strip()
        if full_text.lower().startswith(
            ("słowa kluczowe", "keywords", "key words", "keywords:", "skróty")
        ):
            return True
        return False

    @staticmethod
    def is_math(words: list[WordInfo]) -> bool:
        """
        Evaluates a sequence of words to determine if it represents a mathematical expression.
        Analyzes factors such as the presence of leader dots, space density, and the use of specific math fonts.

        Args:
            words (list[WordInfo]): A list of WordInfo objects to evaluate.

        Returns:
            int: An integer code indicating the detection result (e.g., 0 for not math, 1 for high space density, 2 for dominant math fonts).
        """
        if not words:
            return 0

        full_text_with_spaces = " ".join(w.text for w in words)
        if re.search(r"(?:\.\s*){4,}", full_text_with_spaces):
            return 0

        spaces_count = sum(w.text.count(" ") for w in words) + (len(words) - 1)

        total_chars_with_spaces = sum(len(w.text) for w in words) + (len(words) - 1)

        if total_chars_with_spaces > 0:
            space_density = spaces_count / total_chars_with_spaces
        if space_density > 0.3:
            return 1

        math_font_count = sum(
            1
            for w in words
            if any(
                mf in w.font.lower() for mf in ["math", "cmmi", "cmr", "cmsy", "symbol"]
            )
        )
        if len(words) > 0 and (math_font_count / len(words)) > 0.4:
            return 2

        full_text = "".join(w.text for w in words).replace(" ", "")
        if not full_text:
            return 0

        math_chars_pattern = (
            r"[0-9=\+\-\*/<>\∑\∫\∏\√\∞\≈\≠\≡\≤\≥\{\}\(\)\[\]\|\α-\ω\Α-\Ω]"
        )
        math_chars_count = len(re.findall(math_chars_pattern, full_text))

        single_letter_count = sum(
            1 for w in words if len(w.text) == 1 and w.text.isalpha()
        )

        ratio = (math_chars_count + single_letter_count) / len(full_text)

        if ratio > 0.35:
            return 3
        else:
            return 0

    @staticmethod
    def is_inside_table(block_bbox: list, table_bboxes: list) -> bool:
        """
        Determines whether a specific text block is located within the boundaries of any known tables.
        Calculates the intersection area between the block and the tables, allowing for a slight margin of error.

        Args:
            block_bbox (list): The bounding box of the text block [x0, y0, x1, y1].
            table_bboxes (list): A list of bounding boxes for all detected tables, where each is [x0, y0, x1, y1].

        Returns:
            bool: True if the block's area intersecting with a table exceeds the defined threshold, False otherwise.
        """
        bx0, by0, bx1, by1 = block_bbox
        block_area = (bx1 - bx0) * (by1 - by0)
        if block_area <= 0:
            return False

        for tx0, ty0, tx1, ty1 in table_bboxes:
            ix0 = max(bx0, tx0 - 5)
            iy0 = max(by0, ty0 - 5)
            ix1 = min(bx1, tx1 + 5)
            iy1 = min(by1, ty1 + 5)
            if ix0 < ix1 and iy0 < iy1:
                intersect_area = (ix1 - ix0) * (iy1 - iy0)
                if intersect_area / block_area > PDFMapper.TABLE_INTERSECT_THRESH:
                    return True
        return False

    # Zarządzanie buforami (Wymagają stanu)
    def _empty_paragraph_buffer(self, debug_why_empty=""):
        """
        Processes and clears the current paragraph buffer, converting accumulated lines
        into a single logical block (e.g., paragraph, heading, keywords, or acronyms).
        It handles text concatenation, hyphenation resolution, and evaluates typographical
        errors (widows, bastards, shoemakers) before appending the finalized block to
        the document's logical blocks.

        Args:
            debug_why_empty (str, optional): A debugging note explaining the reason
                for triggering the buffer flush. Defaults to "".
        """
        if not self.paragraph_buffer:
            return

        is_acronym_block = False
        total_lines = len(self.paragraph_buffer)
        is_widow = 0
        is_bekart = 0
        is_szewc = 0

        # Wykrywanie bloków z akronimami (przypadki, gdzie całość zlepiona w jedną linijkę)
        if total_lines == 1:
            content = self.paragraph_buffer[0]["content"].strip()
            starts_with_sep = bool(re.match(r"^\S{1,15}\s*[-–—−‐:=]\s+", content))
            starts_with_upper = bool(re.match(r"^[A-ZĄĆĘŁŃÓŚŹŻ0-9]{2,}\b\s+", content))
            sep_matches = re.findall(r"\s+\S{1,15}\s*[-–—−‐:=]\s+", " " + content)
            if (starts_with_sep or starts_with_upper) and len(sep_matches) >= 3:
                is_acronym_block = True

        # Wykrywanie bloków z akronimami (więcej niż trzy linijki)
        elif total_lines > 3:
            acronym_lines = 0
            for data in self.paragraph_buffer:
                text = data["content"].strip()
                if is_acronym(text) == 1:
                    acronym_lines += 1
            if (acronym_lines / total_lines) >= self.ACRONYM_THRESH:
                is_acronym_block = True

        combined_content = ""
        combined_words = []
        current_offset = 0

        # Pętla zapisująca zawartość bufora
        for i, data in enumerate(self.paragraph_buffer):
            content = data["content"]
            separator = ""
            removed_hyphen = False

            if is_acronym_block and i > 0:
                separator = "\n"
            elif i > 0:
                if combined_content.rstrip().endswith("-"):
                    separator = ""
                    removed_hyphen = True
                else:
                    separator = " "

            if removed_hyphen:
                combined_content = combined_content.rstrip()[:-1]
                current_offset = len(combined_content)

            for word in data["words"]:
                shift = current_offset + len(separator)
                combined_words.append(
                    WordInfo(
                        word_index=len(combined_words),
                        text=word.text,
                        start_char=word.start_char + shift,
                        end_char=word.end_char + shift,
                        font=word.font,
                        size=word.size,
                        bold=word.bold,
                        italic=word.italic,
                        bbox=word.bbox,
                        page_number=word.page_number,
                        line=word.line,
                        is_index = getattr(word, "is_index", False),
                    )
                )

            combined_content += separator + content
            current_offset = len(combined_content)

        block_type = "acronyms" if is_acronym_block else "paragraph"

        if block_type == "paragraph":
            if combined_words:
                avg_size = sum(w.size for w in combined_words) / len(combined_words)
                min_y0 = min(w.bbox[1] for w in combined_words)
                page_num = combined_words[0].page_number
                page_height = getattr(self, "page_heights", {}).get(page_num, 842.0)


            if block_type == "paragraph":
                if PDFMapper.is_header(combined_words):
                    block_type = "heading"
                elif PDFMapper.is_keywords(combined_words):
                    block_type = "keywords"
        if block_type == "paragraph":
            is_widow = (
                is_widow_func(combined_words)
                if is_widow_func(combined_words) != 0
                else 0
            )
            is_bekart = (
                is_bekart_func(combined_words)
                if is_bekart_func(combined_words) != 0
                else 0
            )
            is_szewc = (
                is_szewc_func(combined_words)
                if is_szewc_func(combined_words) != 0
                else 0
            )

        is_bib = getattr(self, "in_bibliography_section", False)

        if is_bib:
            if combined_words:
                bbox = [
                    min(w.bbox[0] for w in combined_words),
                    min(w.bbox[1] for w in combined_words),
                    max(w.bbox[2] for w in combined_words),
                    max(w.bbox[3] for w in combined_words),
                ]
            else:
                bbox = [0, 0, 0, 0]
            item = ListItem(
                item_id=self.paragraph_buffer[0]["block_id"],
                marker_type="none",
                text=combined_content,
                bbox=bbox,
                words=combined_words,
            )
            list_block = ListBlock(
                block_id=self.paragraph_buffer[0]["block_id"],
                content=combined_content,
                bbox=bbox,
                type="list",
                is_bibliography=True,
                items=[item],
                words=combined_words,
            )
            list_block._seq = self._block_seq
            self._block_seq += 1
            self.logical_blocks.append(list_block)
        else:
            p_block = ParagraphBlock(
                block_id=self.paragraph_buffer[0]["block_id"],
                content=combined_content,
                words=combined_words,
                type=block_type,
                is_widow=is_widow,
                is_bekart=is_bekart,
                is_szewc=is_szewc,
                debug_empty=debug_why_empty,
            )
            p_block._seq = self._block_seq
            self._block_seq += 1
            self.logical_blocks.append(p_block)
        self.paragraph_buffer.clear()


    def _item_to_paragraph(self, data) -> ParagraphBlock:
        """
        Converts a single list item into a paragraph block, leaving out marker offset for correct linguistic parsing,
        marker remains as item[0], for bibliography checks.
        """
        item = data["item"]
        words = item.words
        if len(words) > 1 and words[0].start_char == 0 and words[0].end_char == 0:
            words = words[1:]
        return ParagraphBlock(
            block_id=data["block_id"],
            content=item.text,
            words=words,
        )

    def _empty_list_buffer(self, reason: str = ""):
        """
        Processes and clears the current list buffer, aggregating accumulated list items
        into a finalized ListBlock. It combines text, words, and bounding boxes across
        all items, determines the specific list type (e.g., standard list, code snippet),
        and appends the resulting block to the document's logical blocks.
        """
        if not self.list_buffer:
            return

        if getattr(self, "in_bibliography_section", False):
            logger.info(
                "EMPTY_LIST_BUFFER in bib: reason=%r, count=%d, first_item=%r",
                reason,
                len(self.list_buffer),
                self.list_buffer[0]["item"].text[:60] if self.list_buffer else "",
            )

        items = [data["item"] for data in self.list_buffer]
        all_words = []
        for item in items:
            all_words.extend(item.words)

        combined_content = "\n".join(item.text for item in items)

        is_sublist = self.USE_NESTED_LISTS and (bool(self.list_stack) or reason == "koniec podlisty")
        if len(items) > 1 or is_sublist:
            first_item_data = self.list_buffer[0]

            is_bibiography = (
                True if getattr(self, "in_bibliography_section", False) else False
            )
            
            if is_bibiography:
                valid_markers = [item.marker_type for item in items if getattr(item, "marker_type", "none") != "none"]
                dominant_marker = max(set(valid_markers), key=valid_markers.count) if valid_markers else "none"
                for item in items:
                    item.marker_type = dominant_marker

            if not is_bibiography and any(getattr(item, "marker_type", "") == "number_in_brackets" for item in items):
                for data in self.list_buffer:
                    p_block = self._item_to_paragraph(data)
                    p_block._seq = data.get("_seq", self._block_seq)
                    self._block_seq += 1
                    self.logical_blocks.append(p_block)
                self.list_buffer.clear()
                return

            data = self.list_buffer[0]
            item = data["item"]

            is_code = item.marker_type == "listing"
            if self.USE_GEOMETRY_HEURISTICS and not is_code and len(items) > 1:
                is_monospace = any(
                    any(mono in (getattr(w, "font", "") or "").lower() for mono in self.MONOSPACE_FONTS)
                    for w in all_words
                )
                x0_values = [it.bbox[0] for it in items]
                constant_x0 = (max(x0_values) - min(x0_values)) <= 5.0
                if is_monospace and constant_x0:
                    is_code = True

            if is_code:
                list_type = "code_snippet"
            else:
                list_type = "list"

            new_list_block = ListBlock(
                block_id=f"list_{first_item_data['block_id']}",
                content=combined_content,
                is_bibliography=is_bibiography,
                words=all_words,
                items=items,
                bbox=first_item_data["bbox"],
                type=list_type,
            )

            all_bboxes = [item.bbox for item in items]
            new_list_block.bbox = [
                min(b[0] for b in all_bboxes),
                min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes),
                max(b[3] for b in all_bboxes),
            ]
            new_list_block._seq = self.list_buffer[0].get("_seq", self._block_seq)
            self.logical_blocks.append(new_list_block)
        else:
            data = self.list_buffer[0]
            p_block = self._item_to_paragraph(data)
            p_block._seq = data.get("_seq", self._block_seq)
            self.logical_blocks.append(p_block)
        self.list_buffer.clear()

    def _flush_all_lists(self, reason: str = ""):
        """
        Flushes the current list buffer and unwinds any suspended lists from list_stack.
        """
        self._empty_list_buffer(reason)
        while self.list_stack:
            self.list_buffer = self.list_stack.pop()
            self._empty_list_buffer(reason)

    def _marker_x0(self):
        """
        Returns the x0 coordinate of the active list's most recent item.
        Prefers the first word's bounding box (the marker), falls back to bbox[0].
        """
        if not self.list_buffer:
            return None
        last_item = self.list_buffer[-1]
        if last_item.get("words"):
            return last_item["words"][0].bbox[0]
        return last_item.get("bbox", [0])[0]

    def _handle_marker_line(self, new_marker: str, new_x0: float | None):
        """
        Replaces _close_if_marker_changed: manages list nesting and resumption based on marker x0.
        - Deeper (x0 > cur_x0 + 12): suspends current list onto list_stack and starts sublist.
        - Shallower (x0 < cur_x0 - 12): flushes sublist and restores matching parent list.
        - Same level, different marker: flushes current list.
        """
        if not self.list_buffer:
            return

        cur_x0 = self._marker_x0()
        cur_marker = self.list_buffer[-1]["item"].marker_type

        NESTABLE_PARENTS = {"bullet", "dash", "number_with_dot",
                    "number_with_bracket", "letter_with_bracket"}
        NESTABLE_CHILDREN = {"bullet", "dash"}

        can_nest = (
            self.USE_NESTED_LISTS
            and cur_marker in NESTABLE_PARENTS
            and new_marker in NESTABLE_CHILDREN
)

        if (
            can_nest
            and new_x0 is not None
            and cur_x0 is not None
            and new_x0 > cur_x0 + 12
        ):
            self.list_stack.append(self.list_buffer)
            self.list_buffer = []
            return

        if self.USE_NESTED_LISTS and new_x0 is not None and cur_x0 is not None:
            while self.list_stack and new_x0 < cur_x0 - 12:
                self._empty_list_buffer("koniec podlisty")
                self.list_buffer = self.list_stack.pop()
                cur_x0 = self._marker_x0()
                if cur_x0 is None:
                    break
                cur_marker = self.list_buffer[-1]["item"].marker_type

        if cur_marker != new_marker:
            self._empty_list_buffer("zmiana typu markera")

    def _close_if_marker_changed(self, new_marker: str):
        """
        Backwards-compatible wrapper calling _handle_marker_line without geometry.
        """
        self._handle_marker_line(new_marker, None)

    # Logika podziału i (wielokrotnych) spacji
    def _detect_paragraph_break(
        self,
        current_y0,
        current_y1,
        line_x0,
        x0_margin,
        full_text,
        is_valid_list_cont,
        geom: LineGeom | None = None,
        prev_geom: LineGeom | None = None,
        content_right: float | None = None,
        block_line_x0s: list[float] | None = None,
    ):
        """
        Determines whether the current text line indicates the start of a new paragraph
        based on spatial positioning (vertical gaps, horizontal indentation) and
        linguistic cues (ending punctuation).

        Args:
            current_y0 (float): The top Y-coordinate of the current line.
            current_y1 (float): The bottom Y-coordinate of the current line.
            line_x0 (float): The starting X-coordinate of the current line.
            x0_margin (float): The standard left margin X-coordinate of the page.
            full_text (str): The text accumulated in the current line so far.
            is_valid_list_cont (bool): Indicates if the line has been identified as a
                valid continuation of an ongoing list.
            geom (LineGeom, optional): Geometry details of current line.
            prev_geom (LineGeom, optional): Geometry details of previous line.
            content_right (float, optional): Estimated right boundary of content on page.
            block_line_x0s (list[float], optional): List of x0 positions of lines in current block.

        Returns:
            tuple[bool, str]: A tuple containing a boolean indicating if a paragraph
            break was detected, and a string with the debugging reason for the break.
        """
        is_new_paragraph = False
        debug_reason = ""


        if prev_geom and abs(current_y0 - prev_geom.y0) < 3.0:
            return False, ""
        
        finished = True
        if full_text.strip():
            stripped = full_text.strip()
            finished = (
                stripped[-1] in {".", "!", "?", ":", ";"}
                or bool(self.IMG_PATTERN.match(stripped))
                or bool(self.TAB_PATTERN.match(stripped))
            )
        elif self.paragraph_buffer:
            p_content = self.paragraph_buffer[-1]["content"].strip()
            finished = (
                p_content[-1] in {".", "!", "?", ":", ";"}
                or bool(self.IMG_PATTERN.match(p_content))
                or bool(self.TAB_PATTERN.match(p_content))
            )
        elif self.list_buffer:
            item_text = self.list_buffer[-1]["item"].text.strip()
            finished = bool(item_text and item_text[-1] in {".", "!", "?", ":", ";"})

        # Wertykalna przerwa
        if self.last_y1 is not None:
            if current_y0 < self.last_y1 - 10:
                self.last_y1 = None
            else:
                vertical_gap = current_y0 - self.last_y1
                line_height = current_y1 - current_y0
                if vertical_gap > line_height * 1.5:
                    if not finished:
                        is_new_paragraph = False
                    else:
                        is_new_paragraph = True
                        debug_reason = "zbyt duża wertykalna przerwa"

        self.last_y1 = current_y1

        # Wcięcie akapitowe
        if not is_new_paragraph:
            if not full_text.strip() and (
                line_x0 > x0_margin + self.MARGIN_INDENT_THRESH
            ):
                if not finished:
                    is_new_paragraph = False 
                else:
                    is_new_paragraph = True
                    debug_reason = "wcięcie na początku bloku/strony"
            elif full_text.strip() and getattr(self, "in_bibliography_section", False):
                if finished and line_x0 <= x0_margin + self.MARGIN_INDENT_THRESH:
                    is_new_paragraph = True
                    debug_reason = "nowy wpis bibliografii (brak wcięcia)"
            elif full_text.strip():
                prev_type, _ = self._classify_block_content(full_text)
                if prev_type == "list" and finished and line_x0 <= x0_margin + self.MARGIN_INDENT_THRESH:
                    is_new_paragraph = True
                    debug_reason = "koniec elementu listy i nowa linia od marginesu"

        if (
            self._use_geom()
            and not is_new_paragraph
            and geom is not None
            and prev_geom is not None
            and content_right is not None
            and block_line_x0s
        ):
            if (
                self._indent_vs_block(geom.x0, block_line_x0s)
                and self._ended_early(prev_geom, geom.first_word_w, content_right)
                and finished
            ):
                is_new_paragraph = True
                debug_reason = "wcięcie względem bloku oraz przedwczesny koniec linii"
            #TU DODAC SPRAWDZENIE CZY JEST NA TEJ SAMEJ LINII
        if is_valid_list_cont:
            is_new_paragraph = False

        return is_new_paragraph, debug_reason

    def _extract_words_with_spacing(
        self, line, full_text, words_info, page_num, word_counter, line_x1
    ):
        """
        Extracts individual words from a text line, estimates their precise bounding boxes,
        and reconstructs the text with appropriate spacing. It dynamically calculates
        horizontal gaps between spans to intelligently insert missing spaces, particularly
        handling punctuation and hyphenated line breaks.

        Args:
            line: The line object from the PDF extractor containing text spans.
            full_text (str): The accumulated string of text for the current block.
            words_info (list[WordInfo]): The list accumulating WordInfo objects.
            page_num (int): The current page number.
            word_counter (int): The current global index/counter for extracted words.
            line_x1 (float): The ending X-coordinate of the current line.

        Returns:
            tuple[str, list[WordInfo], int]: A tuple containing the updated full_text string,
            the appended words_info list, and the incremented word_counter.
        """
        line_gaps = []
        for i in range(len(line.spans) - 1):
            g = line.spans[i + 1].bbox[0] - line.spans[i].bbox[2]
            if g > 0:
                line_gaps.append(g)

        m_gap = statistics.median(line_gaps) if line_gaps else 3.0
        prev_span_x1 = None

        span_sizes = [s.size for s in line.spans if s.text.strip()]
        line_base_size = statistics.mode(span_sizes) if span_sizes else getattr(self, "main_font_size", 12.0)

        for span in line.spans:
            word_text = span.text.replace("\u200b", "").strip()
            if not word_text:
                continue

            is_span_index = is_index(span, line_base_size)


            is_combining = all(
                unicodedata.category(ch).startswith("M") for ch in word_text
            )
            if is_combining and words_info:
                last_word = words_info[-1]
                merged = unicodedata.normalize("NFC", last_word.text + word_text)
                last_word.text = merged
                last_word.end_char = last_word.start_char + len(merged)
                if full_text.endswith(" "):
                    full_text = full_text[:-1]
                full_text = full_text[: last_word.start_char] + merged + full_text[last_word.start_char + len(merged) :]
                last_word.bbox[2] = max(last_word.bbox[2], span.bbox[2])
                last_word.bbox[3] = max(last_word.bbox[3], span.bbox[3])
                prev_span_x1 = span.bbox[2]
                continue

            if prev_span_x1 is not None:
                current_gap = span.bbox[0] - prev_span_x1

                if not (
                    full_text.rstrip().endswith("-")
                    and abs(line_x1 - span.bbox[2]) < 50
                ):
                    full_text += " "

                after_punct = full_text.strip().endswith((".", "!", "?", ":", ";"))

                if current_gap > 1.2 * m_gap and not after_punct:
                    full_text += " "
                if current_gap > 1.5 * m_gap and after_punct:
                    full_text += " "

            sub_words = word_text.split()
            span_x0, span_y0, span_x1, span_y1 = span.bbox
            span_width = span_x1 - span_x0
            total_chars = len(word_text)
            current_sub_x0 = span_x0

            for i, sub_w in enumerate(sub_words):
                if i > 0:
                    full_text += " "
                sub_w = unicodedata.normalize("NFC", sub_w)
                start_char = len(full_text)
                sub_w_width = (
                    (len(sub_w) / total_chars) * span_width if total_chars > 0 else 0
                )
                sub_x1 = current_sub_x0 + sub_w_width
                approx_bbox = [current_sub_x0, span_y0, sub_x1, span_y1]

                words_info.append(
                    WordInfo(
                        word_index=word_counter,
                        text=sub_w,
                        start_char=start_char,
                        end_char=start_char + len(sub_w),
                        font=span.font,
                        size=span.size,
                        bold=span.bold,
                        italic=span.italic,
                        bbox=approx_bbox,
                        page_number=page_num,
                        line=self.curr_line,
                        is_index = is_span_index,
                    )
                )

                full_text += sub_w
                word_counter += 1

                space_width = (1 / total_chars) * span_width if total_chars > 0 else 0
                current_sub_x0 = sub_x1 + space_width

            prev_span_x1 = span.bbox[2]

        line_y0 = line.spans[0].bbox[1] if line.spans else line.bbox[1]
        if getattr(self, "_last_extracted_y0", None) is not None:
            if abs(line_y0 - self._last_extracted_y0) > 3.0:
                self.curr_line += 1
        self._last_extracted_y0 = line_y0
        if not full_text.endswith(" ") and not full_text.rstrip().endswith("-"):
            full_text += " "

        return full_text, words_info, word_counter

    # Główna metoda mapowania
    def map_to_schema(self, old_doc: DocumentData) -> FinalDocument:
        """
        The main mapping method that transforms raw extracted PDF data into a highly
        structured FinalDocument schema. It reinitializes the mapper's state, processes
        the document page by page, flushes remaining buffers, and runs a comprehensive
        suite of post-processing routines (e.g., merging headings, tagging captions,
        extracting acronyms, and verifying bibliography references).

        Args:
            old_doc (DocumentData): The raw, parsed data extracted directly from the PDF.

        Returns:
            FinalDocument: The fully structured document containing categorized logical blocks,
            floating visual elements, and reference sections ready for NLP analysis.
        """
        self.__init__()

        self.main_font_size = old_doc.get_most_common_font_size()
        self.page_heights = {page.number: page.height for page in old_doc.pages}

        self.avg_line_spacing = old_doc.metadata.get("avarge_line_spacing", 18.0)

        new_doc = FinalDocument(
            metadata=old_doc.metadata,
            floating_elements=FloatingElements(),
            reference_sections=ReferenceSections(),
        )
        self.logical_blocks = new_doc.logical_blocks


        for page in old_doc.pages:
            self._process_page(page, new_doc)

        self._flush_all_lists("koniec dokumentu")
        self._empty_paragraph_buffer("finalne opróżnienie")

        self.logical_blocks.sort(key=lambda b: getattr(b, "_seq", 0))

        self._merge_adjacent_headings()
        self._extract_acronyms_to_schema(new_doc)
        self._tag_special_lists(old_doc)
        self._tag_visual_descriptions(new_doc)
        self._pair_descriptions_with_visuals(new_doc)
        self._verify_bibliography_references()
        self._verify_caption_sources(new_doc)
        self._fix_offsets()
        self._assign_unique_ids()

        new_doc.metadata["main_font"] = old_doc.get_most_common_font()

        return new_doc

    def _process_page(self, page, new_doc):
        """
        Processes a single page of the document to classify and extract its text blocks,
        lines, and visual elements. It evaluates spatial positioning to detect paragraph
        breaks, list continuations, page transitions, and floating artifacts (such as
        page numbers or standalone mathematical expressions). It also populates the
        document's visual elements list with images and tables found on the page.

        Args:
            page: The raw page object containing text blocks, tables, and images.
            new_doc (FinalDocument): The structured document instance currently being populated.
        """
        self.last_y1 = None
        bottom_thresh = page.height - self.BOTTOM_MARGIN_OFFSET
        table_bboxes = [t.bbox for t in page.tables]

        img_pattern = self.IMG_PATTERN
        tab_pattern = self.TAB_PATTERN

        geometry_classifier = GeometryClassifier()
        margins = geometry_classifier.calculate_margins(
            [{"bbox": b.bbox} for b in page.text_blocks], page.width, page.height
        )
        x0_margin = margins["left"]

        if self.USE_GEOMETRY_HEURISTICS:
            xs = sorted(
                l.bbox[2]
                for b in page.text_blocks
                for l in b.lines
                if getattr(l, "alignement", "") == "justified"
            )
            content_right = (
                xs[int(len(xs) * 0.9)]
                if xs
                else (
                    sorted(l.bbox[2] for b in page.text_blocks for l in b.lines)[
                        int(len([l.bbox[2] for b in page.text_blocks for l in b.lines]) * 0.9)
                    ]
                    if [l.bbox[2] for b in page.text_blocks for l in b.lines]
                    else page.width - margins.get("right", 50)
                )
            )
        else:
            content_right = None

        prev_geom: LineGeom | None = None
        prev_line_text: str = ""
        prev_line_justified: bool = False
        ref_pitch: float | None = None
        formula_regions = (
            self._formula_regions(page, content_right) if self._use_geom() else []
        )

        for block in page.text_blocks:
            if getattr(block, "block_type", "") == "footnote":
                new_doc.floating_elements.page_artifacts.append(
                    PageArtifact(
                        artifact_id=f"fn_{block.block_id}",
                        type="footnote",
                        page_number=page.number,
                        text=self._block_text(block),
                        bbox=list(block.bbox),
                    )
                )
                continue
            block_line_x0s = [
                (
                    next((s.bbox[0] for s in l.spans if s.text.strip()), l.spans[0].bbox[0])
                    if l.spans
                    else l.bbox[0]
                )
                for l in block.lines
            ]
            current_marker = (
                self.list_buffer[-1]["item"].marker_type if self.list_buffer else None
            )
            full_text = ""
            words_info = []
            word_counter = 0

            temp_text = (
                "".join(span.text for line in block.lines for span in line.spans)
                .strip()
                .lower()
            )
            x0, y0, x1, y1 = block.bbox

            raw_block_text_for_check = "".join(
                span.text for line in block.lines for span in line.spans
            ).strip()
            sub_caption_pattern = re.compile(r"^\([a-z0-9]+\)\s+", re.IGNORECASE)
            is_visual_caption = bool(
                img_pattern.match(raw_block_text_for_check)
                or tab_pattern.match(raw_block_text_for_check)
                or (sub_caption_pattern.match(raw_block_text_for_check) and bool(page.images or page.tables))
            )
            if is_visual_caption:
                if self.paragraph_buffer:
                    self._empty_paragraph_buffer("odcięcie - wykryto blok podpisu")
                if self.list_buffer:
                    self._flush_all_lists("odcięcie - wykryto blok podpisu")

            is_valid_list_cont = False
            fl_pitch = None
            if raw_block_text_for_check.isdigit():
                logger.info(
                    "DIGIT BLOCK %r y0=%.1f y1=%.1f bottom_thresh=%.1f page_h=%.1f",
                    raw_block_text_for_check,
                    y0,
                    y1,
                    bottom_thresh,
                    page.height,
                )

            if self.list_buffer and not is_visual_caption and not raw_block_text_for_check.isdigit():
                last_item = self.list_buffer[-1]
                item = last_item["item"]
                last_item_page = item.words[-1].page_number if item.words else page.number
                is_page_transition = (last_item_page != page.number) or (self.last_y1 is None) or (y0 < self.last_y1 - 10)

                if is_page_transition and page.images and any(img.bbox[3] <= y0 + 10 for img in page.images):
                    self._flush_all_lists("obraz nad tekstem na nowej stronie")

                if self.list_buffer and self._use_geom() and block.lines:
                    first_line = block.lines[0]
                    self._line_is_monospace = self._is_spans_monospace(first_line.spans)
                    first_v_span = next((s for s in first_line.spans if s.text.strip()), first_line.spans[0] if first_line.spans else None)
                    fl_x0 = first_v_span.bbox[0] if first_v_span else x0
                    fl_x1 = first_line.spans[-1].bbox[2] if first_line.spans else x1
                    fl_y0 = first_line.spans[0].bbox[1] if first_line.spans else y0
                    fl_y1 = first_line.spans[-1].bbox[3] if first_line.spans else y1
                    fl_word_w = (first_v_span.bbox[2] - first_v_span.bbox[0]) if first_v_span else 10.0
                    fl_pitch = (fl_y1 - prev_geom.y1) if (prev_geom and prev_geom.y1 is not None and fl_y1 >= prev_geom.y1) else None
                    fl_font_size = first_v_span.size if first_v_span else None
                    fl_geom = LineGeom(
                        x0=fl_x0,
                        x1=fl_x1,
                        y0=fl_y0,
                        y1=fl_y1,
                        first_word_w=fl_word_w,
                        pitch=fl_pitch,
                        ends_hyphen=bool(raw_block_text_for_check.endswith("-") or raw_block_text_for_check.endswith("–")),
                        font_size=fl_font_size,
                    )
                    is_valid_list_cont = self._is_item_continuation(
                        geom=fl_geom,
                        prev_geom=prev_geom,
                        item=item,
                        ref_pitch=ref_pitch,
                        content_right=content_right,
                        line_text=raw_block_text_for_check,
                        prev_text=prev_line_text,
                        is_justified=prev_line_justified,
                    )
                elif self.list_buffer:
                    last_x0 = last_item["bbox"][0]

                    x_tolerance = 30 if is_page_transition else 20

                    if abs(last_x0 - x0) < x_tolerance or x0 > last_x0:
                        is_valid_list_cont = True

                        if not is_page_transition and self.last_y1 is not None:
                            if (y0 - self.last_y1) > self.MIN_VERTICAL_GAP_LIST:
                                is_valid_list_cont = False

                        strict_x_offset = 20 if is_page_transition else 5
                        if x0 < last_x0 - strict_x_offset:
                            is_valid_list_cont = False

                    if is_valid_list_cont:
                        text_x0 = item.words[1].bbox[0] if len(item.words) > 1 else last_x0
                        ends_sentence = bool(
                            re.search(r'[.!?…]["”»)\]]*$', item.text.rstrip())
                        )
                        if ends_sentence:
                            is_indented = x0 > x0_margin + self.MARGIN_INDENT_THRESH
                            if is_indented and (x0 < text_x0 - 3 or x0 > text_x0 + 5):
                                is_valid_list_cont = False
                            elif not is_indented and x0 <= last_x0 + 5:
                                is_valid_list_cont = False

                    if getattr(self, "in_bibliography_section", False):
                        logger.info(
                            "BIB CONT (legacy): line=%r x0=%.1f last_x0=%.1f -> %s",
                            raw_block_text_for_check[:40],
                            x0,
                            last_x0,
                            is_valid_list_cont,
                        )

            if (
                y1 < self.TOP_MARGIN_THRESH
                or y0 > bottom_thresh
                or (y0 > page.height * 0.88 and re.fullmatch(r"\d{1,4}", raw_block_text_for_check))
            ):
                new_artifact = PageArtifact(
                    artifact_id=block.block_id,
                    type="nr strony (tymczasowo uproszczone)",
                    page_number=page.number,
                    text=temp_text,
                    bbox=list(block.bbox),
                )
                new_doc.floating_elements.page_artifacts.append(new_artifact)
                continue

            if self._use_geom() and any(
                y0r <= (y0 + y1) / 2 <= y1r for y0r, y1r in formula_regions
            ):
                if self.list_buffer:
                    logger.info(
                        "INTERRUPT formula inside list %r: block_id=%s, text=%r",
                        self.list_buffer[0]["item"].text[:30],
                        block.block_id,
                        temp_text[:40],
                    )
                new_doc.floating_elements.page_artifacts.append(
                    PageArtifact(
                        artifact_id=f"eq_{block.block_id}",
                        type="math_formula",
                        page_number=page.number,
                        text=temp_text,
                        bbox=list(block.bbox),
                    )
                )
                prev_geom = None
                ref_pitch = None
                continue

            for line_idx, line in enumerate(block.lines):
                line_bbox = (
                    [
                        line.spans[0].bbox[0],
                        line.bbox[1],
                        line.spans[-1].bbox[2],
                        line.bbox[3],
                    ]
                    if line.spans
                    else line.bbox
                )
                if self.is_inside_table(line_bbox, table_bboxes):
                    continue

                line_y0 = line.spans[0].bbox[1] if line.spans else line.bbox[1]
                line_y1 = line.spans[-1].bbox[3] if line.spans else line.bbox[3]
                tmp_line_text = "".join(s.text for s in line.spans).strip()

                if (
                    line_y1 < self.TOP_MARGIN_THRESH
                    or line_y0 > bottom_thresh
                    or (line_y0 > page.height * 0.88 and re.fullmatch(r"\d{1,4}", tmp_line_text))
                ):
                    new_doc.floating_elements.page_artifacts.append(
                        PageArtifact(
                            artifact_id=f"{block.block_id}_l{line_idx}",
                            type="nr strony (linia)",
                            page_number=page.number,
                            text=tmp_line_text.lower(),
                            bbox=list(line.bbox),
                        )
                    )
                    continue

                if not tmp_line_text:
                    continue
                tmp_line_text_sp = " ".join(
                    s.text.strip() for s in line.spans if s.text.strip()
                )
                self._line_is_monospace = self._is_spans_monospace(line.spans)
                if line_idx == 0 and self._is_bib_heading_line(tmp_line_text_sp):
                    self._empty_paragraph_buffer("wykryto nagłówek bibliografii")
                    self._flush_all_lists("wykryto nagłówek bibliografii")
                    self.in_bibliography_section = True
                    current_marker = None
                    is_valid_list_cont = False
                    hd_text, hd_words, _ = self._extract_words_with_spacing(
                        line, "", [], page.number, 0, line.spans[-1].bbox[2]
                    )
                    hd_block = ParagraphBlock(
                        block_id=block.block_id,
                        content=hd_text.strip(),
                        words=hd_words,
                        type="heading",
                        debug_empty="nagłówek bibliografii (linia)",
                    )
                    hd_block._seq = self._block_seq
                    self._block_seq += 1
                    self.logical_blocks.append(hd_block)
                    self.last_y1 = line.spans[-1].bbox[3]
                    continue

                if line_idx == 0 and getattr(self, "in_bibliography_section", False) and self._is_after_bib_heading_line(tmp_line_text_sp):
                    self._empty_paragraph_buffer("wykryto koniec bibliografii")
                    self._flush_all_lists("wykryto koniec bibliografii")
                    self.in_bibliography_section = False
                    current_marker = None
                    is_valid_list_cont = False
                    hd_text, hd_words, _ = self._extract_words_with_spacing(
                        line, "", [], page.number, 0, line.spans[-1].bbox[2]
                    )
                    hd_block = ParagraphBlock(
                        block_id=block.block_id,
                        content=hd_text.strip(),
                        words=hd_words,
                        type="heading",
                        debug_empty="koniec bibliografii (linia)",
                    )
                    hd_block._seq = self._block_seq
                    self._block_seq += 1
                    self.logical_blocks.append(hd_block)
                    self.last_y1 = line.spans[-1].bbox[3]
                    continue

                line_type, _ = self._classify_block_content(tmp_line_text_sp, current_marker)

                if line_type == "list" and full_text.strip():
                    prev_type, prev_marker = self._classify_block_content(
                        full_text, current_marker
                    )

                    if prev_type == "paragraph":
                        if is_valid_list_cont and self.list_buffer:
                            last_item_data = self.list_buffer[-1]
                            connector = (
                                ""
                                if last_item_data["item"].text.rstrip().endswith("-")
                                else " "
                            )
                            cont_base = len(last_item_data["item"].text) + len(
                                connector
                            )

                            for word in words_info:
                                word.start_char += cont_base
                                word.end_char += cont_base

                            last_item_data["item"].text += connector + full_text.strip()
                            last_item_data["original_text"] += connector + full_text.strip()
                            last_item_data["item"].words.extend(words_info)

                            b = list(block.bbox)
                            last_item_data["item"].bbox = [
                                min(last_item_data["item"].bbox[0], b[0]),
                                min(last_item_data["item"].bbox[1], b[1]),
                                max(last_item_data["item"].bbox[2], b[2]),
                                max(last_item_data["item"].bbox[3], b[3]),
                            ]
                            last_item_data["bbox"] = last_item_data["item"].bbox
                        else:
                            self.paragraph_buffer.append(
                                {
                                    "content": full_text.strip(),
                                    "words": words_info.copy(),
                                    "block_id": block.block_id,
                                }
                            )
                            self._empty_paragraph_buffer("odcięcie wstępu od listy")

                        full_text, words_info = "", []
                        is_valid_list_cont = False

                    elif prev_type == "list":
                        item_x0 = words_info[0].bbox[0] if words_info else block.bbox[0]
                        self._handle_marker_line(prev_marker, item_x0)
                        ref_pitch = None
                        cleaned_text = strip_list_marker(full_text, prev_marker)
                        if not self.list_buffer:
                            item_seq = self._block_seq
                            self._block_seq += 1
                        else:
                            item_seq = self.list_buffer[0].get("_seq", self._block_seq)
                        self.list_buffer.append(
                            {
                                "item": ListItem(
                                    item_id=block.block_id,
                                    marker_type=prev_marker,
                                    text=cleaned_text,
                                    bbox=list(block.bbox),
                                    words=words_info.copy(),
                                ),
                                "words": words_info.copy(),
                                "block_id": block.block_id,
                                "bbox": list(block.bbox),
                                "original_text": full_text,
                                "_seq": item_seq,
                            }
                        )
                        self._adjust_item_word_positions(
                            self.list_buffer[-1]["item"].words,
                            full_text,
                            cleaned_text,
                        )
                        full_text, words_info = "", []
                        ref_pitch = None

                line_x0 = block.bbox[0]
                if line.spans:
                    first_valid_span = next(
                        (s for s in line.spans if s.text.strip()), None
                    )
                    line_x0 = (
                        first_valid_span.bbox[0]
                        if first_valid_span
                        else line.spans[0].bbox[0]
                    )

                line_x1 = line.spans[-1].bbox[2] if line.spans else block.bbox[2]
                current_y0 = line.spans[0].bbox[1] if line.spans else line.bbox[1]
                current_y1 = line.spans[-1].bbox[3] if line.spans else line.bbox[3]

                first_word_w = (first_valid_span.bbox[2] - first_valid_span.bbox[0]) if first_valid_span else 10.0
                pitch = (current_y1 - prev_geom.y1) if (prev_geom and prev_geom.y1 is not None and current_y1 >= prev_geom.y1) else None
                geom = LineGeom(
                    x0=line_x0,
                    x1=line_x1,
                    y0=current_y0,
                    y1=current_y1,
                    first_word_w=first_word_w,
                    pitch=pitch,
                    ends_hyphen=bool(tmp_line_text.endswith("-") or tmp_line_text.endswith("–")),
                    font_size=first_valid_span.size if first_valid_span else None,
                )

                # Detekcja nowego akapitu
                is_new_paragraph, debug_reason = self._detect_paragraph_break(
                    current_y0,
                    current_y1,
                    line_x0,
                    x0_margin,
                    full_text,
                    is_valid_list_cont,
                    geom=geom,
                    prev_geom=prev_geom,
                    content_right=content_right,
                    block_line_x0s=block_line_x0s,
                )

                if is_new_paragraph and full_text.strip():
                    curr_type, _ = self._classify_block_content(full_text, current_marker)
                    next_line_type, _ = self._classify_block_content(
                        tmp_line_text_sp, current_marker
                    )

                    if curr_type == "list" and next_line_type != "list":
                        item_text_x0 = (
                            words_info[1].bbox[0]
                            if len(words_info) > 1
                            else None
                        )
                        aligned_to_text = (
                            item_text_x0 is not None and abs(geom.x0 - item_text_x0) <= 8
                        )
                        if aligned_to_text or not self._use_geom():
                            is_new_paragraph = False

                if is_new_paragraph:
                    if full_text.strip():
                        is_math_type = self.is_math(words_info)
                        if (
                            not getattr(self, "in_bibliography_section", False)
                            and is_math_type > 0
                        ):
                            math_bbox = [
                                min(w.bbox[0] for w in words_info),
                                min(w.bbox[1] for w in words_info),
                                max(w.bbox[2] for w in words_info),
                                max(w.bbox[3] for w in words_info),
                            ]
                            new_doc.floating_elements.page_artifacts.append(
                                PageArtifact(
                                    artifact_id=f"math_{block.block_id}_{self.curr_line}",
                                    type=f"math_{is_math_type}",
                                    page_number=page.number,
                                    text=full_text,
                                    bbox=math_bbox,
                                )
                            )
                        else:
                            prev_type, prev_marker = self._classify_block_content(
                                full_text, current_marker
                            )

                            if prev_type == "list":
                                item_x0 = words_info[0].bbox[0] if words_info else block.bbox[0]
                                self._handle_marker_line(prev_marker, item_x0)
                                ref_pitch = None
                                cleaned_text = strip_list_marker(full_text, prev_marker)
                                if not self.list_buffer:
                                    item_seq = self._block_seq
                                    self._block_seq += 1
                                else:
                                    item_seq = self.list_buffer[0].get("_seq", self._block_seq)
                                self.list_buffer.append(
                                    {
                                        "item": ListItem(
                                            item_id=block.block_id,
                                            marker_type=prev_marker,
                                            text=cleaned_text,
                                            bbox=list(block.bbox),
                                            words=words_info.copy(),
                                        ),
                                        "words": words_info.copy(),
                                        "block_id": block.block_id,
                                        "bbox": list(block.bbox),
                                        "original_text": full_text,
                                        "_seq": item_seq,
                                    }
                                )
                                self._adjust_item_word_positions(
                                    self.list_buffer[-1]["item"].words,
                                    full_text,
                                    cleaned_text,
                                )
                            else:
                                if self.list_buffer and not is_visual_caption:
                                    self._flush_all_lists("nowy akapit wewnątrz bloku")
                                self.paragraph_buffer.append(
                                    {
                                        "content": full_text.strip(),
                                        "words": words_info.copy(),
                                        "block_id": block.block_id,
                                    }
                                )

                    full_text, words_info = "", []

                    if self.paragraph_buffer:
                        self._empty_paragraph_buffer(debug_reason)

                # Ekstrakcja słów do linijki
                full_text, words_info, word_counter = self._extract_words_with_spacing(
                    line, full_text, words_info, page.number, word_counter, line_x1
                )

                prev_geom = geom
                prev_line_text = tmp_line_text
                prev_line_justified = (getattr(line, "alignement", "") == "justified")

            full_text = full_text.strip()
            if not full_text or len(full_text) < 2:
                continue

            is_math_type = self.is_math(words_info)
            if not getattr(self, "in_bibliography_section", False) and is_math_type > 0:
                math_bbox = [
                    min(w.bbox[0] for w in words_info),
                    min(w.bbox[1] for w in words_info),
                    max(w.bbox[2] for w in words_info),
                    max(w.bbox[3] for w in words_info),
                ]
                new_doc.floating_elements.page_artifacts.append(
                    PageArtifact(
                        artifact_id=f"math_{block.block_id}",
                        type=f"math_{is_math_type}",
                        page_number=page.number,
                        text=full_text,
                        bbox=math_bbox,
                    )
                )
                continue

            is_hdr = self.is_header(words_info)
            in_bib = getattr(self, "in_bibliography_section", False)
            if is_hdr and in_bib:
                is_bold = all(w.bold for w in words_info)
                avg_size = sum(w.size for w in words_info) / len(words_info) if words_info else 0
                if not (is_bold or avg_size > 12.5):
                    is_hdr = False

            heading_key = self._strip_heading_number(full_text)
            if re.match(r"^\s*\d+\.\d+", full_text):
                heading_key = full_text.strip().upper()
            is_short = len(full_text) < 80 and len(words_info) <= 8
            if is_short and (
                heading_key in self.BIB_HEADINGS
                or (in_bib and self.AFTER_BIB_HEADING_RE.match(heading_key))
            ):
                is_hdr = True

            if is_hdr:
                self._empty_paragraph_buffer("wykryto nagłówek")
                self._flush_all_lists("wykryto nagłówek")

                if heading_key.rstrip(":. ").startswith(self.BIB_HEADINGS):
                    self.in_bibliography_section = True
                elif in_bib and not (
                    self.AFTER_BIB_HEADING_RE.match(heading_key)
                    or self.HEADING_NUM_RE.match(full_text)
                ):

                    self.in_bibliography_section = True
                else:
                    self.in_bibliography_section = False

                hd_block = ParagraphBlock(
                    block_id=block.block_id,
                    content=full_text,
                    words=words_info,
                    type="heading",
                    debug_empty="wykryto nagłówek",
                )
                hd_block._seq = self._block_seq
                self._block_seq += 1
                self.logical_blocks.append(hd_block)
                continue

            self._line_is_monospace = self._is_spans_monospace(
                [s for l in block.lines for s in l.spans]
            )
            block_type, marker_type = self._classify_block_content(full_text, current_marker)

            if block_type == "list":
                self._empty_paragraph_buffer("wykryto listę")
                item_x0 = words_info[0].bbox[0] if words_info else block.bbox[0]
                self._handle_marker_line(marker_type, item_x0)
                ref_pitch = None
                cleaned_text = strip_list_marker(full_text, marker_type)
                if not self.list_buffer:
                    item_seq = self._block_seq
                    self._block_seq += 1
                else:
                    item_seq = self.list_buffer[0].get("_seq", self._block_seq)
                self.list_buffer.append(
                    {
                        "item": ListItem(
                            item_id=block.block_id,
                            marker_type=marker_type,
                            text=cleaned_text,
                            bbox=list(block.bbox),
                            words=words_info,
                        ),
                        "words": words_info,
                        "block_id": block.block_id,
                        "bbox": list(block.bbox),
                        "original_text": full_text,
                        "_seq": item_seq,
                    }
                )
                self._adjust_item_word_positions(self.list_buffer[-1]["item"].words, full_text, cleaned_text)
                ref_pitch = None
            elif is_valid_list_cont:
                last_item_data = self.list_buffer[-1]
                connector = (
                    "" if last_item_data["item"].text.rstrip().endswith("-") else " "
                )
                cont_base = len(last_item_data["item"].text) + len(connector)
                for word in words_info:
                    word.start_char += cont_base
                    word.end_char += cont_base
                last_item_data["item"].text += connector + full_text
                last_item_data["original_text"] += connector + full_text.strip()
                last_item_data["item"].words.extend(words_info)
                b = list(block.bbox)
                last_item_data["item"].bbox = [
                    min(last_item_data["item"].bbox[0], b[0]),
                    min(last_item_data["item"].bbox[1], b[1]),
                    max(last_item_data["item"].bbox[2], b[2]),
                    max(last_item_data["item"].bbox[3], b[3]),
                ]
                last_item_data["bbox"] = last_item_data["item"].bbox
                if fl_pitch and fl_pitch > 5.0:
                    ref_pitch = fl_pitch
            else:
                self._flush_all_lists("brak kontynuacji listy")
                ref_pitch = None
                self.paragraph_buffer.append(
                    {
                        "content": full_text,
                        "words": words_info,
                        "block_id": block.block_id,
                    }
                )

                if is_visual_caption:
                    self._empty_paragraph_buffer("odcięcie - koniec bloku podpisu")

        for table in page.tables:
            ve = VisualElement(
                element_id=id(table),
                type="table",
                page_number=page.number,
                bbox=list(table.bbox),
                caption={"text": table.description} if getattr(table, "description", "") else {},
                table_data=table.data,
                format={"num_rows": table.row_count, "num_columns": table.col_count},
            )
            new_doc.floating_elements.visual_elements.append(ve)

        for img in page.images:
            ve = VisualElement(
                element_id=id(img),
                type="image",
                page_number=page.number,
                bbox=list(img.bbox),
                caption={"text": img.description} if getattr(img, "description", "") else {},
            )
            new_doc.floating_elements.visual_elements.append(ve)



    def _merge_adjacent_headings(self):
        """
        Iterates through the logical blocks and merges consecutive heading blocks
        into a single heading if they share matching typographical features
        (such as font, size, and horizontal alignment).
        """
        i = 0
        while i < len(self.logical_blocks) - 1:
            curr = self.logical_blocks[i]
            nxt = self.logical_blocks[i + 1]

            if (
                getattr(curr, "type", None) == "heading"
                and getattr(nxt, "type", None) == "heading"
            ):
                if not getattr(curr, "words", []) or not getattr(nxt, "words", []):
                    i += 1
                    continue

                curr_font = curr.words[0].font
                nxt_font = nxt.words[0].font
                curr_size = curr.words[0].size
                nxt_size = nxt.words[0].size

                font_matches = (curr_font == nxt_font) and (
                    abs(curr_size - nxt_size) < 1.0
                )

                if font_matches:
                    separator = " "
                    old_len = len(curr.content) + len(separator)
                    curr.content += separator + nxt.content

                    for word in nxt.words:
                        word.start_char += old_len
                        word.end_char += old_len
                        word.word_index += len(curr.words)
                        curr.words.append(word)

                    self.logical_blocks.pop(i + 1)
                    continue
            i += 1

    def _tag_special_lists(self, old_doc: DocumentData):
        """
        Post-processing routine that identifies and retags text blocks as 'toc' (Table of Contents),
        'tot' (Table of Tables), or 'tof' (Table of Figures) if their bounding boxes intersect
        with the predefined special list coordinates from the raw document extraction.

        Args:
            old_doc (DocumentData): The raw document data containing the original TOC/TOT/TOF entries.
        """
        special_bboxes = {}

        def add_entries(entries, tag):
            if not entries:
                return
            for entry in entries:
                if entry.src_page == -1:
                    continue
                if entry.src_page not in special_bboxes:
                    special_bboxes[entry.src_page] = {"toc": [], "tot": [], "tof": []}
                special_bboxes[entry.src_page][tag].append(entry.bbox)

        add_entries(old_doc.toc.entries if old_doc.toc else [], "toc")
        add_entries(old_doc.tot.entries if old_doc.tot else [], "tot")
        add_entries(old_doc.tof.entries if old_doc.tof else [], "tof")

        if not special_bboxes:
            return

        for block in self.logical_blocks:
            words = getattr(block, "words", [])
            if not words:
                continue

            page_num = words[0].page_number
            if page_num not in special_bboxes:
                continue

            bx0 = min(w.bbox[0] for w in words)
            by0 = min(w.bbox[1] for w in words)
            bx1 = max(w.bbox[2] for w in words)
            by1 = max(w.bbox[3] for w in words)

            matched_tag = None
            for tag in ["toc", "tot", "tof"]:
                for t_bbox in special_bboxes[page_num][tag]:
                    ix0 = max(bx0, t_bbox[0] - 5)
                    iy0 = max(by0, t_bbox[1] - 5)
                    ix1 = min(bx1, t_bbox[2] + 5)
                    iy1 = min(by1, t_bbox[3] + 5)

                    if ix0 < ix1 and iy0 < iy1:  #
                        matched_tag = tag
                        break
                if matched_tag:
                    break

            if matched_tag:
                block.type = matched_tag

    def _tag_visual_descriptions(self, new_doc):
        """
        Change block types to image_description and table_description. 
        Protect from connecting description to lists below.
        """
        known_captions = set()
        for v in new_doc.floating_elements.visual_elements:
            caption_dict = getattr(v, "caption", {})
            if caption_dict and isinstance(caption_dict, dict):
                text = caption_dict.get("text", "").strip()
                if text:
                    known_captions.add(text)

        if not known_captions:
            return

        list_marker_pattern = re.compile(
            r"^(?:[-–—•●○■]|\d{1,3}[\.)]|[a-z]\))\s*", re.IGNORECASE
        )

        for block in self.logical_blocks:
            if getattr(block, "type", None) in ["paragraph", "heading"]:
                content = getattr(block, "content", "").strip()
                if not content:
                    continue

                first_few_words = " ".join(w.text for w in getattr(block, "words", [])[:3]).strip()
                if (
                    list_marker_pattern.match(content)
                    or list_marker_pattern.match(first_few_words)
                    or self._classify_block_content(first_few_words)[0] == "list"
                    or self._classify_block_content(content)[0] == "list"
                ):
                    continue

                if content in known_captions:
                    if content.lower().startswith(("tabela", "tab.", "table", "tablica", "tab")):
                        block.type = "table_description"
                    else:
                        block.type = "image_description"

    def _pair_descriptions_with_visuals(self, new_doc):
        """
        Post-processing routine that pairs identified caption blocks ('table_description' or
        'image_description') with their corresponding visual elements (tables or images)
        on the same page by calculating and minimizing the vertical distance between them.

        Args:
            new_doc (FinalDocument): The structured document containing the floating visual elements to be updated.
        """
        visuals = new_doc.floating_elements.visual_elements

        for block in self.logical_blocks:
            if getattr(block, "type", None) not in [
                "table_description",
                "image_description",
            ]:
                continue

            words = getattr(block, "words", [])
            if not words:
                continue

            page_num = words[0].page_number

            desc_y0 = min(w.bbox[1] for w in words)
            desc_y1 = max(w.bbox[3] for w in words)

            target_type = "table" if block.type == "table_description" else "image"

            candidates = [
                v
                for v in visuals
                if v.type == target_type and v.page_number == page_num
            ]

            if not candidates:
                continue

            closest_visual = None
            min_distance = float("inf")

            for candidate in candidates:
                v_y0, v_y1 = candidate.bbox[1], candidate.bbox[3]

                if desc_y1 <= v_y0:
                    dist = v_y0 - desc_y1
                elif desc_y0 >= v_y1:
                    dist = desc_y0 - v_y1
                else:
                    dist = 0

                if dist < min_distance:
                    min_distance = dist
                    closest_visual = candidate

            if closest_visual:
                closest_visual.caption = {"text": block.content.strip()}

    def _verify_bibliography_references(self):
        """
        Post-processing routine that collects valid index numbers from the bibliography
        section by inspecting the first word (marker) of each bibliography item. It then
        scans the entire document to validate in-text citations against this collected set.
        Invalid or missing references are flagged by setting 'incorrect_bibliography = 1'
        on the respective WordInfo objects.
        """
        valid_bib_numbers = set()

        for block in self.logical_blocks:
            if getattr(block, "is_bibliography", False) and hasattr(block, "items"):
                for item in block.items:
                    # Wpisy bez markera (marker_type="none") nie mają numeru - pierwszym słowem jest autor
                    if getattr(item, "marker_type", None) == "none":
                        continue
                    if item.words:
                        marker_text = item.words[0].text

                        match = re.search(r"\d+", marker_text)
                        if match:
                            valid_bib_numbers.add(match.group())

        # Bibliografia nienumerowana (np. APA) - brak numerów do weryfikacji cytowań [n]
        if not valid_bib_numbers:
            return

        citation_pattern = re.compile(r"\[([\d\s,\-]+)\]")

        math_interval_context = re.compile(
            r"\b(przedział\w*|przedziale\w*|zakres\w*|zakresie\w*|zbior\w*|zbiorze\w*|dziedzin\w*|wartość\w*|wartości\w*|interval|range)\s*$",
            re.IGNORECASE)

        def extract_numbers_from_citation(cit_str):
            """
            Parses a citation string and extracts individual reference numbers,
            handling both comma-separated values and hyphenated ranges.

            Args:
                cit_str (str): The raw citation string (e.g., '1, 3, 5-7').

            Returns:
                set[str]: A set of unique reference numbers extracted from the string.
            """
            nums = set()
            for part in cit_str.split(","):
                part = part.strip()
                if "-" in part:
                    try:
                        start_s, end_s = part.split("-")
                        for i in range(int(start_s), int(end_s) + 1):
                            nums.add(str(i))
                    except ValueError:
                        pass
                else:
                    num_match = re.search(r"\d+", part)
                    if num_match:
                        nums.add(num_match.group())
            return nums

        for block in self.logical_blocks:
            if (
                getattr(block, "is_bibliography", False)
                or getattr(block, "type", None) == "heading"
                or getattr(block, "type", False) == "code_snippet"
            ):
                continue

            search_targets = []
            if hasattr(block, "items"):
                for item in block.items:
                    search_targets.append((item.text, item.words))
            else:
                search_targets.append(
                    (getattr(block, "content", ""), getattr(block, "words", []))
                )

            for text, words in search_targets:
                if not text or not words:
                    continue

                for match in citation_pattern.finditer(text):
                    cit_content = match.group(1)

                    if len(re.findall(r"\d+", cit_content)) > 1:
                        continue

                    nums = extract_numbers_from_citation(cit_content)

                    if "0" in nums or any(n == "0" for n in nums):
                        continue
                    
                    if nums and not nums.issubset(valid_bib_numbers):
                        start_pos, end_pos = match.span()
                        for word in words:
                            if word.start_char < end_pos and word.end_char > start_pos:
                                word.incorrect_bibliography = 1

    def _extract_acronyms_to_schema(self, new_doc):
        """
        Post-processing routine that identifies the acronyms, abbreviations, or symbols
        section within the document. It reconstructs fragmented lines and uses complex
        regular expressions to parse and extract the terms along with their corresponding
        definitions, appending them as structured AcronymItem objects to the document's
        reference schema.

        Args:
            new_doc (FinalDocument): The structured document where extracted acronyms will be saved.
        """

        # BAZA AKRONIMU
        acr_dash_pattern = (
            r"^(?![a-ząćęłńóśźż]{3,})(?!\d+\s+[-–—−‐:=])(.{1,40}?)\s+[-–—−‐:=]\s+"
        )
        acr_space_pattern = r"^(?=[A-Za-zĄĆĘŁŃÓŚŹŻ0-9\-/\u0370-\u03FF]*[A-Za-zĄĆĘŁŃÓŚŹŻ\u0370-\u03FF])(?:[A-ZĄĆĘŁŃÓŚŹŻ0-9\u0370-\u03FF]|[a-ząćęłńóśźż][A-ZĄĆĘŁŃÓŚŹŻ\u0370-\u03FF])[A-Za-zĄĆĘŁŃÓŚŹŻ0-9\-/\u0370-\u03FF]{0,20}"
        mc = r"\u0370-\u03FF\u2100-\u214F\u2200-\u22FF\U0001D400-\U0001D7FF"

        # BAZA SYMBOLI
        math_symbol_pattern = (
            r"^([A-Za-zĄĆĘŁŃÓŚŹŻ0-9\-/\^\|_=<>\.,\(\)\*\s"
            + mc
            + r"]{1,30})\s+[-–—−‐:=]?\s*(?=[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ])"
        )

        in_acronym_section = False
        acronym_words = []
        fallback_blocks = []
        has_words_data = False

        for block in self.logical_blocks:
            header_text = (
                getattr(block, "content", "").strip().upper()
                if not isinstance(block, dict)
                else block.get("content", "").strip().upper()
            )
            block_type = (
                getattr(block, "type", None)
                if not isinstance(block, dict)
                else block.get("type", None)
            )

            pattern_target = r"\b(ACRONYM|ACRONYMS|SKRÓT|SKRÓTY|SKRÓTÓW|ABBREVIATION|ABBREVIATIONS|OZNACZEŃ|OZNACZENIA|SYMBOL|SYMBOLI|SYMBOLE)\b"

            # sprawdzanie czy blok jest spisem akronimów
            is_header = False
            if block_type == "heading":
                is_header = True
            elif len(header_text) < 80 and re.search(pattern_target, header_text):
                is_header = True

            if is_header:
                is_target = (
                    re.search(pattern_target, header_text)
                    and "SKRÓT DYPLOMU" not in header_text
                    and "SKRÓT PRACY" not in header_text
                )

                if is_target:
                    in_acronym_section = True
                    continue
                elif in_acronym_section:
                    if re.match(r"^\d+\.", header_text) or any(
                        kw in header_text
                        for kw in [
                            "WSTĘP",
                            "SPIS",
                            "BIBLIOGRAFIA",
                            "ROZDZIAŁ",
                            "STRESZCZENIE",
                            "ABSTRACT",
                            "SUMMARY",
                            "PODSUMOWANIE",
                            "WPROWADZENIE",
                        ]
                    ):
                        in_acronym_section = False
                        continue

            if in_acronym_section:
                fallback_blocks.append(block)

                # Nadpisywanie typu bloków na "acronyms"
                if block_type != "heading":
                    if isinstance(block, dict):
                        block["type"] = "acronyms"
                    else:
                        try:
                            block.type = "acronyms"
                        except Exception:
                            pass

                def fetch_words(obj):
                    """
                    Recursively traverses a structured object or dictionary to locate
                    and extract all nested word elements.

                    Args:
                        obj (Union[dict, object]): The data structure (block, item, or line) to search.

                    Returns:
                        list: A flattened list of word objects or dictionaries found within the structure.
                    """
                    w = []
                    try:
                        if isinstance(obj, dict):
                            if obj.get("words"):
                                w.extend(obj["words"])
                            if obj.get("items"):
                                for item in obj["items"]:
                                    w.extend(fetch_words(item))
                            if obj.get("lines"):
                                for line in obj["lines"]:
                                    if isinstance(line, dict) and line.get("spans"):
                                        w.extend(line["spans"])
                        else:
                            if getattr(obj, "words", None):
                                w.extend(obj.words)
                            if getattr(obj, "items", None):
                                for item in obj.items:
                                    w.extend(fetch_words(item))
                            if getattr(obj, "lines", None):
                                for line in obj.lines:
                                    if hasattr(line, "spans"):
                                        w.extend(line.spans)
                                    elif isinstance(line, dict) and line.get("spans"):
                                        w.extend(line["spans"])
                    except Exception:
                        pass
                    return w

                words = fetch_words(block)

                if words:
                    has_words_data = True
                    acronym_words.extend(words)

        reconstructed_lines = []

        if has_words_data and acronym_words:

            def get_y0(w):
                """
                Safely retrieves the starting Y-coordinate (y0) from a word object's bounding box.

                Args:
                    w (Union[dict, object]): The word object or dictionary containing bounding box data.

                Returns:
                    float: The starting Y-coordinate, or 0.0 if the extraction fails.
                """
                try:
                    bbox = (
                        w.get("bbox")
                        if isinstance(w, dict)
                        else getattr(w, "bbox", None)
                    )
                    if bbox and len(bbox) >= 4:
                        return (float(bbox[1]) + float(bbox[3])) / 2.0
                    elif bbox and len(bbox) >= 2:
                        return float(bbox[1])
                    return 0.0
                except Exception:
                    return 0.0

            def get_x0(w):
                """
                Safely retrieves the starting X-coordinate (x0) from a word object's bounding box.

                Args:
                    w (Union[dict, object]): The word object or dictionary containing bounding box data.

                Returns:
                    float: The starting X-coordinate, or 0.0 if the extraction fails.
                """
                try:
                    bbox = (
                        w.get("bbox")
                        if isinstance(w, dict)
                        else getattr(w, "bbox", None)
                    )
                    return float(bbox[0]) if bbox and len(bbox) >= 1 else 0.0
                except Exception:
                    return 0.0

            def get_page(w):
                """
                Safely retrieves the page number associated with a word object.

                Args:
                    w (Union[dict, object]): The word object or dictionary containing page data.

                Returns:
                    int: The page number, or 0 if the extraction fails.
                """
                try:
                    p = (
                        w.get("page_number")
                        if isinstance(w, dict)
                        else getattr(w, "page_number", None)
                    )
                    return int(p) if p is not None else 0
                except Exception:
                    return 0

            def get_text(w):
                """
                Safely retrieves the string text content from a word object.

                Args:
                    w (Union[dict, object]): The word object or dictionary containing text data.

                Returns:
                    str: The extracted text string, or an empty string if the extraction fails.
                """
                try:
                    if isinstance(w, dict):
                        return str(w.get("text", ""))
                    return str(getattr(w, "text", ""))
                except Exception:
                    return ""

            unique_words = []
            seen_coords = set()
            for w in acronym_words:
                txt = get_text(w)
                if not txt or not txt.strip():
                    continue
                coord_key = (txt, round(get_x0(w), 1), round(get_y0(w), 1), get_page(w))
                if coord_key not in seen_coords:
                    seen_coords.add(coord_key)
                    unique_words.append(w)

            acronym_words = unique_words
            acronym_words.sort(key=lambda w: (get_page(w), get_y0(w)))

            current_line_words = []
            current_y = None
            current_page = None
            tolerance = 8.0

            for word in acronym_words:
                y0 = get_y0(word)
                page = get_page(word)

                if current_y is None:
                    current_y = y0
                    current_page = page
                    current_line_words.append(word)
                elif current_page == page and abs(y0 - current_y) <= tolerance:
                    current_line_words.append(word)
                else:
                    current_line_words.sort(key=get_x0)
                    line_str = " ".join([get_text(w) for w in current_line_words])
                    # Zapisywanie tekstu razem z zapamiętaną stroną
                    reconstructed_lines.append((line_str, current_page))

                    current_line_words = [word]
                    current_y = y0
                    current_page = page

            if current_line_words:
                current_line_words.sort(key=get_x0)
                line_str = " ".join([get_text(w) for w in current_line_words])
                reconstructed_lines.append((line_str, current_page))

        else:
            for block in fallback_blocks:
                b_type = (
                    getattr(block, "type", None)
                    if not isinstance(block, dict)
                    else block.get("type", getattr(block, "block_type", None))
                )

                # Wyciągnięcie numeru strony z bloku, jeśli jest dostępny
                try:
                    b_page = (
                        block.get("page_number")
                        if isinstance(block, dict)
                        else getattr(block, "page_number", 0)
                    )
                    b_page = int(b_page) if b_page is not None else 0
                except Exception:
                    b_page = 0

                if b_type in [
                    "acronyms",
                    "paragraph",
                    "math",
                    "list",
                    "text",
                    "table",
                    "heading",
                ]:
                    content = getattr(block, "content", "")
                    if not content:
                        continue
                    for raw_line in content.split("\n"):
                        if "|" in raw_line:
                            raw_line = re.sub(r"^\|\s*", "", raw_line)
                            raw_line = re.sub(r"\s*\|$", "", raw_line)
                            raw_line = re.sub(r"\s*\|\s*[-–—−‐:=]?\s*", " - ", raw_line)
                            if set(raw_line.strip()).issubset({"-", "|", " "}):
                                continue
                        raw_line = raw_line.replace("$", "")
                        reconstructed_lines.append((raw_line, b_page))

        final_lines = []

        # Sprawdzamy myślniki pobierając tylko tekst
        uses_dashes = any(
            re.search(acr_dash_pattern, line[0].strip()) for line in reconstructed_lines
        )

        for raw_line, line_page in reconstructed_lines:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if re.search(r"(\.\s*){3,}", raw_line):
                continue

            parts = [raw_line]
            if uses_dashes:
                parts = re.split(
                    r"\s+(?=[A-Za-zĄĆĘŁŃÓŚŹŻ0-9\-]{2,15}\s+[-–—−‐:=]\s+)", raw_line
                )

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                is_new = False
                if uses_dashes:
                    if re.match(acr_dash_pattern, part):
                        is_new = True
                    else:
                        m = re.match(math_symbol_pattern, part)
                        if m and re.search(r"[" + mc + r"]", m.group(1)):
                            is_new = True
                else:
                    if re.match(acr_space_pattern + r"\s+[A-ZĄĆĘŁŃÓŚŹŻ]", part):
                        is_new = True
                    else:
                        m = re.match(math_symbol_pattern, part)
                        if m and re.search(r"[" + mc + r"]", m.group(1)):
                            is_new = True

                if is_new or not final_lines:
                    final_lines.append([part, line_page])
                else:
                    final_lines[-1][0] += " " + part

        for line_data in final_lines:
            line = line_data[0]
            line_page = line_data[1]  # Wyciągamy przechwyconą stronę do zapisu!

            match = None
            if uses_dashes:
                match = re.match(acr_dash_pattern + r"(.*)$", line)
                if not match:
                    m = re.match(
                        r"^([A-Za-zĄĆĘŁŃÓŚŹŻ0-9\-/\^\|_=<>\.,\(\)\*\s"
                        + mc
                        + r"]{1,30})\s+[-–—−‐:=]?\s*(.*)$",
                        line,
                    )
                    if m and re.search(r"[" + mc + r"]", m.group(1)):
                        match = m

            if not match:
                match = re.match(r"^(" + acr_space_pattern + r")\s+(.*)$", line)

            if not match:
                m = re.match(
                    r"^([A-Za-zĄĆĘŁŃÓŚŹŻ0-9\-/\^\|_=<>\.,\(\)\*\s"
                    + mc
                    + r"]{1,30})\s+(.*)$",
                    line,
                )
                if m and re.search(r"[" + mc + r"]", m.group(1)):
                    match = m

            if match:
                acronym = match.group(1).strip()
                raw_definition = match.group(2).strip()

                raw_definition = re.sub(r"\)\s*\)$", ")", raw_definition)

                def_match = re.match(
                    r"^(.*?)(?:\.\s*([\d,\-\s\u2013\u2014]+))?$", raw_definition
                )

                if def_match:
                    definition = def_match.group(1).strip()
                    pages = def_match.group(2).strip() if def_match.group(2) else ""
                else:
                    definition = raw_definition
                    pages = ""

                definition = re.sub(r"^[-–—−‐:=]\s*", "", definition)

                if len(definition) < 2 or len(definition) > 350:
                    continue

                new_item = AcronymItem(
                    acronym=acronym,
                    definition=definition,
                    pages=pages,
                    bbox=[],
                    words=[],
                    src_page=line_page,
                )
                new_doc.reference_sections.acronyms.append(new_item)

    def _verify_caption_sources(self, new_doc):
        """
        Post-processing routine that verifies whether table and image captions contain
        proper source attributions. It checks for the presence of bibliography citations
        (e.g., '[1]') or specific source-indicating keywords (e.g., 'source', 'own elaboration').
        If neither is found, it flags the caption block or visual element by setting
        'incorrect_caption = 1'.

        Args:
            new_doc (FinalDocument): The structured document containing the visual elements to be verified.
        """
        citation_pattern = re.compile(r"\[\s*\d+[\d\s,\-]*(\s*,\s*(?:s\.?|p\.?|pp\.?)\s*\d+(?:-\d+)?)*\s*\]", re.IGNORECASE)

        source_pattern = re.compile(
            r"\b(źródło|zródło|zrodlo|source|opracowanie własne|na podstawie)\b",
            re.IGNORECASE,
        )

        for block in self.logical_blocks:
            if getattr(block, "type", None) in ["image_description"]:
                content = getattr(block, "content", "")
                if not content:
                    continue

                has_citation = bool(citation_pattern.search(content))
                has_keyword = bool(source_pattern.search(content))

                if not (has_citation or has_keyword):
                    block.incorrect_caption = 1
                else:
                    block.incorrect_caption = 0

        for visual in new_doc.floating_elements.visual_elements:
            caption_dict = getattr(visual, "caption", {})
            if caption_dict and isinstance(caption_dict, dict):
                content = caption_dict.get("text", "")
                if content:
                    has_citation = bool(citation_pattern.search(content))
                    has_keyword = bool(source_pattern.search(content))

                    if not (has_citation or has_keyword):
                        visual.incorrect_caption = 1
                    else:
                        visual.incorrect_caption = 0

    def _fix_offsets(self):
        """
        Post-processing for wrongly seperated words. Function takes full blocks content, searches for 
        issues in parsing, fixes them and writes arrys in place (so that references in both list and paragraph blocks stay the same).
        """
        for block in self.logical_blocks:
            original_words = list(getattr(block, "words", []))
            removed = []

            if isinstance(block, ListBlock):
                for item in block.items:
                    item.text, item_removed = self._merge_split_words(item.text, item.words)
                    removed.extend(item_removed)
            elif isinstance(block, ParagraphBlock):
                block.content, removed = self._merge_split_words(block.content, block.words)

            if not removed:
                continue

            removed_ids = {id(w) for w in removed}
            if isinstance(block, ListBlock):
                for item in block.items:
                    item.words[:] = [w for w in item.words if id(w) not in removed_ids]
                block.content = "\n".join(item.text for item in block.items)
            block.words[:] = [w for w in block.words if id(w) not in removed_ids]
            self._shift_word_indices(original_words, removed_ids)


    def _merge_split_words(self, text, words):
        """
        Glues every word starting with ą/ę to the preceding word when there is space befere them on the same line.
        """
        split_spaces = {m.start() for m in self.SPLIT_POLISH_SIGNS_RE.finditer(text)}
        if not split_spaces:
            return text, []

        removed = []
        deleted_spaces = []
        target = None
        for word in words:
            space = word.start_char - 1
            if (
                target is not None
                and space in split_spaces
                and target.end_char == space
                and text[word.start_char:word.start_char + 1] == word.text[:1]
                and (target.page_number, target.line) == (word.page_number, word.line)
            ):
                target.text += word.text
                target.end_char = word.end_char
                target.bbox = [
                    min(target.bbox[0], word.bbox[0]),
                    min(target.bbox[1], word.bbox[1]),
                    max(target.bbox[2], word.bbox[2]),
                    max(target.bbox[3], word.bbox[3]),
                ]
                target.incorrect_bibliography = max(
                    target.incorrect_bibliography, word.incorrect_bibliography
                )
                deleted_spaces.append(space)
                removed.append(word)
            else:
                target = word

        if not removed:
            return text, []

        removed_ids = {id(w) for w in removed}
        for word in words:
            if id(word) in removed_ids:
                continue
            word.start_char -= sum(space_place < word.start_char for space_place in deleted_spaces)
            word.end_char -= sum(space_place < word.end_char for space_place in deleted_spaces)

        deleted = set(deleted_spaces)
        new_text = "".join(char for i, char in enumerate(text) if i not in deleted)
        return new_text, removed


    def _shift_word_indices(self, words, removed_ids):
        """
        Decrements word_index of the words following removed ones. In lists the
        numbering restarts with every raw text block, so the shift is reset when the index stops growing.
        """
        shift = 0
        prev_index = None
        for word in words:
            if prev_index is not None and word.word_index <= prev_index:
                shift = 0
            prev_index = word.word_index
            if id(word) in removed_ids:
                shift += 1
            else:
                word.word_index -= shift

    def _assign_unique_ids(self):
        """
        Renumbers block_id (unique in the document), item_id (unique in the list) and word_index (unique in the block).
        """
        for block_no, block in enumerate(self.logical_blocks):
            block.block_id = f"list_{block_no}" if isinstance(block, ListBlock) else block_no
            for word_no, word in enumerate(block.words):
                word.word_index = word_no
            for item_no, item in enumerate(getattr(block, "items", [])):
                item.item_id = item_no


# Funkcje zewnętrzne

MULTISPACE_RE = re.compile(r"\s+")


def clean_ws(text: str) -> str:
    """
    Cleans up whitespace and hidden characters in a given text string.
    It removes soft hyphens, replaces non-breaking spaces with standard spaces,
    and collapses multiple consecutive spaces into a single space.

    Args:
        text (str): The raw text string to be cleaned.

    Returns:
        str: The cleaned, single-spaced, and stripped text string.
    """
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    return MULTISPACE_RE.sub(" ", text).strip()


def get_plain_text(pdf_path):
    """
    Extracts and compiles a clean, continuous plain text representation of a PDF document,
    filtering out non-narrative elements like lists, captions, and mathematical formulas.

    Args:
        pdf_path (Union[str, Path]): The file path to the target PDF document.

    Returns:
        str: A single string containing the cleaned narrative text of the document.
    """
    raw_doc = extractPDF(str(pdf_path))

    mapper = PDFMapper()
    mapped_doc = mapper.map_to_schema(raw_doc)

    parts = []
    for block in mapped_doc.logical_blocks:
        block_type = getattr(block, "type", None)
        text = clean_ws(getattr(block, "content", "") or "")

        if not text:
            continue

        if block_type in {"list", "table_description", "image_description", "math"}:
            continue

        parts.append(text)

    return clean_ws(" ".join(parts))


def get_acronyms_lut(doc) -> dict:
    """
    Generates a Look-Up Table (LUT) mapping acronyms to their definitions based on
    the parsed reference sections of the document.

    Args:
        doc (FinalDocument): The fully structured document schema containing acronym data.

    Returns:
        dict: A dictionary where keys are acronyms (str) and values are their definitions (str).
    """
    return {item.acronym: item.definition for item in doc.reference_sections.acronyms}


def get_acronym_pages(doc) -> list[int]:
    """
    Retrieves a unique, sorted list of page numbers where acronym or abbreviation
    sections were detected in the document.

    Args:
        doc (FinalDocument): The fully structured document schema containing acronym data.

    Returns:
        list[int]: A sorted list of valid, positive page numbers.
    """
    pages = [item.src_page for item in doc.reference_sections.acronyms]
    valid_pages = sorted(list(set(p for p in pages if p > 0)))
    return valid_pages
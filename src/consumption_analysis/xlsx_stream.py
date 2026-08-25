"""Row-at-a-time reader for very large .xlsx files.

Utility billing extracts run to hundreds of megabytes of sheet XML - far more
than openpyxl or pandas will comfortably hold. This pulls rows straight out of
the zipped XML and discards each element after it is yielded, so memory stays
flat regardless of file size.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterator
from xml.etree.ElementTree import iterparse

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_COL_RE = re.compile(r"([A-Z]+)")


def col_index(ref: str) -> int:
    """'AB12' -> 28"""
    n = 0
    for ch in _COL_RE.match(ref).group(1):
        n = n * 26 + ord(ch) - 64
    return n


class Workbook:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        self._sheets = self._read_sheet_index()
        self._sst: list[str] | None = None

    def _read_sheet_index(self) -> dict[str, str]:
        book = self.zip.read("xl/workbook.xml").decode("utf8", "ignore")
        rels = dict(
            re.findall(
                r'Id="([^"]*)"[^>]*Target="([^"]*)"',
                self.zip.read("xl/_rels/workbook.xml.rels").decode("utf8", "ignore"),
            )
        )
        out = {}
        for name, rid in re.findall(r'<sheet[^>]*name="([^"]*)"[^>]*r:id="([^"]*)"', book):
            name = (name.replace("&amp;", "&").replace("&lt;", "<")
                        .replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'"))
            out[name] = "xl/" + rels[rid].lstrip("/")
        return out

    @property
    def sheet_names(self) -> list[str]:
        return list(self._sheets)

    @property
    def shared_strings(self) -> list[str]:
        if self._sst is None:
            out: list[str] = []
            try:
                handle = self.zip.open("xl/sharedStrings.xml")
            except KeyError:
                self._sst = out
                return out
            for _, el in iterparse(handle, ("end",)):
                if el.tag == NS + "si":
                    out.append("".join(t.text or "" for t in el.iter(NS + "t")))
                    el.clear()
            self._sst = out
        return self._sst

    def rows(self, sheet: str, first: int = 1, last: int = 10**9) -> Iterator[tuple[int, dict[str, str]]]:
        """Yield (row_number, {column_letter: cached_value}). Blank cells are omitted."""
        if sheet not in self._sheets:
            raise KeyError(f"no sheet named {sheet!r}; have {self.sheet_names}")
        sst = self.shared_strings
        for _, el in iterparse(self.zip.open(self._sheets[sheet]), ("end",)):
            if el.tag != NS + "row":
                continue
            r = int(el.get("r"))
            if r < first:
                el.clear()
                continue
            if r > last:
                el.clear()
                break
            row: dict[str, str] = {}
            for c in el.iter(NS + "c"):
                v = c.find(NS + "v")
                if v is None or v.text is None:
                    continue
                text = sst[int(v.text)] if c.get("t") == "s" else v.text
                ref = c.get("r")
                row[ref[: len(ref) - len(str(r))]] = text
            el.clear()
            yield r, row

    def table(self, sheet: str, header_row: int = 1, first_data_row: int | None = None):
        """Yield dicts keyed by header label, skipping fully blank rows."""
        first_data_row = first_data_row or header_row + 1
        header: dict[str, str] = {}
        for r, row in self.rows(sheet, first=header_row):
            if r == header_row:
                header = {col: str(v).strip() for col, v in row.items()}
                continue
            if r < first_data_row or not row:
                continue
            yield {header[col]: v for col, v in row.items() if col in header}

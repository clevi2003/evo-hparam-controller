import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
from .logging_base import BaseAppender, _ensure_parent_dir, _normalize_row


class CSVAppender(BaseAppender):
    def __init__(self, path: Union[str, Path], fieldnames: Sequence[str]) -> None:
        self.path = _ensure_parent_dir(path)
        self.fieldnames = list(fieldnames)
        self._fh = None
        self._writer = None

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        file_exists = self.path.exists()
        self._fh = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames, extrasaction="ignore")
        if (not file_exists) or (self.path.stat().st_size == 0):
            self._writer.writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        self._ensure_open()
        assert self._writer is not None
        self._writer.writerow(_normalize_row(row, self.fieldnames, fill=""))

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None
            self._writer = None


class JSONLAppender(BaseAppender):
    def __init__(self, path: Union[str, Path], keep_fields: Optional[Sequence[str]] = None) -> None:
        self.path = _ensure_parent_dir(path)
        self.keep_fields = list(keep_fields) if keep_fields else None
        self._fh = None

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")

    def append(self, row: Dict[str, Any]) -> None:
        self._ensure_open()
        r = row if self.keep_fields is None else _normalize_row(row, self.keep_fields, fill=None)
        self._fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None
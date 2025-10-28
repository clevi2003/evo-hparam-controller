from typing import Any, Dict, Optional, Iterable, List, Sequence, Union
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .logging_base import BaseAppender, _ensure_parent_dir, _ensure_dir

# class ParquetAppender(BaseAppender):
#     """
#     Append rows to a Parquet file with a fixed schema
#     - If file exists, reads/locks the schema from disk
#     - If new, builds schema from schema or fieldnames
#     - Buffers rows up to buffer_rows before flushing to disk for throughput
#
#     Column order is preserved from the schema
#     Missing keys become NULLs
#     """
#
#     def __init__(
#         self,
#         path: Union[str, Path],
#         fieldnames: Optional[Sequence[str]] = None,
#         schema: Optional[pa.Schema] = None,
#         buffer_rows: int = 1024
#     ) -> None:
#         self.path = _ensure_parent_dir(path)
#         self.buffer_rows = max(1, int(buffer_rows))
#         self._schema: Optional[pa.Schema] = None
#         self._writer: Optional[pq.ParquetWriter] = None
#         self._fieldnames: List[str] = []
#         self._buffer: List[Dict[str, Any]] = []
#
#         self._init_schema_and_writer(fieldnames, schema)
#
#     def _init_schema_and_writer(
#         self,
#         fieldnames: Optional[Sequence[str]],
#         schema: Optional[pa.Schema]
#     ) -> None:
#         if self.path.exists() and self.path.stat().st_size > 0:
#             # Reuse existing schema from file (strongly recommended in long runs)
#             existing = pq.read_schema(self.path)
#             self._schema = existing
#             self._fieldnames = list(existing.names)
#         else:
#             if schema is None:
#                 if not fieldnames:
#                     raise ValueError("New Parquet files require either a pyarrow Schema or fieldnames.")
#                 # default to string columns if no schema given
#                 self._fieldnames = list(fieldnames)
#                 self._schema = pa.schema([pa.field(n, pa.string()) for n in self._fieldnames])
#             else:
#                 self._schema = schema
#                 self._fieldnames = list(schema.names)
#
#         # Create writer (append mode if file exists, pyarrow handles row groups)
#         self._writer = pq.ParquetWriter(
#             where=self.path,
#             schema=self._schema,
#             use_deprecated_int96_timestamps=True,  # interop
#             compression="snappy",
#             use_dictionary=True,
#         )
#
#     def _flush_buffer(self) -> None:
#         if not self._buffer:
#             return
#         assert self._writer is not None and self._schema is not None
#         cols: List[List[Any]] = [[] for _ in self._fieldnames]
#         for row in self._buffer:
#             for i, name in enumerate(self._fieldnames):
#                 cols[i].append(row.get(name, None))
#         arrays = [pa.array(col, type=self._schema.field(i).type) for i, col in enumerate(cols)]
#         table = pa.Table.from_arrays(arrays, names=self._fieldnames)
#         self._writer.write_table(table)
#         self._buffer.clear()
#
#     @property
#     def fieldnames(self) -> List[str]:
#         return list(self._fieldnames)
#
#     @property
#     def schema(self) -> pa.Schema:
#         assert self._schema is not None
#         return self._schema
#
#     def append(self, row: Dict[str, Any]) -> None:
#         # assume row keys may be a superset; we select by schema order
#         self._buffer.append(row)
#         if len(self._buffer) >= self.buffer_rows:
#             self._flush_buffer()
#
#     def append_many(self, rows: Iterable[Dict[str, Any]]) -> None:
#         for r in rows:
#             self.append(r)
#
#     def flush(self) -> None:
#         self._flush_buffer()
#
#     def close(self) -> None:
#         try:
#             self._flush_buffer()
#             if self._writer is not None:
#                 self._writer.close()
#         finally:
#             self._writer = None
#             self._schema = None
#             self._fieldnames = []

# Global cache so multiple appenders to the same path reuse a single writer
_WRITER_CACHE: dict[str, pq.ParquetWriter] = {}
_WRITER_REFCOUNT: dict[str, int] = {}

class ParquetAppender(BaseAppender):
    def __init__(self, path: Path, schema: pa.Schema, buffer_rows=1024) -> None:
        self._path = Path(path)
        self._schema = schema
        self._key = self._path.as_posix()
        # bump refcount
        _WRITER_REFCOUNT[self._key] = _WRITER_REFCOUNT.get(self._key, 0) + 1

    def _get_or_create_writer(self) -> pq.ParquetWriter:
        w = _WRITER_CACHE.get(self._key)
        if w is None:
            # Create a writer only once; subsequent appenders reuse it.
            w = pq.ParquetWriter(self._key, self._schema)
            _WRITER_CACHE[self._key] = w
        return w

    def append(self, row: Dict[str, Any]) -> None:
        writer = self._get_or_create_writer()
        arrays = []
        for field in self._schema:
            v = row.get(field.name, None)
            arrays.append(pa.array([v], type=field.type))
        tbl = pa.Table.from_arrays(arrays, schema=self._schema)
        writer.write_table(tbl)

    def close(self) -> None:
        # decrement refcount and only close when the last user closes
        key = self._key
        if key in _WRITER_REFCOUNT:
            _WRITER_REFCOUNT[key] -= 1
            if _WRITER_REFCOUNT[key] <= 0:
                w = _WRITER_CACHE.pop(key, None)
                _WRITER_REFCOUNT.pop(key, None)
                if w is not None:
                    w.close()


class PartitionedParquetAppender(BaseAppender):
    """
    Partitioned Parquet writer for very large logs
    Writes multiple parquet files inside a directory, rolling to a new file
    after rows_per_file. Schema is locked from the first file created.

    File layout:
      <dir>/
        part-00001.parquet
        part-00002.parquet
        ...
    Good for long training runs where a single file might become too large.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        fieldnames: Optional[Sequence[str]] = None,
        schema: Optional[pa.Schema] = None,
        rows_per_file: int = 100_000,
        buffer_rows: int = 2048,
    ) -> None:
        self.dir = _ensure_dir(directory)
        self.rows_per_file = max(1, int(rows_per_file))
        self.buffer_rows = max(1, int(buffer_rows))
        self._schema: Optional[pa.Schema] = None
        self._fieldnames: List[str] = []
        self._buffer: List[Dict[str, Any]] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._rows_in_current_file = 0
        self._part_index = self._next_part_index()

        # initialize schema from existing first file or params
        self._init_schema(fieldnames, schema)
        self._open_new_writer()

    @staticmethod
    def idx(p: Path) -> int:
        stem = p.stem  # "part-00012"
        return int(stem.split("-")[-1])

    def _next_part_index(self) -> int:
        existing = sorted(self.dir.glob("part-*.parquet"))
        if not existing:
            return 1
        return max(self.idx(p) for p in existing) + 1

    def _init_schema(self, fieldnames: Optional[Sequence[str]], schema: Optional[pa.Schema]) -> None:
        first_file = self.dir / "part-00001.parquet"
        if first_file.exists() and first_file.stat().st_size > 0:
            existing = pq.read_schema(first_file)
            self._schema = existing
            self._fieldnames = list(existing.names)
        else:
            if schema is None:
                if not fieldnames:
                    raise ValueError("New partitioned parquet logs require a Schema or fieldnames.")
                self._fieldnames = list(fieldnames)
                self._schema = pa.schema([pa.field(n, pa.string()) for n in self._fieldnames])
            else:
                self._schema = schema
                self._fieldnames = list(schema.names)

    def _open_new_writer(self) -> None:
        assert self._schema is not None
        filename = self.dir / f"part-{self._part_index:05d}.parquet"
        self._writer = pq.ParquetWriter(
            where=filename,
            schema=self._schema,
            use_deprecated_int96_timestamps=True,
            compression="snappy",
            use_dictionary=True,
        )
        self._rows_in_current_file = 0
        self._part_index += 1

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        assert self._writer is not None and self._schema is not None
        cols: List[List[Any]] = [[] for _ in self._fieldnames]
        for row in self._buffer:
            for i, name in enumerate(self._fieldnames):
                cols[i].append(row.get(name, None))
        arrays = [pa.array(col, type=self._schema.field(i).type) for i, col in enumerate(cols)]
        table = pa.Table.from_arrays(arrays, names=self._fieldnames)
        self._writer.write_table(table)
        self._rows_in_current_file += len(self._buffer)
        self._buffer.clear()

        if self._rows_in_current_file >= self.rows_per_file:
            self._writer.close()
            self._open_new_writer()

    def append(self, row: Dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_rows:
            self._flush_buffer()

    def append_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            self.append(r)

    def flush(self) -> None:
        self._flush_buffer()

    def close(self) -> None:
        try:
            self._flush_buffer()
            if self._writer is not None:
                self._writer.close()
        finally:
            self._writer = None
            self._schema = None
            self._fieldnames = []
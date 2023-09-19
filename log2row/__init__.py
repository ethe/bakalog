from __future__ import annotations

import glob
import logging
import re
import sys
from typing import Generator

import duckdb
from pypika import Column, Query, Schema, Table

from .util import Log, Memory


class Sink:
    def __init__(self, path: str, max_size: int = 512):
        self.path = path
        self.max_size = max_size
        self.buffer = []

    def __iter__(self):
        files = glob.glob(self.path)
        logging.info(f"detect log files: {files}")

        def f():
            for file in files:
                with open(file, "r") as f:
                    for line in f:
                        yield line[:-1][: self.max_size]

        def b():
            while len(self.buffer) != 0:
                line = self.buffer.pop()
                yield line

        file = f()

        while True:
            buf = b()
            yield from buf
            yield from file
            yield None
            if len(self.buffer) == 0:
                break

    def send(self, lines):
        self.buffer += lines


Memory().serialize(re.Pattern, lambda p: p.pattern)


class Match:
    def __init__(self, memory: Memory, sink: Sink):
        self.sink = sink
        self.patterns = memory.load("patterns", [])
        for offset in range(0, len(self.patterns)):
            self.patterns[offset] = re.compile(self.patterns[offset])

    def __iter__(self):
        for line in self.sink:
            if line is None:
                yield None
                continue

            for regex in self.patterns:
                match = regex.match(line)
                if match is None:
                    continue
                yield Log(regex.pattern, line, match.groups())
                break
            else:
                yield line

    def send(self, pattern: re.Pattern):
        self.patterns.append(pattern)


def collect(
    logs: Generator[Log, None, None],
    max_lines: int,
    db_file: str = ":default:",
) -> duckdb.DuckDBPyConnection:
    if max_lines <= 0:
        max_lines = sys.maxsize

    db = duckdb.connect(db_file)

    for line, _ in zip(logs, range(max_lines)):
        schema = Schema("information_schema")
        table = db.execute(
            Query.from_(schema.tables)
            .select("*")
            .where(schema.tables.table_name == line.pattern)
            .get_sql()
        ).fetchone()
        if table is None:
            columns = [Column(f"c{id}", "string") for id in range(0, len(line.groups))]
            db.sql(
                Query.create_table(line.pattern.replace('"', '""'))
                .columns(*columns)
                .get_sql()
            )

        table = Table(line.pattern.replace('"', '""'))
        db.sql(Query.into(table).insert(*line.groups).get_sql())

    return db

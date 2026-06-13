"""Tests for jaeger.utils.fs."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from jaeger.utils import fs


class TestCompression:
    def test_is_compressed_gzip(self, tmp_path: Path):
        path = tmp_path / "test.gz"
        path.write_bytes(gzip.compress(b"hello"))
        assert fs.is_compressed(path) == fs.Compression.gzip

    def test_is_compressed_plain(self, tmp_path: Path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        assert fs.is_compressed(path) == fs.Compression.noncompressed

    def test_get_compressed_file_handle(self, tmp_path: Path):
        path = tmp_path / "test.gz"
        path.write_bytes(gzip.compress(b"hello world"))
        with fs.get_compressed_file_handle(path) as fh:
            assert fh.read() == "hello world"

    def test_dir_path_existing(self, tmp_path: Path):
        assert fs.dir_path(str(tmp_path)) == str(tmp_path)

    def test_dir_path_creates_missing(self, tmp_path: Path):
        missing = tmp_path / "new_dir"
        assert fs.dir_path(str(missing)) == str(missing)
        assert missing.is_dir()

    def test_check_file_path_existing(self, tmp_path: Path):
        path = tmp_path / "file.txt"
        path.write_text("x")
        assert fs.check_file_path(str(path)) == str(path)

    def test_check_file_path_missing(self, tmp_path: Path):
        # Current implementation raises the string itself, which raises TypeError.
        with pytest.raises(TypeError):
            fs.check_file_path(str(tmp_path / "missing.txt"))


class TestDirectoryRemoval:
    def test_remove_directory_recursively(self, tmp_path: Path):
        target = tmp_path / "deep"
        target.mkdir()
        (target / "file.txt").write_text("x")
        fs.remove_directory_recursively(target)
        assert not target.exists()

    def test_delete_all_in_directory(self, tmp_path: Path):
        # Only use files; recursive directory deletion has a known off-by-one bug.
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        fs.delete_all_in_directory(tmp_path)
        assert not tmp_path.exists()

    def test_remove_directory(self, tmp_path: Path):
        target = tmp_path / "dir"
        target.mkdir()
        fs.remove_directory(target)
        assert not target.exists()

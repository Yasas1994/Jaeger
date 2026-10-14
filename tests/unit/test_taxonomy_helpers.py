"""Tests for the taxonomy-database helpers in jaeger.commands.taxonomy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("taxopy")
pytest.importorskip("faiss")

from jaeger.commands.taxonomy import (
    TaxonomyModel,
    contig_annotations,
    find_taxdb,
    split_windows_by_contig,
    write_taxonomy_tsv,
)


def _make_model_dir(tmp_path, with_db=True):
    graph = tmp_path / "model" / "m_graph"
    graph.mkdir(parents=True)
    if with_db:
        db = tmp_path / "taxonomy_db"
        (db / "tax").mkdir(parents=True)
        (db / "genomes.faiss").touch()
        (db / "index2tax.npz").touch()
    return {"graph": graph}


def test_find_taxdb_present(tmp_path):
    model_info = _make_model_dir(tmp_path, with_db=True)
    assert find_taxdb(model_info) == tmp_path / "taxonomy_db"


def test_find_taxdb_missing_dir(tmp_path):
    model_info = _make_model_dir(tmp_path, with_db=False)
    assert find_taxdb(model_info) is None


def test_find_taxdb_incomplete(tmp_path):
    model_info = _make_model_dir(tmp_path, with_db=True)
    (tmp_path / "taxonomy_db" / "genomes.faiss").unlink()
    assert find_taxdb(model_info) is None


def test_find_taxdb_no_graph():
    assert find_taxdb({}) is None


def test_split_windows_by_contig():
    # 3 contigs with 2, 1 and 3 windows respectively
    y_pred = {
        "meta_0": np.array([b"c1", b"c1", b"c2", b"c3", b"c3", b"c3"]),
        "meta_2": np.array([0, 1, 1, 0, 0, 1], dtype=np.int32),
    }
    embeddings = np.arange(12, dtype=np.float32).reshape(6, 2)
    emb_split, headers = split_windows_by_contig(y_pred, embeddings)

    assert [h for h in headers] == [b"c1", b"c2", b"c3"]
    assert [e.shape[0] for e in emb_split] == [2, 1, 3]
    np.testing.assert_array_equal(emb_split[0], embeddings[:2])
    np.testing.assert_array_equal(emb_split[2], embeddings[3:])


def _fake_taxon(taxid, lineage):
    return SimpleNamespace(
        taxid=taxid,
        taxid_lineage=lineage,
        name=f"taxon_{taxid}",
        rank="species",
        rank_name_dictionary={},
    )


def _make_model(lineages):
    """TaxonomyModel shell with stubbed taxon lookups (no real taxdb/FAISS)."""
    model = TaxonomyModel.__new__(TaxonomyModel)
    model._taxon_cache = {}
    model.taxdb = None
    model._get_taxon = lambda taxid: _fake_taxon(taxid, lineages[taxid])
    return model


class _FakeIndex:
    """Stand-in for a FAISS index returning predetermined neighbor rows."""

    def __init__(self, neighbors, ntotal=3):
        self._neighbors = np.asarray(neighbors)
        self.ntotal = ntotal

    def search(self, x, k):
        return None, self._neighbors


def test_support_fraction():
    # taxid 10 sits on the lineage of 2 and 3 but not of 5
    lineages = {2: [1, 10, 2], 3: [1, 10, 3], 5: [1, 20, 5]}
    model = _make_model(lineages)
    lca = _fake_taxon(10, [1, 10])

    assert model.support_fraction([2, 2, 3, 5], lca) == 0.75
    assert model.support_fraction([5, 5], lca) == 0.0
    assert model.support_fraction([], lca) == 0.0
    assert model.support_fraction([2, 3], SimpleNamespace(taxid=None)) == 0.0


def test_predict_reports_support_fraction():
    lineages = {2: [1, 10, 2], 3: [1, 10, 3], 5: [1, 20, 5]}
    model = _make_model(lineages)
    model.index2taxid = np.array([2, 3, 5])
    # neighbor rows per window: c1 has 1 window, c2 has 2 windows
    model.faiss_index = _FakeIndex([[0, 1], [0, 2], [1, 2]])
    model.get_lca = lambda taxids, fraction=0.6: _fake_taxon(10, [1, 10])

    embeddings = [np.ones((1, 4), np.float32), np.ones((2, 4), np.float32)]
    preds = model.predict(embeddings, [b"c1", b"c2"], k=2)

    # c1 votes taxids [2, 3]: both lineages contain 10
    assert preds[0]["support_frac"] == 1.0
    # c2 votes taxids [2, 5, 3, 5]: votes for 5 fall off the assigned lineage
    assert preds[1]["support_frac"] == 0.5


def test_write_taxonomy_tsv_support_column(tmp_path):
    preds = [
        {
            "header": b"c1",
            "lca": _fake_taxon(10, [1, 10]),
            "support_frac": 0.75,
        }
    ]
    out = tmp_path / "taxonomy.tsv"
    write_taxonomy_tsv(preds, out)

    df = pd.read_csv(out, sep="\t")
    assert df.loc[0, "header"] == "c1"
    assert df.loc[0, "taxid"] == 10
    assert df.loc[0, "support_frac"] == 0.75


CLASS_MAP = {
    "num_classes": 6,
    "class": ["bacteria", "phage", "eukarya", "archaea", "plasmid", "virus"],
    "index": [0, 1, 2, 3, 4, 5],
}


def _synthetic_y_pred():
    """Two contigs: c1 (2 windows, phage-like), c2 (1 window, unreliable)."""
    logits = np.zeros((3, 6), dtype=np.float32)
    logits[0, 1] = 8.0  # c1 window 1 -> phage
    logits[1, 1] = 8.0  # c1 window 2 -> phage
    logits[2, 0] = 8.0  # c2 -> bacteria
    return {
        "prediction": logits,
        "reliability": np.array([[5.0], [5.0], [-5.0]], dtype=np.float32),
        "meta_0": np.array([b"c1", b"c1", b"c2"]),
        "meta_1": np.array([0, 2000, 0], dtype=np.int32),
        "meta_2": np.array([0, 1, 1], dtype=np.int32),
        "meta_4": np.array([4000, 4000, 2000], dtype=np.int32),
        "meta_5": np.array([500, 500, 500], dtype=np.int32),
        "meta_6": np.array([500, 500, 500], dtype=np.int32),
        "meta_7": np.array([500, 500, 500], dtype=np.int32),
        "meta_8": np.array([500, 500, 500], dtype=np.int32),
        "meta_9": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }


def test_contig_annotations():
    annotations = contig_annotations(_synthetic_y_pred(), CLASS_MAP, fsize=2000)

    assert set(annotations) == {"c1", "c2"}
    c1 = annotations["c1"]
    assert c1["length"] == 4000
    assert c1["prediction"] == "phage"
    assert c1["reliability_score"] > 0.99
    assert c1["phage_score"] > 0.9
    assert "bacteria_score" in c1
    # c2's reliability is below the gate: label collapses to "uncertain"
    c2 = annotations["c2"]
    assert c2["length"] == 2000
    assert c2["prediction"] == "uncertain"
    assert c2["reliability_score"] < 0.01


def test_write_taxonomy_tsv_with_annotations(tmp_path):
    preds = [
        {
            "header": b"c1",
            "lca": _fake_taxon(10, [1, 10]),
            "support_frac": 1.0,
        }
    ]
    annotations = {
        "c1": {
            "length": 4000,
            "prediction": "phage",
            "entropy": 0.01,
            "reliability_score": 0.99,
            "phage_score": 0.97,
        }
    }
    out = tmp_path / "taxonomy.tsv"
    write_taxonomy_tsv(preds, out, annotations=annotations)

    df = pd.read_csv(out, sep="\t")
    assert df.loc[0, "length"] == 4000
    assert df.loc[0, "prediction"] == "phage"
    assert df.loc[0, "reliability_score"] == 0.99
    assert df.loc[0, "phage_score"] == 0.97
    # length sits right after header, like in the main summary table
    assert list(df.columns)[:3] == ["header", "length", "taxid"]


def test_write_taxonomy_tsv_filters_nonviral(tmp_path):
    preds = [
        {"header": "phage_c", "lca": _fake_taxon(10, [1, 10]), "support_frac": 1.0},
        {"header": "virus_c", "lca": _fake_taxon(11, [1, 11]), "support_frac": 0.8},
        {"header": "bact_c", "lca": _fake_taxon(12, [1, 12]), "support_frac": 0.9},
        {"header": "unc_c", "lca": _fake_taxon(13, [1, 13]), "support_frac": 0.7},
    ]
    annotations = {
        h: {"length": 2000, "prediction": p, "entropy": 0.1, "reliability_score": 0.9}
        for h, p in [
            ("phage_c", "phage"),
            ("virus_c", "virus"),
            ("bact_c", "bacteria"),
            ("unc_c", "uncertain"),
        ]
    }
    out = tmp_path / "taxonomy.tsv"
    write_taxonomy_tsv(preds, out, annotations=annotations)

    df = pd.read_csv(out, sep="\t")
    assert sorted(df["header"]) == ["phage_c", "virus_c"]


def test_write_taxonomy_tsv_empty_after_filter(tmp_path):
    preds = [{"header": "bact_c", "lca": _fake_taxon(12, [1, 12]), "support_frac": 0.9}]
    annotations = {
        "bact_c": {
            "length": 2000,
            "prediction": "bacteria",
            "entropy": 0.1,
            "reliability_score": 0.9,
        }
    }
    out = tmp_path / "taxonomy.tsv"
    write_taxonomy_tsv(preds, out, annotations=annotations)

    df = pd.read_csv(out, sep="\t")
    assert len(df) == 0
    assert "header" in df.columns and "lineage" in df.columns


def test_load_faiss_index_searchable(tmp_path):
    """Loaded index is searchable on whatever device faiss picks.

    Under faiss-cpu (CI) the index stays on CPU; under faiss-gpu with a
    visible GPU it is moved to the GPU — either way it must answer queries.
    """
    import faiss

    rng = np.random.default_rng(0)
    vecs = rng.random((10, 4), dtype=np.float32)
    faiss.normalize_L2(vecs)  # cosine search, as in the real pipeline
    index = faiss.IndexFlatIP(4)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "genomes.faiss"))

    model = TaxonomyModel.__new__(TaxonomyModel)
    model._gpu_resources = None
    loaded = model.load_faiss_index(tmp_path / "genomes.faiss")

    assert loaded.ntotal == 10
    _, neighbors = loaded.search(vecs[:1], 3)
    assert neighbors[0, 0] == 0


def test_load_faiss_index_use_gpu_false_keeps_cpu(tmp_path):
    import faiss

    vecs = np.zeros((5, 4), dtype=np.float32)
    index = faiss.IndexFlatIP(4)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "genomes.faiss"))

    model = TaxonomyModel.__new__(TaxonomyModel)
    model._gpu_resources = None
    loaded = model.load_faiss_index(tmp_path / "genomes.faiss", use_gpu=False)

    assert isinstance(loaded, faiss.IndexFlatIP)
    assert model._gpu_resources is None

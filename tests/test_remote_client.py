"""Tests for uq.remote — SMS-API client and cd1 TSV parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest


# ── cd1 TSV parsing tests (no network needed) ────────────────────────


@pytest.fixture()
def mock_cd1_archive(tmp_path: Path) -> Path:
    """Create a mock extracted archive with cd1 TSV files."""
    exp_dir = tmp_path / "sim-test-exp"
    analyses = exp_dir / "analyses" / "variant=0" / "plots"

    # higher_order_properties
    hop_dir = analyses / "analysis=cd1_higher_order_properties"
    hop_dir.mkdir(parents=True)
    (hop_dir / "higher_order_properties.tsv").write_text(
        "Properties\tmean\tstd\n"
        "Cell mass (mg/10^9 cells)\t680.5\t12.3\n"
        "Cell volume (um^3)\t1.21\t0.05\n"
        "Doubling time (min)\t44.2\t3.1\n"
    )

    # transcriptomics (small subset)
    tx_dir = analyses / "analysis=cd1_transcriptomics"
    tx_dir.mkdir(parents=True)
    (tx_dir / "transcriptomics.tsv").write_text(
        "EcoCyc Gene ID\tmean\tstd\n"
        "EG10001\t45.2\t8.1\n"
        "EG10002\t120.8\t22.4\n"
        "EG10003\t0.5\t0.2\n"
    )

    # proteomics
    pr_dir = analyses / "analysis=cd1_proteomics"
    pr_dir.mkdir(parents=True)
    (pr_dir / "proteomics.tsv").write_text(
        "EcoCyc Monomer ID\tmean\tstd\n"
        "1-PFK-MONOMER\t312.4\t45.6\n"
        "ACEA-MONOMER\t89.7\t12.3\n"
    )

    # fluxomics
    fx_dir = analyses / "analysis=cd1_fluxomics"
    fx_dir.mkdir(parents=True)
    (fx_dir / "cd1_fluxomics_detailed.tsv").write_text(
        "EcoCyc Reaction ID\tmean\tstd\n"
        "1-ACYLGLYCEROL-3-P-ACYLTRANSFER-RXN\t0.0012\t0.0003\n"
        "PGLUCISOM-RXN\t4.567\t0.89\n"
    )

    # metabolomics
    mx_dir = analyses / "analysis=cd1_metabolomics"
    mx_dir.mkdir(parents=True)
    (mx_dir / "metabolomics.tsv").write_text(
        "EcoCyc Compound ID\tmean\tstd\n"
        "2-KETOGLUTARATE\t0.45\t0.12\n"
    )

    # metadata.json (as the real archive has)
    import json
    for d in [hop_dir, tx_dir, pr_dir, fx_dir, mx_dir]:
        (d / "metadata.json").write_text(json.dumps({
            "analysis_type": "multiseed",
            "analysis_name": d.name.replace("analysis=", ""),
        }))

    return exp_dir


class TestParseCd1Tsv:
    def test_parse_higher_order(self, mock_cd1_archive: Path) -> None:
        from uq.remote import parse_cd1_tsv

        path = mock_cd1_archive / "analyses/variant=0/plots/analysis=cd1_higher_order_properties/higher_order_properties.tsv"
        ids, means, stds = parse_cd1_tsv(path)

        assert len(ids) == 3
        assert ids[0] == "Cell mass (mg/10^9 cells)"
        assert means[0] == pytest.approx(680.5)
        assert stds[0] == pytest.approx(12.3)

    def test_parse_transcriptomics(self, mock_cd1_archive: Path) -> None:
        from uq.remote import parse_cd1_tsv

        path = mock_cd1_archive / "analyses/variant=0/plots/analysis=cd1_transcriptomics/transcriptomics.tsv"
        ids, means, stds = parse_cd1_tsv(path)

        assert len(ids) == 3
        assert ids[0] == "EG10001"
        assert means[1] == pytest.approx(120.8)

    def test_parse_empty_tsv(self, tmp_path: Path) -> None:
        from uq.remote import parse_cd1_tsv

        tsv = tmp_path / "empty.tsv"
        tsv.write_text("EcoCyc Gene ID\tmean\tstd\n")  # header only
        ids, means, stds = parse_cd1_tsv(tsv)

        assert len(ids) == 0
        assert means.shape == (0,)


class TestFindCd1Tsvs:
    def test_find_all_modules(self, mock_cd1_archive: Path) -> None:
        from uq.remote import find_cd1_tsvs

        found = find_cd1_tsvs(mock_cd1_archive, variant=0)

        assert "higher_order" in found
        assert "transcriptome" in found
        assert "proteome" in found
        assert "fluxome" in found
        assert "exchange_fluxes" in found
        assert len(found) == 5

    def test_find_variant_filter(self, mock_cd1_archive: Path) -> None:
        from uq.remote import find_cd1_tsvs

        # variant=1 doesn't exist
        found = find_cd1_tsvs(mock_cd1_archive, variant=1)
        # Should still find files (fallback to any match)
        assert len(found) > 0


class TestParseCd1TsvsComposite:
    def test_all_presets(self, mock_cd1_archive: Path) -> None:
        from uq.remote import parse_cd1_tsvs

        Y, names = parse_cd1_tsvs(mock_cd1_archive)

        # 3 + 3 + 2 + 2 + 1 = 11 total observables
        assert Y.shape == (11,)
        assert len(names) == 11
        # Names are prefixed with module name
        assert names[0].startswith("cd1_higher_order_properties:")
        assert "cd1_transcriptomics:EG10001" in names

    def test_subset_presets(self, mock_cd1_archive: Path) -> None:
        from uq.remote import parse_cd1_tsvs

        Y, names = parse_cd1_tsvs(mock_cd1_archive, presets=["higher_order"])

        assert Y.shape == (3,)
        assert all("cd1_higher_order_properties:" in n for n in names)

    def test_multi_variant(self, mock_cd1_archive: Path) -> None:
        from uq.remote import parse_cd1_tsvs_multi_variant

        # Only variant=0 exists in fixture, so n_variants=1
        Y, names = parse_cd1_tsvs_multi_variant(
            mock_cd1_archive, n_variants=1, presets=["higher_order"]
        )

        assert Y.shape == (1, 3)
        assert len(names) == 3


class TestCd1ModuleMap:
    def test_all_presets_mapped(self) -> None:
        from uq.remote import CD1_MODULE_MAP

        expected = {"higher_order", "transcriptome", "proteome", "fluxome", "exchange_fluxes"}
        assert set(CD1_MODULE_MAP.keys()) == expected

    def test_module_names(self) -> None:
        from uq.remote import CD1_MODULE_MAP

        assert CD1_MODULE_MAP["transcriptome"]["module"] == "cd1_transcriptomics"
        assert CD1_MODULE_MAP["proteome"]["module"] == "cd1_proteomics"
        assert CD1_MODULE_MAP["fluxome"]["module"] == "cd1_fluxomics"
        assert CD1_MODULE_MAP["exchange_fluxes"]["module"] == "cd1_metabolomics"
        assert CD1_MODULE_MAP["higher_order"]["module"] == "cd1_higher_order_properties"


class TestSmsApiClientInit:
    def test_default_base_url(self) -> None:
        from uq.remote import SmsApiClient

        client = SmsApiClient()
        assert client.base_url == "http://localhost:8080"
        client.close()

    def test_custom_base_url(self) -> None:
        from uq.remote import SmsApiClient

        client = SmsApiClient(base_url="https://sms.cam.uchc.edu")
        assert client.base_url == "https://sms.cam.uchc.edu"
        client.close()

    def test_context_manager(self) -> None:
        from uq.remote import SmsApiClient

        with SmsApiClient() as client:
            assert client.base_url == "http://localhost:8080"


class TestSmsApiError:
    def test_error_format(self) -> None:
        from uq.remote import SmsApiError

        err = SmsApiError(504, "Gateway Timeout")
        assert "504" in str(err)
        assert "Gateway Timeout" in str(err)
        assert err.status_code == 504
        assert err.detail == "Gateway Timeout"

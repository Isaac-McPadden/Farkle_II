from __future__ import annotations

from pathlib import Path

from scripts.check_terminology import find_violations

AMBIGUOUS = "\x70\x6f\x6f\x6c"


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_repository_terminology_is_precise() -> None:
    assert find_violations() == []


def test_forbidden_product_term_in_production_path_fails(tmp_path: Path) -> None:
    _write(tmp_path, "src/farkle/product.py", f'ARTIFACT = "ratings_{AMBIGUOUS}ed"\n')

    violations = find_violations(tmp_path)

    assert len(violations) == 1
    assert violations[0].startswith(f"{Path('src/farkle/product.py')}:1:")


def test_historical_evidence_is_outside_enforcement_scope(tmp_path: Path) -> None:
    _write(tmp_path, "docs/reviews/prior_review.md", f"historically {AMBIGUOUS}ed\n")
    _write(tmp_path, "docs/remediation/task_evidence.md", f"formerly {AMBIGUOUS}ing\n")

    assert find_violations(tmp_path) == []


def test_legitimate_external_api_symbols_do_not_exempt_other_text(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/farkle/workers.py",
        "import multiprocessing\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "executor = ProcessPoolExecutor()\n"
        "workers = multiprocessing.Pool()\n"
        f"workers = multiprocessing.Pool(); description = '{AMBIGUOUS} results'\n",
    )

    violations = find_violations(tmp_path)

    assert len(violations) == 1
    assert "description" in violations[0]


def test_case_token_boundary_and_path_rules_are_deterministic(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/config_reference.md",
        f"{AMBIGUOUS.upper()} {AMBIGUOUS}s swimming{AMBIGUOUS} {AMBIGUOUS}side\n",
    )
    _write(tmp_path, "docs/reviews/ignored.md", f"{AMBIGUOUS}\n")
    _write(tmp_path, "src/z_last.py", f"value = 'x_{AMBIGUOUS}ed'\n")
    _write(tmp_path, "src/a_first.py", f"value = '{AMBIGUOUS}ing'\n")

    first = find_violations(tmp_path)
    second = find_violations(tmp_path)

    assert first == second
    assert [item.split(":", maxsplit=1)[0] for item in first] == [
        str(Path("docs/config_reference.md")),
        str(Path("src/a_first.py")),
        str(Path("src/z_last.py")),
    ]

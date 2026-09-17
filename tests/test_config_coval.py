from pathlib import Path

import pytest
import yaml

from polyloop.config import load_loop

RECIPE = Path(__file__).resolve().parent.parent / "recipes" / "voice-coval" / "loop.yaml"


def _write(tmp_path, **over):
    raw = yaml.safe_load(RECIPE.read_text())
    for k, v in over.items():
        cur = raw
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = v
    p = tmp_path / "loop.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_recipe_loads():
    cfg = load_loop(RECIPE)
    assert cfg.coval is not None
    assert cfg.model == "nvidia/Nemotron-Flash-3B-Instruct"
    assert cfg.renderer == "role_colon"
    assert cfg.tasks.startswith("coval://") and cfg.gate.holdout.startswith("coval://")
    assert cfg.stages[0].kind == "opsd" and cfg.stages[0].harness == "coval"
    assert cfg.coval.temperature == cfg.gate.temperature
    assert cfg.system_prompt_path == RECIPE.parent / "system.md"
    assert cfg.system_prompt_path.exists()


def test_other_recipes_have_no_coval():
    for name in ("pydantic-v2/loop-4b.yaml", "pydantic-v2/loop.yaml", "swe-mini/loop.yaml"):
        cfg = load_loop(RECIPE.parent.parent / name)
        assert cfg.coval is None
        assert cfg.system_prompt_path is None


def test_holdout_must_differ_from_pool(tmp_path):
    p = _write(tmp_path, **{"gate.holdout": "coval://REPLACE_POOL_TEST_SET"})
    with pytest.raises(ValueError, match="same Coval test set"):
        load_loop(p)


def test_coval_requires_coval_datasets(tmp_path):
    p = _write(tmp_path, tasks="~/some/harbor/dir")
    with pytest.raises(ValueError, match="coval://"):
        load_loop(p)


def test_rl_stage_rejected_for_coval(tmp_path):
    raw = yaml.safe_load(RECIPE.read_text())
    raw["stages"].insert(0, {"kind": "rl", "harness": "cookbook-bash"})
    p = tmp_path / "loop.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="rl"):
        load_loop(p)


def test_unknown_coval_key_rejected(tmp_path):
    p = _write(tmp_path, **{"coval.bogus": 1})
    with pytest.raises(ValueError, match="unknown keys"):
        load_loop(p)

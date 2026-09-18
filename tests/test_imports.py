import ast
from pathlib import Path


def test_every_name_used_in_stages_is_imported():
    """The train stage launches workers with subprocess; a refactor once dropped the import and only
    a GPU run noticed. Check module-level names statically."""
    src = (Path(__file__).resolve().parents[1] / "polyloop" / "stages.py").read_text()
    tree = ast.parse(src)
    imported = {n.asname or n.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for n in node.names}
    for mod in ("subprocess", "shutil", "json", "time", "asyncio"):
        assert mod in imported, f"stages.py uses {mod} but does not import it"

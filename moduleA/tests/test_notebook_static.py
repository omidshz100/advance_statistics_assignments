"""Static checks for the seven Module A notebooks. No gym environment is required."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestModuleANotebooks(unittest.TestCase):
    def test_all_notebooks_parse_as_python(self):
        notebooks = sorted(ROOT.rglob("*.ipynb"))
        self.assertEqual(len(notebooks), 7, "Expected the seven original Module A notebooks")
        for file in notebooks:
            with self.subTest(path=str(file.relative_to(ROOT))):
                nb = json.loads(file.read_text(encoding="utf-8"))
                self.assertEqual(nb["nbformat"], 4)
                for index, cell in enumerate(nb["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = cell["source"]
                    code = "".join(source) if isinstance(source, list) else source
                    self.assertNotIn("from gym.wrappers.monitoring import video_recorder", code)
                    # Jupyter shell/magic commands are not Python syntax.
                    lines = [line for line in code.splitlines()
                             if not line.lstrip().startswith(("!", "%"))]
                    try:
                        ast.parse("\n".join(lines), filename=f"{file.name}:cell-{index}")
                    except SyntaxError as exc:
                        self.fail(f"{file.name} cell {index}: {exc}")


if __name__ == "__main__":
    unittest.main()

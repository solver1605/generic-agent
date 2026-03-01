import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.emergent_planner.tools import create_pptx_deck, write_excel_file, write_file


class _FakeWorksheet:
    def __init__(self, title: str):
        self.title = title
        self._cells = {}

    @property
    def max_row(self):
        if not self._cells:
            return 0
        return max(r for (r, _c) in self._cells.keys())

    def cell(self, row: int, column: int, value=None):
        self._cells[(row, column)] = value
        return SimpleNamespace(value=value)

    def delete_rows(self, _start: int, _amount: int):
        self._cells = {}


class _FakeWorkbook:
    def __init__(self):
        self.worksheets = [_FakeWorksheet("Sheet")]
        self.active = self.worksheets[0]

    @property
    def sheetnames(self):
        return [w.title for w in self.worksheets]

    def remove(self, ws):
        self.worksheets = [w for w in self.worksheets if w is not ws]
        self.active = self.worksheets[0] if self.worksheets else None

    def create_sheet(self, title: str):
        ws = _FakeWorksheet(title)
        self.worksheets.append(ws)
        if self.active is None:
            self.active = ws
        return ws

    def __getitem__(self, name: str):
        for w in self.worksheets:
            if w.title == name:
                return w
        raise KeyError(name)

    def save(self, path: str):
        Path(path).write_bytes(b"fake-xlsx")


class _FakeTextFrame:
    def __init__(self):
        self.text = ""
        self._paragraphs = []

    def add_paragraph(self):
        p = SimpleNamespace(text="")
        self._paragraphs.append(p)
        return p


class _FakePlaceholder:
    def __init__(self):
        self.text = ""
        self.text_frame = _FakeTextFrame()


class _FakeShapes:
    def __init__(self, title_placeholder):
        self.title = title_placeholder


class _FakeSlide:
    def __init__(self):
        t = _FakePlaceholder()
        b = _FakePlaceholder()
        self.placeholders = [t, b]
        self.shapes = _FakeShapes(t)
        self.notes_slide = SimpleNamespace(notes_text_frame=SimpleNamespace(text=""))


class _FakeSlides(list):
    def add_slide(self, _layout):
        s = _FakeSlide()
        self.append(s)
        return s


class _FakePresentation:
    def __init__(self, _path=None):
        self.slide_layouts = [0, 1, 2, 3, 4]
        self.slides = _FakeSlides()

    def save(self, path: str):
        Path(path).write_bytes(b"fake-pptx")


class TestOfficeTools(TestCase):
    def test_write_excel_file_missing_dependency(self):
        with patch(
            "src.emergent_planner.tools.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'openpyxl'"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                write_excel_file.invoke(
                    {"path": "tmp.xlsx", "sheets": [{"name": "S1", "rows": [["a", "b"]]}]}
                )
        self.assertIn("openpyxl", str(ctx.exception))

    def test_create_pptx_deck_missing_dependency(self):
        with patch(
            "src.emergent_planner.tools.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'pptx'"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                create_pptx_deck.invoke(
                    {"path": "tmp.pptx", "slides": [{"title": "T1", "bullets": ["a"]}]}
                )
        self.assertIn("python-pptx", str(ctx.exception))

    def test_write_excel_file_success_with_fake_openpyxl(self):
        fake_openpyxl = SimpleNamespace(
            Workbook=_FakeWorkbook,
            load_workbook=lambda _path: _FakeWorkbook(),
        )

        def _import_side_effect(name: str):
            if name == "openpyxl":
                return fake_openpyxl
            raise ModuleNotFoundError(name)

        with tempfile.TemporaryDirectory() as td:
            prev_cwd = Path.cwd()
            os.chdir(td)
            try:
                out_path = Path(td) / "artifacts" / "excel" / "report.xlsx"
                with patch("src.emergent_planner.tools.importlib.import_module", side_effect=_import_side_effect):
                    out = write_excel_file.invoke(
                        {
                            "path": "report.xlsx",
                            "mode": "overwrite",
                            "sheets": [
                                {"name": "Summary", "rows": [["Metric", "Value"], ["Rows", 42]]},
                                {"name": "Data", "rows": [[1, 2, 3], [4, 5, 6]]},
                            ],
                        }
                    )
            finally:
                os.chdir(prev_cwd)

            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["rows_written"], 4)
            self.assertEqual(out["sheets_written"], ["Summary", "Data"])
            self.assertEqual(out["path"], "artifacts/excel/report.xlsx")
            self.assertTrue(out_path.exists())

    def test_create_pptx_deck_success_with_fake_pptx(self):
        fake_pptx = SimpleNamespace(Presentation=_FakePresentation)

        def _import_side_effect(name: str):
            if name == "pptx":
                return fake_pptx
            raise ModuleNotFoundError(name)

        with tempfile.TemporaryDirectory() as td:
            prev_cwd = Path.cwd()
            os.chdir(td)
            try:
                out_path = Path(td) / "artifacts" / "ppt" / "deck.pptx"
                with patch("src.emergent_planner.tools.importlib.import_module", side_effect=_import_side_effect):
                    out = create_pptx_deck.invoke(
                        {
                            "path": "deck.pptx",
                            "title": "Quarterly Review",
                            "subtitle": "Q1",
                            "slides": [
                                {"title": "Status", "bullets": ["On track", "2 blockers"]},
                                {"title": "Next Steps", "body": "Finalize rollout plan", "notes": "Owner: PM"},
                            ],
                        }
                    )
            finally:
                os.chdir(prev_cwd)

            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["slides_added"], 3)
            self.assertEqual(out["total_slides"], 3)
            self.assertEqual(out["path"], "artifacts/ppt/deck.pptx")
            self.assertTrue(out_path.exists())

    def test_write_file_routes_report_like_outputs_into_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            prev_cwd = Path.cwd()
            os.chdir(td)
            try:
                msg = write_file.invoke(
                    {
                        "path": "long_report.md",
                        "content": "# Findings\n- A\n- B\n",
                        "mode": "overwrite",
                    }
                )
            finally:
                os.chdir(prev_cwd)

            expected = Path(td) / "artifacts" / "reports" / "long_report.md"
            self.assertTrue(expected.exists())
            self.assertIn("artifacts/reports/long_report.md", msg)

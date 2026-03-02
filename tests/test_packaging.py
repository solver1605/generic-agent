from __future__ import annotations

import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.src_path = (self.repo_root / "src").as_posix()

    def _run(self, code: str, *, warn_default: bool = False) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = self.src_path
        args = [sys.executable]
        if warn_default:
            args.append("-Wdefault")
        args.extend(["-c", code])
        return subprocess.run(args, cwd=self.repo_root, env=env, text=True, capture_output=True)

    def test_pyproject_scripts_and_src_layout(self):
        pyproject_path = self.repo_root / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        scripts = data.get("project", {}).get("scripts", {})
        self.assertEqual(scripts.get("generic-agent"), "emergent_planner.cli:main")
        self.assertEqual(scripts.get("generic-agent-ui"), "emergent_planner.ui_launcher:main")

        pkg_dir = data.get("tool", {}).get("setuptools", {}).get("package-dir", {})
        self.assertEqual(pkg_dir.get(""), "src")

        includes = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("include", [])
        self.assertIn("emergent_planner*", includes)
        opt = data.get("project", {}).get("optional-dependencies", {})
        self.assertIn("adk", opt)

    def test_public_namespace_import(self):
        proc = self._run(
            "import emergent_planner as ep; "
            "assert hasattr(ep, 'build_app'); "
            "assert hasattr(ep, 'build_runtime_app'); "
            "print('ok')"
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("ok", proc.stdout)

    def test_legacy_namespace_emits_deprecation_warning(self):
        proc = self._run(
            "import warnings; warnings.simplefilter('default', DeprecationWarning); import src.emergent_planner; print('ok')",
            warn_default=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("ok", proc.stdout)
        self.assertIn("DeprecationWarning", proc.stderr)
        self.assertIn("src.emergent_planner", proc.stderr)

    def test_module_entrypoints_help(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = self.src_path

        cli_proc = subprocess.run(
            [sys.executable, "-m", "emergent_planner.cli", "--help"],
            cwd=self.repo_root,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(cli_proc.returncode, 0, msg=cli_proc.stderr)
        self.assertIn("Emergent Planner CLI", cli_proc.stdout)

        ui_proc = subprocess.run(
            [sys.executable, "-m", "emergent_planner.ui_launcher", "--help"],
            cwd=self.repo_root,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(ui_proc.returncode, 0, msg=ui_proc.stderr)
        self.assertIn("Launch GenericAgent Streamlit UI", ui_proc.stdout)


if __name__ == "__main__":
    unittest.main()

## 🍢 font library proliferation — many sources, runtime install
"""
Fonts used to reach ToFU through exactly one directory: pantry() returned
the FIRST candidate that existed and FontRegistry discovered that one path.
Nothing downstream cared where a font came from -- _fonts is flat and
path-keyed, load_font/discover are additive -- so growing the library was
always a feeding problem rather than a restructuring one.

The tests that matter here are the ones protecting what did NOT change: the
eval harnesses still resolve a single root, and the render path still never
needs an OS-level font installation.
"""

import io
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from conftest import LATIN_FONT
from tofu.layers.fonts import (
    FONT_EXTS, FontCoverage, FontRegistry, faces_of, pantries, pantry,
)


class TestPantries:
    def _roots(self, monkeypatch, *, env=None, bundled=None, system=None, installed=None):
        """Point every root somewhere controlled. Missing ones are set to a
        path that does not exist, so a stray real directory on the machine
        cannot leak into an assertion."""
        monkeypatch.setenv("TOFU_FONT_DIR", str(env) if env else "")
        monkeypatch.setattr("tofu.layers.fonts._BUNDLED_ROOT", Path(bundled or "/nope-b"))
        monkeypatch.setattr("tofu.layers.fonts._INSTALLED_ROOT", Path(installed or "/nope-i"))
        monkeypatch.setattr(
            "tofu.layers.fonts._system_font_dir", lambda: str(system or "/nope-s")
        )

    def test_the_installed_root_never_becomes_pantry(self, monkeypatch, tmp_path):
        """THE invariant. The eval harnesses call pantry() to name the one
        library they measure against. If a runtime-installed root could win
        here, the first font a user uploaded would repoint every harness
        from the system library to a near-empty directory -- measured
        during development at 409 faces down to 4."""
        system = tmp_path / "system"
        installed = tmp_path / "installed"
        system.mkdir()
        installed.mkdir()
        self._roots(monkeypatch, system=system, installed=installed)
        assert pantry() == str(system)
        assert str(installed) in pantries()      # discovered, but never THE pantry

    def test_pantry_is_none_when_only_the_installed_root_exists(self, monkeypatch, tmp_path):
        installed = tmp_path / "installed"
        installed.mkdir()
        self._roots(monkeypatch, installed=installed)
        assert pantry() is None
        assert pantries() == [str(installed)]

    def test_nested_roots_are_not_discovered_twice(self, monkeypatch, tmp_path):
        """discover() walks with rglob, so a root contained by an earlier
        one would load every face a second time."""
        outer = tmp_path / "outer"
        (outer / "inner").mkdir(parents=True)
        self._roots(monkeypatch, env=outer, bundled=outer / "inner", system=outer)
        assert pantries() == [str(outer)]

    def test_absent_directories_are_skipped(self, monkeypatch, tmp_path):
        self._roots(monkeypatch)
        assert pantries() == []
        assert pantry() is None

    def test_precedence_is_env_then_bundled_then_system(self, monkeypatch, tmp_path):
        env, bundled, system = (tmp_path / n for n in ("env", "bundled", "system"))
        for d in (env, bundled, system):
            d.mkdir()
        self._roots(monkeypatch, env=env, bundled=bundled, system=system)
        assert pantries() == [str(env), str(bundled), str(system)]
        assert pantry() == str(env)


class TestRegistryConstruction:
    def _dir_with_font(self, tmp_path: Path, name: str) -> Path:
        if LATIN_FONT is None:
            pytest.skip("no Latin face on this machine")
        directory = tmp_path / name
        directory.mkdir()
        (directory / LATIN_FONT.name).write_bytes(LATIN_FONT.read_bytes())
        return directory

    def test_accepts_a_single_path_a_list_or_nothing(self, tmp_path):
        one = self._dir_with_font(tmp_path, "one")
        assert len(FontRegistry(str(one)).faces) >= 1
        assert len(FontRegistry([str(one)]).faces) >= 1
        assert FontRegistry(None).faces == {}
        assert FontRegistry([]).faces == {}

    def test_several_roots_merge_rather_than_replace(self, tmp_path):
        one = self._dir_with_font(tmp_path, "one")
        two = self._dir_with_font(tmp_path, "two")
        merged = FontRegistry([str(one), str(two)])
        single = FontRegistry(str(one))
        assert len(merged.faces) == 2 * len(single.faces)

    def test_discover_stays_additive_after_construction(self, tmp_path):
        one = self._dir_with_font(tmp_path, "one")
        two = self._dir_with_font(tmp_path, "two")
        registry = FontRegistry(str(one))
        before = len(registry.faces)
        registry.discover(str(two))
        assert len(registry.faces) == 2 * before


class TestFacesAccessor:
    def test_faces_exposes_the_same_mapping(self):
        registry = FontRegistry()
        registry._fonts = {"a.ttf": FontCoverage("a.ttf", "A")}
        assert registry.faces is registry._fonts

    def test_faces_of_tolerates_a_missing_registry(self):
        """Every consumer accepts font_registry=None -- that tolerance is
        why the call sites reached for getattr in the first place."""
        assert faces_of(None) == {}
        assert faces_of(object()) == {}

    def test_faces_of_reads_a_real_registry(self):
        registry = FontRegistry()
        registry._fonts = {"a.ttf": FontCoverage("a.ttf", "A")}
        assert list(faces_of(registry)) == ["a.ttf"]


class TestUnloadFont:
    def test_removes_a_single_face(self):
        registry = FontRegistry()
        registry._fonts = {"a.ttf": FontCoverage("a.ttf", "A")}
        assert registry.unload_font("a.ttf") == 1
        assert registry.faces == {}

    def test_removes_every_face_of_a_collection(self):
        """Collections expand to path#0, path#1 …; removing the file has to
        remove all of them or the registry offers faces whose file is gone."""
        registry = FontRegistry()
        registry._fonts = {
            f"pack.ttc#{i}": FontCoverage(f"pack.ttc#{i}", "P") for i in range(3)
        }
        registry._fonts["other.ttf"] = FontCoverage("other.ttf", "O")
        assert registry.unload_font("pack.ttc") == 3
        assert list(registry.faces) == ["other.ttf"]

    def test_unloading_something_absent_is_not_an_error(self):
        assert FontRegistry().unload_font("nope.ttf") == 0


class TestEmbeddingPermission:
    """OS/2 fsType is the vendor's embedding permission, and real fonts do
    not respect the spec's claim that Restricted excludes the others."""

    @pytest.fixture(scope="class")
    def main(self):
        return pytest.importorskip("main")

    @pytest.mark.parametrize("value,restricted", [
        (0x0000, False),   # Installable
        (0x0002, True),    # Restricted
        (0x0004, False),   # Preview & Print
        (0x0008, False),   # Editable
        (0x000E, True),    # Restricted AND Preview&Print AND Editable
    ])
    def test_bit_one_decides_regardless_of_the_others(self, main, value, restricted):
        assert bool(value & main.FSTYPE_RESTRICTED) is restricted

    def test_the_contradictory_case_defeats_a_naive_check(self, main):
        """fsType=14 is a real value on a real installed face. A check that
        asks "is Preview&Print or Editable set?" passes it."""
        value = 0x000E
        naive_allows = bool(value & (main.FSTYPE_PREVIEW_PRINT | main.FSTYPE_EDITABLE))
        assert naive_allows is True
        assert bool(value & main.FSTYPE_RESTRICTED) is True

    def test_no_subset_is_read_separately_from_embedding(self, main):
        assert bool(0x0108 & main.FSTYPE_NO_SUBSET) is True
        assert bool(0x0108 & main.FSTYPE_RESTRICTED) is False

    def test_an_unreadable_font_fails_open(self, main, tmp_path):
        """A font whose table cannot be parsed is treated as installable,
        matching how the rest of the registry degrades."""
        junk = tmp_path / "not-a-font.ttf"
        junk.write_bytes(b"nope")
        permission = main._embedding_permission(junk)
        assert permission["readable"] is False
        assert permission["restricted"] is False

    def test_a_real_face_is_read(self, main):
        if LATIN_FONT is None:
            pytest.skip("no Latin face on this machine")
        permission = main._embedding_permission(LATIN_FONT)
        assert permission["readable"] is True
        assert isinstance(permission["fs_type"], int)


class TestServedPathGuard:
    """/api/font-file resolved ANY readable path. Harmless while fonts only
    came from fixed system directories; not once callers can place files."""

    @pytest.fixture(scope="class")
    def main(self):
        return pytest.importorskip("main")

    def test_a_font_inside_a_font_root_is_served(self, main):
        if LATIN_FONT is None:
            pytest.skip("no Latin face on this machine")
        assert main._is_served_font(LATIN_FONT) is True

    @pytest.mark.parametrize("path", [
        "C:/Windows/System32/drivers/etc/hosts",
        "/etc/passwd",
    ])
    def test_a_path_outside_every_font_root_is_refused(self, main, path):
        assert main._is_served_font(Path(path)) is False

    def test_the_project_database_is_refused(self, main):
        assert main._is_served_font(ROOT / "server" / "tofu.db") is False

    def test_traversal_out_of_a_font_root_is_refused(self, main):
        roots = main._font_dirs()
        if not roots:
            pytest.skip("no font directory on this machine")
        assert main._is_served_font(Path(roots[0]) / ".." / "secret.txt") is False


class TestPackArchiveSafety:
    """extractall() would happily write a member that names somewhere else."""

    @pytest.fixture(scope="class")
    def main(self):
        return pytest.importorskip("main")

    @pytest.mark.parametrize("member", [
        "../escape.ttf",
        "a/../../escape.ttf",
        "/abs/escape.ttf",          # POSIX-absolute
        "\\abs\\escape.ttf",        # backslash-absolute
        "C:/abs/escape.ttf",        # drive letter
        "C:\\abs\\escape.ttf",
        "a\\..\\..\\escape.ttf",    # climbing with backslashes
        "",
    ])
    def test_escaping_members_are_rejected(self, main, member):
        assert main._unsafe_archive_member(member) is True

    def test_the_windows_absolute_path_trap(self, main):
        """On Windows, Path("/abs/evil.ttf").is_absolute() is FALSE -- an
        absolute Windows path needs a drive letter -- so the obvious
        pathlib check lets a root-anchored member through. This is the case
        that broke the first version of the guard."""
        assert Path("/abs/evil.ttf").is_absolute() is (sys.platform != "win32")
        assert main._unsafe_archive_member("/abs/evil.ttf") is True

    @pytest.mark.parametrize("member", [
        "fonts/Regular.ttf", "pack.json", "fonts/sub/Bold.otf", "a..b/ok.ttf",
    ])
    def test_ordinary_members_are_accepted(self, main, member):
        assert main._unsafe_archive_member(member) is False

    def test_a_pack_zip_round_trips(self, tmp_path):
        if LATIN_FONT is None:
            pytest.skip("no Latin face on this machine")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("pack.json", '{"name":"demo","license":"OFL-1.1"}')
            archive.write(LATIN_FONT, f"fonts/{LATIN_FONT.name}")
        buf.seek(0)
        with zipfile.ZipFile(buf) as archive:
            names = archive.namelist()
        assert "pack.json" in names
        assert any(Path(n).suffix.lower() in FONT_EXTS for n in names)

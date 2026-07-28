## 🍢 /api/fonts — ranked dropdown subset vs. full Font Manager catalog
"""
The dropdown and the Font Manager read the same endpoint in two modes, and
the difference between them is load-bearing: the default caps at 24
families (which is why doppelganger.ts and TargetPreviewCanvas.tsx resolve
font paths without consulting the list at all), while ``full=true`` returns
the whole installed library so a font outside that cut can be picked.
"""

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from tofu.layers.fonts import FontCoverage, FontRegistry
from tofu.layers.sift import CATEGORIES

httpx = pytest.importorskip("httpx")


class ASGIClient:
    """Minimal sync client over the ASGI app.

    Not starlette's TestClient: the pinned starlette (0.36) passes ``app=``
    to httpx.Client, which httpx removed in 0.28, so TestClient raises
    before it can issue a request in this environment. ASGITransport is the
    supported route and needs nothing from starlette. main.app registers no
    startup/lifespan hooks, so skipping the lifespan costs us nothing.
    """

    def __init__(self, app):
        self._app = app

    def _call(self, method, url, **kwargs):
        async def go():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as c:
                return await getattr(c, method)(url, **kwargs)

        return asyncio.run(go())

    def get(self, url, params=None):
        return self._call("get", url, params=params)

    def post(self, url, files=None, params=None):
        return self._call("post", url, files=files, params=params)

    def delete(self, url):
        return self._call("delete", url)


@pytest.fixture(scope="module")
def client():
    import main

    # Discovery is a full rglob of the font directory with fontTools; the
    # server keeps one process-global validator for exactly this reason, so
    # build the client once for the module rather than per test.
    return ASGIClient(main.app)


@pytest.fixture(scope="module")
def has_fonts(client):
    import main

    if main.get_validator().font_registry is None:
        pytest.skip("no font library on this machine")
    return True


def _families(client, lang="fr", **params):
    r = client.get("/api/fonts", params={"lang": lang, **params})
    assert r.status_code == 200, r.text
    return r.json()


# -- contract ------------------------------------------------------------

def test_unknown_language_is_rejected(client):
    assert client.get("/api/fonts", params={"lang": "xx-nope"}).status_code == 400


def test_default_mode_truncates_to_24_families(client, has_fonts):
    body = _families(client)
    assert len(body["families"]) <= 24
    assert body["script"] == "Latn"  # ISO 15924, not the script's English name


def test_limit_is_still_honoured(client, has_fonts):
    assert len(_families(client, limit=3)["families"]) <= 3


def test_full_mode_returns_at_least_as_much_as_the_default(client, has_fonts):
    default = _families(client)["families"]
    full = _families(client, full=True)["families"]
    assert len(full) >= len(default)


def test_full_mode_omits_the_flat_face_list(client, has_fonts):
    """The manager picks per family, so a ranked face list would be
    computed for nobody."""
    assert _families(client, full=True)["fonts"] == []
    assert _families(client)["fonts"] != []


def test_full_mode_ignores_limit(client, has_fonts):
    """`full` is an explicit mode, not a bigger number — a stray limit must
    not silently truncate the catalog again."""
    assert len(_families(client, full=True, limit=2)["families"]) > 2


def test_full_mode_is_cached_by_script_and_language(client, has_fonts):
    import main

    main._CATALOG_CACHE.clear()
    body = _families(client, full=True)
    assert (body["script"], "fr") in main._CATALOG_CACHE
    assert _families(client, full=True)["families"] == body["families"]


# -- payload shape -------------------------------------------------------

def test_every_family_carries_a_browsing_category(client, has_fonts):
    for fam in _families(client, full=True)["families"]:
        assert fam["category"] in CATEGORIES


def test_default_mode_carries_categories_too(client, has_fonts):
    for fam in _families(client)["families"]:
        assert fam["category"] in CATEGORIES


def test_weights_are_grouped_sorted_and_flagged(client, has_fonts):
    for fam in _families(client, full=True)["families"]:
        assert fam["weights"], f"{fam['family']} has no faces"
        classes = [w["weight_class"] for w in fam["weights"]]
        assert classes == sorted(classes)
        for w in fam["weights"]:
            assert set(w) == {"path", "subfamily", "weight_class", "italic", "coverage"}
            assert isinstance(w["italic"], bool)


def test_best_path_is_the_highest_coverage_face(client, has_fonts):
    for fam in _families(client, full=True)["families"]:
        assert fam["best_path"] in {w["path"] for w in fam["weights"]}
        assert fam["best_coverage"] == pytest.approx(
            max(w["coverage"] for w in fam["weights"]), abs=1e-4
        )


def test_families_are_ranked_by_coverage(client, has_fonts):
    covers = [f["best_coverage"] for f in _families(client, full=True)["families"]]
    assert covers == sorted(covers, reverse=True)


def test_collection_faces_keep_their_hash_index(client, has_fonts):
    """A '.ttc#2' path is how the browser's @font-face and the server's
    /api/font-file extractor address one face inside a collection."""
    paths = [
        w["path"]
        for fam in _families(client, full=True)["families"]
        for w in fam["weights"]
    ]
    collection = [p for p in paths if "#" in p]
    if not collection:
        pytest.skip("no font collections installed")
    for p in collection:
        base, _, index = p.rpartition("#")
        assert index.isdigit()
        assert Path(base).suffix.lower() in {".ttc", ".otc"}


# -- registry-level behaviour, without touching the disk ------------------

def _registry() -> FontRegistry:
    glyphs = {ord(ch) for ch in "MURS "}
    registry = FontRegistry()
    registry._fonts = {
        "thin.ttf": FontCoverage("thin.ttf", "Fake Sans", "Thin", 100, glyphs),
        "bold.ttf": FontCoverage("bold.ttf", "Fake Sans", "Bold", 700, glyphs),
        "serif.ttf": FontCoverage("serif.ttf", "Fake Serif", "Regular", 400, set()),
    }
    return registry


def test_limit_none_returns_every_family():
    assert len(_registry().families_with_weights("Latin", None, limit=None)) == 2


def test_limit_still_cuts_the_ranked_tail():
    ranked = _registry().families_with_weights("Latin", None, limit=1)
    assert [f["family"] for f in ranked] == ["Fake Sans"]


def test_faces_group_under_one_family():
    fams = {f["family"]: f for f in _registry().families_with_weights("Latin", None, limit=None)}
    assert [w["subfamily"] for w in fams["Fake Sans"]["weights"]] == ["Thin", "Bold"]


def test_zero_coverage_family_still_reports_a_usable_path():
    """`cov > best_cov` never fires when every face scores 0, so best_path
    has to fall back rather than come back None."""
    fams = {f["family"]: f for f in _registry().families_with_weights("Latin", None, limit=None)}
    assert fams["Fake Serif"]["best_coverage"] == 0.0
    assert fams["Fake Serif"]["best_path"] == "serif.ttf"


def test_each_face_is_scored_exactly_once(monkeypatch):
    """coverage() only memoizes when extra_chars is None, so a language with
    LANG_REQUIRED extras used to re-walk the whole script sample three times
    per face. An unlimited catalog is only affordable because it does not.
    """
    registry = _registry()
    calls = []
    original = registry.coverage
    monkeypatch.setattr(
        registry, "coverage",
        lambda path, script, extra_chars=None: (calls.append(path), original(path, script, extra_chars=extra_chars))[1],
    )
    registry.families_with_weights("Latin", "vi", limit=None)
    assert sorted(calls) == sorted(registry._fonts)


# -- runtime installation -------------------------------------------------
# No OS-level font install is involved anywhere below: scribe renders through
# ImageFont.truetype(path), the preview is served over /api/font-file, and
# the registry keys on absolute paths. The system font directory was only
# ever a discovery convenience.

@pytest.fixture
def clean_user_dir():
    """Remove anything these tests install, whatever they assert."""
    import main
    from tofu.layers.fonts import USER_SUBDIR

    target = ROOT / "server" / "font-library" / USER_SUBDIR
    before = set(target.iterdir()) if target.is_dir() else set()
    yield target
    if target.is_dir():
        for path in set(target.iterdir()) - before:
            registry = main.get_validator().font_registry
            if registry is not None:
                registry.unload_font(str(path))
            path.unlink(missing_ok=True)
    main._CATALOG_CACHE.clear()


def _font_bytes():
    from conftest import LATIN_FONT
    if LATIN_FONT is None:
        pytest.skip("no Latin face on this machine")
    return LATIN_FONT.read_bytes()


def test_upload_rejects_a_non_font_extension(client, clean_user_dir):
    r = client.post("/api/fonts/upload", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 400


def test_upload_rejects_a_file_fonttools_cannot_read(client, clean_user_dir):
    """A corrupt file left on disk would fail every future discover() of
    that directory, silently, for the life of the deployment."""
    r = client.post("/api/fonts/upload", files={"file": ("broken.ttf", b"not a font")})
    assert r.status_code == 400
    assert not (clean_user_dir / "broken.ttf").exists()


def test_upload_installs_and_appears_in_the_catalog(client, has_fonts, clean_user_dir):
    import main

    name = "tofu-upload-test.ttf"
    before = len(_families(client, full=True)["families"])
    r = client.post("/api/fonts/upload", files={"file": (name, _font_bytes())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["faces"] >= 1 and body["source"] == "user"
    assert (clean_user_dir / name).is_file()
    # available immediately, with no restart and no re-discovery
    assert len(_families(client, full=True)["families"]) >= before


def test_upload_clears_the_catalog_cache(client, has_fonts, clean_user_dir):
    """_CATALOG_CACHE's own comment says it is safe because fonts on disk do
    not change while we run. Runtime install is exactly what breaks that."""
    import main

    _families(client, full=True)
    assert main._CATALOG_CACHE
    client.post("/api/fonts/upload", files={"file": ("tofu-cache-test.ttf", _font_bytes())})
    assert not main._CATALOG_CACHE


def test_upload_refuses_to_overwrite(client, has_fonts, clean_user_dir):
    """Replacing bytes at an existing path would strand scribe's
    (font_family, size) cache on the previous file."""
    name = "tofu-dupe-test.ttf"
    assert client.post("/api/fonts/upload", files={"file": (name, _font_bytes())}).status_code == 200
    assert client.post("/api/fonts/upload", files={"file": (name, _font_bytes())}).status_code == 409


def test_font_file_refuses_a_path_outside_the_font_roots(client):
    r = client.get("/api/font-file", params={"path": str(ROOT / "server" / "tofu.db")})
    assert r.status_code == 404


def test_packs_list_is_available_even_with_none_installed(client):
    r = client.get("/api/fonts/packs")
    assert r.status_code == 200 and isinstance(r.json()["packs"], list)


def test_pack_install_rejects_a_non_zip(client):
    r = client.post("/api/fonts/packs/install", files={"file": ("p.zip", b"not a zip")})
    assert r.status_code == 400


def test_pack_install_rejects_zip_slip(client):
    import io, zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("../escaped.ttf", _font_bytes())
    r = client.post(
        "/api/fonts/packs/install", files={"file": ("evil.zip", buf.getvalue())}
    )
    assert r.status_code == 400
    assert "unsafe path" in r.text


def test_pack_round_trip(client, has_fonts):
    import io, zipfile, main

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("pack.json", '{"name":"tofu-test-pack","license":"OFL-1.1"}')
        archive.writestr("fonts/packed.ttf", _font_bytes())
    try:
        r = client.post(
            "/api/fonts/packs/install",
            files={"file": ("tofu-test-pack.zip", buf.getvalue())},
        )
        assert r.status_code == 200, r.text
        assert r.json()["faces"] >= 1
        names = [p["name"] for p in client.get("/api/fonts/packs").json()["packs"]]
        assert "tofu-test-pack" in names

        removed = client.delete("/api/fonts/packs/tofu-test-pack")
        assert removed.status_code == 200
        # the registry must forget the faces, or it keeps offering families
        # whose files are gone and the failure surfaces at render time
        assert removed.json()["faces_unloaded"] >= 1
        names = [p["name"] for p in client.get("/api/fonts/packs").json()["packs"]]
        assert "tofu-test-pack" not in names
    finally:
        import shutil
        shutil.rmtree(
            ROOT / "server" / "font-library" / "packs" / "tofu-test-pack", ignore_errors=True
        )
        main._CATALOG_CACHE.clear()

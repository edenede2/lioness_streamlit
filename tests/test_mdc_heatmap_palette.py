import json
import numpy as np
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from app_helpers import data
from scripts.mdc_heatmap_palette import PALETTE, heatmap_cmap
from scripts.build_brain_mdc_heatmaps import paired_matrix, figure
from scripts.build_muscle_mdc_heatmaps import render


def test_low_weights_are_yellow_and_high_weights_are_red():
    cmap = heatmap_cmap()
    np.testing.assert_allclose(cmap(0.0), [1, 1, 0, 1])
    np.testing.assert_allclose(cmap(1.0), [1, 0, 0, 1])
    np.testing.assert_allclose(cmap(np.nan), [1, 1, 1, 1])
    assert np.diff(cmap(np.linspace(0, 1, 256))[:, 1]).max() <= 0


def test_brain_palette_preserves_ad_control_values_order_and_scale():
    ad = np.array([[0, .1, .8], [.1, 0, .3], [.8, .3, 0]])
    control = np.array([[0, .4, .2], [.4, 0, .7], [.2, .7, 0]])
    combined, order = paired_matrix(ad, control)
    original = combined.copy()
    fig, cap = figure(combined, np.array([0, 1, 2]), order, "Test modules", 7,
                      {"total": 1., "ts": np.nan, "ct": 1.})
    image = fig.axes[0].images[0]
    assert image.cmap.name == PALETTE["name"]
    assert image.get_clim() == (0, cap)
    assert cap == np.quantile(combined[np.isfinite(combined)], .99)
    np.testing.assert_array_equal(combined, original)
    np.testing.assert_allclose(image.cmap(image.norm(0)), [1, 1, 0, 1])
    np.testing.assert_allclose(image.cmap(image.norm(cap)), [1, 0, 0, 1])
    plt.close(fig)


def test_muscle_palette_preserves_display_cap_and_matrices(tmp_path):
    ad = np.array([[0, .1, .8], [.1, 0, .3], [.8, .3, 0]])
    control = np.array([[0, .4, .2], [.4, 0, .7], [.2, .7, 0]])
    a, c = ad.copy(), control.copy()
    path = tmp_path / "M7.png"
    row = render(7, ad, control, path, {"total": 1., "TS": np.nan, "CT": 1.})
    assert row["color_palette"] == PALETTE["name"]
    np.testing.assert_array_equal(ad, a)
    np.testing.assert_array_equal(control, c)
    combined, _ = paired_matrix(ad, control)
    assert row["display_q99"] == np.quantile(combined[np.isfinite(combined)], .99)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
        assert ((pixels[..., 0] == 255) & (pixels[..., 1] == 255) & (pixels[..., 2] == 0)).any()
        assert ((pixels[..., 0] == 255) & (pixels[..., 1] == 0) & (pixels[..., 2] == 0)).any()


def test_muscle_catalog_is_variant_specific_and_old_routes_remain_valid(tmp_path, monkeypatch):
    monkeypatch.setitem(data.MODULE_SET_DIRS, "muscle_control_l4", tmp_path)
    root = tmp_path / "mdc_split_heatmaps"
    root.mkdir()
    assert data._mdc_split_heatmap_path("muscle_control_l4", 7, "original") == root / "all_edges/M7.png"
    (root / "catalog.json").write_text(json.dumps({
        "status": "complete", "module_set": "muscle_control_l4",
        "image_directories": {"original": "yellow_original", "m5_high_no_q90": "yellow_no_q90"},
    }))
    assert data._mdc_split_heatmap_path("muscle_control_l4", 7, "original") == root / "yellow_original/M7.png"
    assert data._mdc_split_heatmap_path("muscle_control_l4", 7, "m5_high_no_q90") == root / "yellow_no_q90/M7.png"


def test_variant_catalog_rejects_unsafe_directory(tmp_path, monkeypatch):
    monkeypatch.setitem(data.MODULE_SET_DIRS, "muscle_control_l4", tmp_path)
    root = tmp_path / "mdc_split_heatmaps"
    root.mkdir()
    (root / "catalog.json").write_text(json.dumps({
        "status": "complete", "module_set": "muscle_control_l4",
        "image_directories": {"m5_high_no_q90": "../another_dataset"},
    }))
    with pytest.raises(ValueError, match="Unsafe"):
        data._mdc_split_heatmap_path("muscle_control_l4", 7, "m5_high_no_q90")


def test_parallel_upload_checks_every_file_through_app_reader(tmp_path, monkeypatch):
    from scripts import publish_brain_mdc_heatmaps as publisher
    from types import SimpleNamespace
    import threading

    paths = [tmp_path / f"M{n}.png" for n in range(6)]
    for path in paths:
        path.write_bytes(b"fixture")
    observed = []
    uploads = []
    lock = threading.Lock()
    writer, reader = SimpleNamespace(credentials=object()), object()
    monkeypatch.setattr(publisher, "AuthorizedSession", lambda credentials: SimpleNamespace(credentials=credentials))
    monkeypatch.setattr(publisher, "list_children", lambda session, parent: (list(observed), 1))

    def create(session, path, name, parent, digest):
        assert session is not writer and session.credentials is writer.credentials
        assert parent == "module-images"
        with lock:
            uploads.append(name)
            observed.append({"id": name, "name": name, "size": path.stat().st_size,
                             "md5Checksum": digest})
    monkeypatch.setattr(publisher, "create_file", create)
    result = publisher.sync_verified(writer, reader, "module-images", paths, workers=4)
    assert set(result) == set(uploads) == {p.name for p in paths}
    # Already verified assets need no further write calls.
    publisher.sync_verified(writer, reader, "module-images", paths, workers=4)
    assert len(uploads) == len(paths)
    observed[0]["md5Checksum"] = "incorrect"
    monkeypatch.setattr(publisher, "update_file", lambda *args: None)
    with pytest.raises(RuntimeError, match="checksum failed"):
        publisher.sync_verified(writer, reader, "module-images", paths, workers=4)

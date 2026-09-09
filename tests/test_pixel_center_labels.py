import numpy as np
import driftsense.generate as g


def test_pixel_center_label_selects_its_own_jitter_row(monkeypatch):
    canvas = np.zeros((1200, 1200), np.uint8)
    shift = np.arange(1000, dtype=np.float32) % 2 * 3
    monkeypatch.setattr(g, 'generate_fine_canvas_zoned', lambda *a, **kw: {'canvas': canvas})
    monkeypatch.setattr(g, 'image_search_traced', lambda *a: (canvas[:1000,:1000], shift, 0.0))
    monkeypatch.setattr(g, '_pick_visible_crop_origin', lambda *a: (80, 84))
    monkeypatch.setattr(g.sem_imaging, 'image_reference', lambda crop, **kw: crop.copy())
    spec = g.PoseSpec(rotation_deg=(1,1), magnification=(10,10))
    old = g.make_pairs(7, ['finfet'], 'low', crops=1, pose=spec)[0]
    new = g.make_pairs(7, ['finfet'], 'low', crops=1, pose=spec, pixel_center_labels=True)[0]
    x,y = g.apply_affine_point(g.search_affine(1200,1000,10,1),579.5,583.5)
    assert np.allclose([new['gt_x_raw'],new['gt_y_raw']],[x,y])
    assert new['gt_x']==x-shift[round(y)]
    assert np.array_equal(old['reference'],new['reference'])
    assert np.array_equal(old['search'],new['search'])
    assert old['gt_y_raw']!=new['gt_y_raw']

import numpy as np

from actuate.viz.preview import _write_browser_video


def test_portable_preview_is_browser_decodable_h264(tmp_path):
    import av

    output = tmp_path / "preview.mp4"
    frames = [
        np.full((64, 96, 3), (index * 40, 80, 160), dtype=np.uint8)
        for index in range(3)
    ]
    _write_browser_video(output, frames, hold_frames=2, fps=10)

    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))

    assert stream.codec_context.name == "h264"
    assert stream.codec_context.format.name == "yuv420p"
    assert len(decoded) == 6

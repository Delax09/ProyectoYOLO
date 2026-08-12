from pathlib import Path

from src.detector.yolo_detector import resolve_output_path


def test_resolve_output_path_for_directory_name(tmp_path):
    source = tmp_path / "sample.jpg"

    output = resolve_output_path(tmp_path / "output", source)

    assert output.parent == tmp_path / "output"
    assert output.name == "out_sample.jpg"

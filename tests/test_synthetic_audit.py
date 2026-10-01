import pytest
from PIL import Image, ImageDraw

from veyra.synthetic_audit import diagram_objects, visual_predicate


def test_pixel_oracle_recovers_count_order_color_and_shape(tmp_path):
    path = tmp_path / "objects.png"
    with Image.new("RGB", (384, 256), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 20, 70, 80), fill="red")
        draw.ellipse((104, 30, 164, 90), fill="blue")
        draw.polygon([(228, 40), (198, 100), (258, 100)], fill="green")
        image.save(path)
    objects = diagram_objects(path)
    assert objects == [("red", "square"), ("blue", "circle"), ("green", "triangle")]
    assert visual_predicate(objects, "the image contains at least 3 objects")
    assert not visual_predicate(objects, "the image contains at least 4 objects")
    assert visual_predicate(objects, "the rightmost object is a triangle")
    assert visual_predicate(objects, "the leftmost object's color is red")


def test_pixel_oracle_supports_blank_training_counterfactual(tmp_path):
    path = tmp_path / "blank.png"
    with Image.new("RGB", (384, 256), "white") as image:
        image.save(path)
    assert diagram_objects(path) == []
    assert not visual_predicate([], "the image contains at least 1 objects")
    with pytest.raises(ValueError):
        visual_predicate([], "the leftmost object is a circle")

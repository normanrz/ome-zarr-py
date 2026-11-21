"""Tests for RFC-15 (snake_case metadata naming) under OME-Zarr version 0.9dev42."""

import json

import numpy as np
import ome_zarr_models.v09dev42.coordinate_transforms as ozmt_v09
import pytest
import zarr
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.v09dev42 import (
    HCS as Models09HCS,
)
from ome_zarr_models.v09dev42 import (
    Image as Models09Image,
)
from ome_zarr_models.v09dev42 import (
    ImageLabel as Models09ImageLabel,
)
from ome_zarr_models.v09dev42 import (
    Labels as Models09Labels,
)
from ome_zarr_models.v09dev42 import (
    Scene as Models09Scene,
)
from ome_zarr_models.v09dev42 import (
    Well as Models09Well,
)
from pydantic import TypeAdapter, ValidationError

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale, OMEZarrScene
from ome_zarr.classes.tnd_utils import _ozmp_tf_to_tnd
from ome_zarr.cli import main as cli_main
from ome_zarr.csv import dict_to_zarr
from ome_zarr.data import create_zarr
from ome_zarr.format import FormatV04, FormatV05, FormatV09Dev42, format_from_version
from ome_zarr.io import parse_url
from ome_zarr.reader import Label, Multiscales, Node, Plate, Reader, Well
from ome_zarr.utils import download, find_multiscales, info
from ome_zarr.writer import (
    write_image,
    write_label_metadata,
    write_labels,
    write_multiscale,
    write_multiscale_labels,
    write_multiscales_metadata,
    write_plate_metadata,
    write_well_metadata,
)


def _sample_data(shape=(2, 16, 16), dtype=np.uint8):
    rng = np.random.default_rng(42)
    return rng.integers(0, 50, size=shape, dtype=dtype)


class TestImageAndLabelsClassV09Dev42:
    def test_multiscale_image_roundtrip_and_snake_case_metadata(self, tmp_path):
        data = _sample_data((2, 16, 16))
        img = OMEZarrImage(
            data=data,
            axes="cyx",
            scale={"c": 1.0, "y": 0.5, "x": 0.5},
            axes_units={"y": "micrometer", "x": "micrometer"},
            name="example",
        )

        # Pass v09dev42 CoordinateSystem (with long_name) and ByDimension / MapAxis transform
        world_cs = ozmt_v09.CoordinateSystem(
            name="world",
            axes=(
                ozmt_v09.Axis(name="c", type="channel", discrete=True),
                ozmt_v09.Axis(
                    name="y", type="space", unit="micrometer", long_name="height"
                ),
                ozmt_v09.Axis(
                    name="x", type="space", unit="micrometer", long_name="width"
                ),
            ),
        )
        by_dim_tf = ozmt_v09.ByDimension(
            input=ozmt_v09.CoordinateSystemIdentifier(name="physical"),
            output=ozmt_v09.CoordinateSystemIdentifier(name="world"),
            transformations=(
                ozmt_v09.ByDimensionTransform(
                    input_axes=(0,),
                    output_axes=(0,),
                    transformation=ozmt_v09.Identity(),
                ),
                ozmt_v09.ByDimensionTransform(
                    input_axes=(1, 2),
                    output_axes=(1, 2),
                    transformation=ozmt_v09.MapAxis(map_axis=(1, 0)),
                ),
            ),
        )

        label_data = (data[0] > 25).astype(np.uint8)
        lbl_img = OMEZarrImage(
            data=label_data,
            axes="yx",
            scale={"y": 0.5, "x": 0.5},
            name="seg",
        )
        lbl_ms = OMEZarrLabels(image=lbl_img, scale_factors=[2])

        ms = OMEZarrMultiscale(
            image=img,
            scale_factors=[2],
            coordinate_systems=[world_cs],
            coordinate_transformations=(by_dim_tf,),
            labels=lbl_ms,
            channel_names=["DAPI", "GFP"],
            channel_colors=["0000FF", "00FF00"],
        )

        store_path = tmp_path / "image.zarr"
        ms.to_ome_zarr(store_path, version="0.9dev42", overwrite=True)

        # Verify raw zarr.json on disk uses snake_case exclusively
        root_json = json.loads((store_path / "zarr.json").read_text(encoding="utf-8"))
        ome = root_json["attributes"]["ome"]
        assert ome["version"] == "0.9dev42"

        ms_meta = ome["multiscales"][0]
        assert "coordinate_systems" in ms_meta
        assert "coordinateSystems" not in ms_meta
        assert "coordinate_transformations" in ms_meta
        assert "coordinateTransformations" not in ms_meta

        # Check long_name on axis
        world_axes = ms_meta["coordinate_systems"][1]["axes"]
        assert world_axes[1]["long_name"] == "height"
        assert "longName" not in world_axes[1]

        # Check dataset coordinate_transformations
        ds0 = ms_meta["datasets"][0]
        assert "coordinate_transformations" in ds0
        assert "coordinateTransformations" not in ds0

        # Check by_dimension / input_axes / output_axes / map_axis
        top_tf = ms_meta["coordinate_transformations"][0]
        assert top_tf["type"] == "by_dimension"
        assert "input_axes" in top_tf["transformations"][0]
        assert "output_axes" in top_tf["transformations"][0]
        assert "inputAxes" not in top_tf["transformations"][0]
        assert top_tf["transformations"][1]["transformation"]["type"] == "map_axis"
        assert top_tf["transformations"][1]["transformation"]["map_axis"] == [1, 0]

        # Transitional omero metadata is NOT renamed per RFC-15
        assert "omero" in ome
        assert len(ome["omero"]["channels"]) == 2

        # Verify label metadata uses image_label and label_value
        seg_json = json.loads(
            (store_path / "labels" / "seg" / "zarr.json").read_text(encoding="utf-8")
        )
        seg_ome = seg_json["attributes"]["ome"]
        assert seg_ome["version"] == "0.9dev42"
        assert "image_label" in seg_ome
        assert "image-label" not in seg_ome
        for color in seg_ome["image_label"]["colors"]:
            assert "label_value" in color
            assert "label-value" not in color
        for prop in seg_ome["image_label"]["properties"]:
            assert "label_value" in prop
            assert "label-value" not in prop

        # Validate with ome-zarr-models v09dev42
        grp = zarr.open_group(store_path, mode="r")
        model = open_ome_zarr(grp)
        assert isinstance(model, Models09Image)
        assert isinstance(open_ome_zarr(grp["labels"]), Models09Labels)
        assert isinstance(open_ome_zarr(grp["labels"]["seg"]), Models09ImageLabel)

        # Read back with class-based API
        ms_read = OMEZarrMultiscale.from_ome_zarr(str(store_path))
        assert np.array_equal(ms_read.images[0].data.compute(), data)
        assert ms_read.omero is not None
        assert ms_read.labels is not None
        assert "seg" in ms_read.labels
        assert ms_read.labels["seg"].image_label is not None
        assert ms_read.get_local_coordinate_system("world") is not None

        # Read back with legacy Reader
        reader = Reader(parse_url(str(store_path)))
        nodes = list(reader())
        assert len(nodes) == 3
        assert any(isinstance(s, Multiscales) for s in nodes[0].specs)
        assert "coordinate_transformations" in nodes[0].metadata
        assert "coordinateTransformations" not in nodes[0].metadata
        label_nodes = [n for n in nodes if any(isinstance(s, Label) for s in n.specs)]
        assert len(label_nodes) == 1

    def test_rejects_old_spellings_in_09dev42_image_and_label(self, tmp_path):
        store_path = tmp_path / "bad_img.zarr"
        grp = zarr.open_group(store_path, mode="w", zarr_format=3)
        grp.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint8")[:] = 0
        grp.attrs["ome"] = {
            "version": "0.9dev42",
            "multiscales": [
                {
                    "name": "bad",
                    "coordinateSystems": [
                        {
                            "name": "physical",
                            "axes": [
                                {"name": "y", "type": "space"},
                                {"name": "x", "type": "space"},
                            ],
                        }
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [1.0, 1.0],
                                    "input": {"path": "0"},
                                    "output": {"name": "physical"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        with pytest.raises(ValidationError):
            OMEZarrMultiscale.from_ome_zarr(str(store_path))
        with pytest.raises(ValueError, match="coordinateSystems"):
            list(Reader(parse_url(str(store_path)))())

        # Reject image-label key in 0.9dev42
        lbl_path = tmp_path / "bad_label.zarr"
        lbl_grp = zarr.open_group(lbl_path, mode="w", zarr_format=3)
        lbl_grp.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint8")[:] = 0
        lbl_grp.attrs["ome"] = {
            "version": "0.9dev42",
            "multiscales": [
                {
                    "name": "lbl",
                    "coordinate_systems": [
                        {
                            "name": "physical",
                            "axes": [
                                {"name": "y", "type": "space"},
                                {"name": "x", "type": "space"},
                            ],
                        }
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinate_transformations": [
                                {
                                    "type": "scale",
                                    "scale": [1.0, 1.0],
                                    "input": {"path": "0"},
                                    "output": {"name": "physical"},
                                }
                            ],
                        }
                    ],
                }
            ],
            "image-label": {"colors": [{"label-value": 0, "rgba": [0, 0, 0, 0]}]},
        }
        with pytest.raises(ValueError, match="image-label"):
            OMEZarrMultiscale.from_ome_zarr(str(lbl_path))
        with pytest.raises(ValueError, match="image-label"):
            list(Reader(parse_url(str(lbl_path)))())

        # Reject label-value inside image_label in 0.9dev42
        lbl_grp.attrs["ome"] = {
            **{k: v for k, v in lbl_grp.attrs["ome"].items() if k != "image-label"},
            "image_label": {"colors": [{"label-value": 0, "rgba": [0, 0, 0, 0]}]},
        }
        with pytest.raises(ValidationError, match="label-value"):
            OMEZarrMultiscale.from_ome_zarr(str(lbl_path))
        with pytest.raises(ValueError, match="label-value"):
            list(Reader(parse_url(str(lbl_path)))())


class TestSceneV09Dev42:
    def test_scene_roundtrip_transforms_and_displacements(self, tmp_path):
        img_a = OMEZarrMultiscale(
            OMEZarrImage(data=_sample_data((16, 16)), name="imageA", axes="yx")
        )
        img_b = OMEZarrMultiscale(
            OMEZarrImage(data=_sample_data((16, 16)), name="imageB", axes="yx")
        )
        dfield = OMEZarrMultiscale(
            OMEZarrImage(
                data=np.zeros((2, 16, 16), dtype=np.float32),
                name="disp",
                axes="cyx",
                axes_types={"c": "displacement", "y": "space", "x": "space"},
            )
        )

        world_cs = ozmt_v09.CoordinateSystem(
            name="world",
            axes=(
                ozmt_v09.Axis(name="y", type="space", long_name="vertical"),
                ozmt_v09.Axis(name="x", type="space"),
            ),
        )
        tf_a = ozmt_v09.MapAxis(
            map_axis=(1, 0),
            input=ozmt_v09.CoordinateSystemIdentifier(path="imageA", name="physical"),
            output=ozmt_v09.CoordinateSystemIdentifier(name="world"),
        )
        tf_b = ozmt_v09.Displacements(
            path="coordinate_transformations/disp",
            input=ozmt_v09.CoordinateSystemIdentifier(name="world"),
            output=ozmt_v09.CoordinateSystemIdentifier(path="imageB", name="physical"),
        )

        scene = OMEZarrScene(
            images=[img_a, img_b],
            coordinate_systems=[world_cs],
            coordinate_transformations=[tf_a, tf_b],
            coordinate_displacements={"disp": dfield},
        )

        scene_path = tmp_path / "scene.zarr"
        scene.to_ome_zarr(scene_path, overwrite=True, version="0.9dev42")

        # Verify raw JSON on disk
        raw = json.loads((scene_path / "zarr.json").read_text(encoding="utf-8"))
        ome = raw["attributes"]["ome"]
        assert ome["version"] == "0.9dev42"
        assert "coordinate_systems" in ome["scene"]
        assert "coordinateSystems" not in ome["scene"]
        assert "coordinate_transformations" in ome["scene"]
        assert "coordinateTransformations" not in ome["scene"]
        assert ome["scene"]["coordinate_systems"][0]["axes"][0]["long_name"] == "vertical"
        assert ome["scene"]["coordinate_transformations"][0]["type"] == "map_axis"
        assert ome["scene"]["coordinate_transformations"][0]["map_axis"] == [1, 0]

        grp = zarr.open_group(scene_path, mode="r")
        assert "coordinate_transformations" in grp
        assert "disp" in grp["coordinate_transformations"]
        assert isinstance(open_ome_zarr(grp), Models09Scene)

        # find_multiscales & download preserve 0.9dev42
        found = find_multiscales(scene_path)
        assert len(found) == 1
        assert found[0][0] == scene_path / "imageA"

        dl_dir = tmp_path / "downloaded"
        dl_dir.mkdir()
        download(str(scene_path), str(dl_dir))
        dl_raw = json.loads(
            (dl_dir / "scene.zarr" / "zarr.json").read_text(encoding="utf-8")
        )
        assert dl_raw["attributes"]["ome"]["version"] == "0.9dev42"
        assert "coordinate_transformations" in dl_raw["attributes"]["ome"]["scene"]

        # Read back with OMEZarrScene.from_ome_zarr
        loaded = OMEZarrScene.from_ome_zarr(scene_path)
        assert set(loaded.images.keys()) == {"imageA", "imageB"}
        assert "disp" in loaded.coordinate_displacements
        assert len(loaded._graph.graph.nodes) == 3

    def test_scene_rejects_camel_case_in_09dev42(self, tmp_path):
        scene_path = tmp_path / "bad_scene.zarr"
        grp = zarr.open_group(scene_path, mode="w", zarr_format=3)
        grp.attrs["ome"] = {
            "version": "0.9dev42",
            "scene": {"coordinateTransformations": []},
        }
        with pytest.raises(ValidationError, match="coordinateTransformations"):
            OMEZarrScene.from_ome_zarr(scene_path)

    def test_tnd_utils_accepts_v09dev42_transforms(self):
        adapter = TypeAdapter(ozmt_v09.AnyTransform)
        for tf_dict, kwargs in [
            ({"type": "map_axis", "map_axis": [1, 0]}, {}),
            (
                {
                    "type": "project_axis",
                    "created_outputs": [1],
                    "dropped_inputs": [0],
                },
                {"source_ndim": 3},
            ),
            (
                {
                    "type": "by_dimension",
                    "transformations": [
                        {
                            "input_axes": [0],
                            "output_axes": [0],
                            "transformation": {"type": "scale", "scale": [2.0]},
                        },
                        {
                            "input_axes": [1],
                            "output_axes": [1],
                            "transformation": {
                                "type": "translation",
                                "translation": [5.0],
                            },
                        },
                    ],
                },
                {},
            ),
        ]:
            tf = adapter.validate_python(tf_dict)
            tnd_tf = _ozmp_tf_to_tnd(
                tf,
                "",
                kwargs.get("source_ndim"),
                kwargs.get("target_ndim"),
                {},
            )
            assert tnd_tf is not None


class TestLegacyFormatWriterReaderHCSV09Dev42:
    def test_format_lookup(self):
        fmt = format_from_version("0.9dev42")
        assert isinstance(fmt, FormatV09Dev42)
        assert fmt.zarr_format == 3

    def test_plate_well_image_legacy_writer_and_reader(self, tmp_path):
        fmt = FormatV09Dev42()
        plate_path = tmp_path / "plate.zarr"
        root = zarr.open_group(plate_path, mode="w", zarr_format=3)

        write_plate_metadata(
            root,
            rows=["A", "B"],
            columns=["1", "2"],
            wells=["A/1", "B/2"],
            fmt=fmt,
            name="test_plate",
            field_count=1,
            acquisitions=[
                {
                    "id": 0,
                    "name": "acq0",
                    "maximum_field_count": 2,
                    "start_time": 1000,
                    "end_time": 2000,
                }
            ],
        )

        plate_ome = root.attrs["ome"]
        assert plate_ome["version"] == "0.9dev42"
        assert "version" not in plate_ome["plate"]
        assert plate_ome["plate"]["wells"] == [
            {"path": "A/1", "row_index": 0, "column_index": 0},
            {"path": "B/2", "row_index": 1, "column_index": 1},
        ]
        assert plate_ome["plate"]["acquisitions"][0]["maximum_field_count"] == 2
        assert plate_ome["plate"]["acquisitions"][0]["start_time"] == 1000
        assert plate_ome["plate"]["acquisitions"][0]["end_time"] == 2000

        for well_path in ["A/1", "B/2"]:
            row, col = well_path.split("/")
            well_grp = root.require_group(row).require_group(col)
            write_well_metadata(well_grp, [{"path": "0", "acquisition": 0}], fmt=fmt)
            img_grp = well_grp.require_group("0")
            write_image(
                np.ones((1, 1, 1, 16, 16), dtype=np.uint8),
                img_grp,
                fmt=fmt,
                scale_factors=[2],
            )
            assert isinstance(open_ome_zarr(well_grp), Models09Well)

        # Validate entire HCS plate with ome-zarr-models v09dev42
        assert isinstance(open_ome_zarr(root), Models09HCS)

        # Read with legacy Reader & Node
        loc = parse_url(str(plate_path))
        assert loc is not None
        assert loc.version == "0.9dev42"
        nodes = list(Reader(loc)())
        assert len(nodes) == 1
        assert isinstance(nodes[0].specs[0], Plate)

        well_node = Node(parse_url(str(plate_path / "A" / "1")), list())
        assert isinstance(well_node.specs[0], Well)

    def test_plate_rejects_old_keys_in_09dev42(self, tmp_path):
        fmt = FormatV09Dev42()
        grp = zarr.open_group(tmp_path / "bad_plate.zarr", mode="w", zarr_format=3)

        with pytest.raises(ValueError, match="rowIndex"):
            write_plate_metadata(
                grp,
                rows=["A"],
                columns=["1"],
                wells=[{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
                fmt=fmt,
            )

        for old_key in ("maximumfieldcount", "starttime", "endtime"):
            with pytest.raises(ValueError, match=old_key):
                write_plate_metadata(
                    grp,
                    rows=["A"],
                    columns=["1"],
                    wells=["A/1"],
                    fmt=fmt,
                    acquisitions=[{"id": 0, old_key: 1}],
                )

    def test_legacy_multiscale_and_labels_writer_and_csv(self, tmp_path):
        fmt = FormatV09Dev42()
        img_path = tmp_path / "legacy_img.zarr"
        grp = zarr.open_group(img_path, mode="w", zarr_format=3)

        pyramid = [np.zeros((16, 16), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)]
        write_multiscale(pyramid, grp, fmt=fmt, axes="yx")
        assert grp.attrs["ome"]["version"] == "0.9dev42"
        ds0 = grp.attrs["ome"]["multiscales"][0]["datasets"][0]
        assert "coordinate_transformations" in ds0
        assert "coordinateTransformations" not in ds0

        # Reject old key in write_multiscales_metadata
        with pytest.raises(ValueError, match="coordinateTransformations"):
            write_multiscales_metadata(
                grp,
                datasets=[
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]}
                        ],
                    }
                ],
                fmt=fmt,
                axes="yx",
            )

        # write_multiscale_labels + write_label_metadata with snake_case
        lbl_pyramid = [
            np.ones((16, 16), dtype=np.uint8),
            np.ones((8, 8), dtype=np.uint8),
        ]
        write_multiscale_labels(
            lbl_pyramid,
            grp,
            name="0",
            fmt=fmt,
            axes="yx",
            label_metadata={
                "colors": [{"label_value": 1, "rgba": [255, 0, 0, 255]}],
                "properties": [{"label_value": 1, "cell_type": "neuron"}],
            },
        )
        lbl_ome = grp["labels"]["0"].attrs["ome"]
        assert "image_label" in lbl_ome
        assert "image-label" not in lbl_ome
        assert lbl_ome["image_label"]["colors"][0]["label_value"] == 1

        # Update properties via dict_to_zarr
        dict_to_zarr({"1": {"area": 42}}, str(img_path), "label_value")
        updated_props = grp["labels"]["0"].attrs["ome"]["image_label"]["properties"]
        assert updated_props[0]["area"] == 42

        # Reject old label-value in write_label_metadata
        with pytest.raises(ValueError, match="label-value"):
            write_label_metadata(
                grp["labels"],
                "0",
                colors=[{"label-value": 1, "rgba": [255, 0, 0, 255]}],
                fmt=fmt,
            )

        # write_labels with FormatV09Dev42
        write_labels(
            np.ones((16, 16), dtype=np.uint8),
            grp,
            name="seg2",
            fmt=fmt,
            axes="yx",
            scale_factors=[2],
        )
        assert "image_label" in grp["labels"]["seg2"].attrs["ome"]
        assert "image-label" not in grp["labels"]["seg2"].attrs["ome"]

    def test_create_zarr_and_cli_09dev42(self, tmp_path):
        zarr_dir = tmp_path / "cli_coins.zarr"
        cli_main(["create", "--format", "0.9dev42", str(zarr_dir)])

        raw = json.loads((zarr_dir / "zarr.json").read_text(encoding="utf-8"))
        assert raw["attributes"]["ome"]["version"] == "0.9dev42"
        assert "coordinate_systems" in raw["attributes"]["ome"]["multiscales"][0]

        nodes = list(info(str(zarr_dir)))
        assert len(nodes) == 3

        # Also verify older formats still use their original spellings
        v05_dir = tmp_path / "v05_coins.zarr"
        create_zarr(str(v05_dir), fmt=FormatV05())
        v05_raw = json.loads((v05_dir / "zarr.json").read_text(encoding="utf-8"))
        assert v05_raw["attributes"]["ome"]["version"] == "0.5"
        assert "coordinateTransformations" in (
            v05_raw["attributes"]["ome"]["multiscales"][0]["datasets"][0]
        )

        v04_dir = tmp_path / "v04_coins.zarr"
        create_zarr(str(v04_dir), fmt=FormatV04())
        v04_raw = json.loads((v04_dir / ".zattrs").read_text(encoding="utf-8"))
        assert v04_raw["multiscales"][0]["version"] == "0.4"
        assert "coordinateTransformations" in (
            v04_raw["multiscales"][0]["datasets"][0]
        )

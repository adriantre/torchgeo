# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyproj
import pytest
import rasterio
import shapely
import shapely.ops
import shapely.wkt
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from rasterio import DatasetReader
from rasterio.vrt import WarpedVRT
from shapely import Polygon

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    Sentinel1,
    Sentinel2,
    UnionDataset,
)


class TestSentinel1:
    @pytest.fixture(
        params=[
            # Only horizontal or vertical receive
            ['HH'],
            ['HV'],
            ['VV'],
            ['VH'],
            # Both horizontal and vertical receive
            ['HH', 'HV'],
            ['HV', 'HH'],
            ['VV', 'VH'],
            ['VH', 'VV'],
        ]
    )
    def dataset(self, request: SubRequest) -> Sentinel1:
        root = os.path.join('tests', 'data', 'sentinel1')
        bands = request.param
        transforms = nn.Identity()
        return Sentinel1(root, bands=bands, transforms=transforms)

    def test_getitem(self, dataset: Sentinel1) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: Sentinel1) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: Sentinel1) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Sentinel1) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Sentinel1(tmp_path)

    def test_empty_bands(self) -> None:
        with pytest.raises(AssertionError, match="'bands' cannot be an empty list"):
            Sentinel1(bands=[])

    @pytest.mark.parametrize('bands', [['HH', 'HH'], ['HH', 'HV', 'HH']])
    def test_duplicate_bands(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="'bands' contains duplicate bands"):
            Sentinel1(bands=bands)

    @pytest.mark.parametrize('bands', [['HH_HV'], ['HH', 'HV', 'HH_HV']])
    def test_invalid_bands(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="invalid band 'HH_HV'"):
            Sentinel1(bands=bands)

    @pytest.mark.parametrize(
        'bands', [['HH', 'VV'], ['HH', 'VH'], ['VV', 'HV'], ['HH', 'HV', 'VV', 'VH']]
    )
    def test_dual_transmit(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="'bands' cannot contain both "):
            Sentinel1(bands=bands)

    def test_invalid_index(self, dataset: Sentinel1) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[-1:-1, -1:-1, pd.Timestamp.min : pd.Timestamp.min]


class TestSentinel2:
    @pytest.fixture
    def dataset(self) -> Sentinel2:
        root = os.path.join('tests', 'data', 'sentinel2')
        res = (10.0, 10.0)
        transforms = nn.Identity()
        bands = [
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B09',
            'B11',
            'B12',
        ]
        return Sentinel2(root, res=res, transforms=transforms, bands=bands)

    def test_getitem(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: Sentinel2) -> None:
        assert len(dataset) == 4

    def test_and(self, dataset: Sentinel2) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Sentinel2) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Sentinel2(tmp_path)

    def test_plot(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_wrong_bands(self, dataset: Sentinel2) -> None:
        bands = ['B02']
        ds = Sentinel2(dataset.paths, res=dataset.res, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds.plot(x)

    def test_invalid_index(self, dataset: Sentinel2) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_float_res(self, dataset: Sentinel2) -> None:
        Sentinel2(dataset.paths, res=10.0, bands=dataset.bands)

    @pytest.mark.parametrize('dataset_type', [DatasetReader, WarpedVRT])
    @pytest.mark.parametrize('has_footprint', [True, False])
    def test_footprint_from_datasource_metadata_file(
        self,
        dataset: Sentinel2,
        monkeypatch: pytest.MonkeyPatch,
        dataset_type: type[DatasetReader] | type[WarpedVRT],
        has_footprint_tag: bool,
    ) -> None:
        footprint_wkt = dataset.index.geometry.to_crs(4326).values[0].wkt
        filepath = next(iter(dataset.files))

        class FakeMetadataSrc:
            def tags(self) -> dict[str, str]:
                return {'FOOTPRINT': footprint_wkt} if has_footprint_tag else {}

            def __enter__(self) -> 'FakeMetadataSrc':
                return self

            def __exit__(self, *args: object) -> None:
                pass

        # Open the real band file before patching rasterio.open
        real_src = rasterio.open(filepath)
        src_dataset: DatasetReader | WarpedVRT = (
            WarpedVRT(real_src) if dataset_type is WarpedVRT else real_src
        )
        bounds = src_dataset.bounds

        monkeypatch.setattr(rasterio, 'open', lambda _: FakeMetadataSrc())
        monkeypatch.setattr(os.path, 'exists', lambda _: True)

        result = dataset._footprint_from_datasource(src_dataset)

        if isinstance(src_dataset, WarpedVRT):
            src_dataset.close()
        real_src.close()

        if has_footprint_tag:
            transformer = pyproj.Transformer.from_crs(
                pyproj.CRS('EPSG:4326'), dataset.crs, always_xy=True
            ).transform
            expected = shapely.ops.transform(
                transformer, shapely.wkt.loads(footprint_wkt)
            )
            assert isinstance(result, Polygon)
            assert result.equals_exact(expected, tolerance=1e-9)
        else:
            assert result.equals_exact(shapely.box(*bounds), tolerance=1e-9)  # type: ignore[arg-type]

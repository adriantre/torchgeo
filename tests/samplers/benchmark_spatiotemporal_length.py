# TODO: probably not the correct place to put this
import contextlib
import io
import time
from collections.abc import Iterator

import shapely
from geopandas import GeoDataFrame
from pandas import IntervalIndex, Timestamp
from pyproj import CRS

from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import GeoSlice, Sample
from torchgeo.samplers import SpatialSampler, SpatioTemporalSampler, TemporalSampler


class FakeDataset(GeoDataset):
    def __init__(self) -> None:
        intervals = [
            (Timestamp(2025, 4, 1), Timestamp(2025, 4, 2)),
            (Timestamp(2025, 4, 15), Timestamp(2025, 4, 16)),
            (Timestamp(2025, 4, 29), Timestamp(2025, 4, 30)),
        ]
        index = IntervalIndex.from_tuples(intervals, closed='both', name='datetime')
        geometry = [
            shapely.box(0, 0, 100, 100),
            shapely.box(0, 0, 10, 10),
            shapely.box(90, 90, 100, 100),
        ]
        self.index = GeoDataFrame(
            index=index, geometry=geometry, crs=CRS.from_epsg(3005)
        )
        self.res = 2

    def __getitem__(self, index: GeoSlice) -> Sample:
        return {'bounds': self._slice_to_tensor(index)}


class FakeSpatial(SpatialSampler):
    strategy = 'random'  # will be overridden in bench

    def __init__(self, dataset: GeoDataset, length: int) -> None:
        super().__init__(dataset)
        self._length = length

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        # All yields land inside the big 100x100 geometry so _iter_subset(loc)
        # always finds intervals — keeps the (sequential, random) case happy.
        for _ in range(self._length):
            yield slice(5, 5), slice(5, 5)


class FakeTemporal(TemporalSampler):
    strategy = 'random'  # flip to 'sequential' in the loop below

    def _iter_subset(
        self, location: tuple[slice, slice] = (slice(None), slice(None))
    ) -> Iterator[tuple[slice, slice, slice]]:
        intervals = self._init_subset(location)
        intervals = intervals.to_series().sample(frac=1, random_state=0)
        x, y = location
        for interval in intervals:
            yield x, y, slice(interval.left, interval.right)


def bench(
    spatial_strategy: str, temporal_strategy: str, N: int
) -> tuple[float, float, int]:
    with contextlib.redirect_stdout(io.StringIO()):  # silence print() from geo.py
        ds = FakeDataset()

        t0 = time.perf_counter()

        spatial = FakeSpatial(ds, length=N)
        spatial.strategy = spatial_strategy

        temporal = FakeTemporal(ds)
        temporal.strategy = temporal_strategy

        sampler = SpatioTemporalSampler(spatial, temporal)
        creation = time.perf_counter() - t0

        t1 = time.perf_counter()

        length = len(sampler)
        len_call = time.perf_counter() - t1

        return creation, len_call, length


if __name__ == '__main__':
    import logging
    import warnings

    warnings.filterwarnings('ignore', message='random_sampler @ sequential_sampler')
    logging.getLogger('torchgeo').setLevel(logging.WARNING)

    print(
        f'{"strategy":25s} {"N_spatial":>10s} {"creation (ms)":>15s} {"first len() (ms)":>20s} {"len value":>12s}'
    )
    print('-' * 90)
    for combo in [
        ('random', 'random'),
        ('sequential', 'random'),
        ('sequential', 'sequential'),
        ('random', 'sequential'),
    ]:
        for N in [100, 10_000]:  # , 1_000_000]:
            creation, len_call, length = bench(*combo, N)
            print(
                f'{combo[0] + "+" + combo[1]:25s} {N:>10d} {creation * 1000:>15.3f} {len_call * 1000:>20.3f} {length:>12d}'
            )

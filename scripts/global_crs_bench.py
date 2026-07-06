#!/usr/bin/env python3
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Benchmark sampling a global dataset whose neighbouring tiles use different CRSs.

TorchGeo's :class:`~torchgeo.datasets.IOBench` covers a single Landsat scene in one UTM
zone. This script instead exercises the *multi-CRS* regime: it generates a synthetic
global dataset of overlapping tile pairs, each pair straddling a UTM zone boundary so the
two tiles are authored in *different* native CRSs (chosen by
``GeoSeries.estimate_utm_crs()``), then times sampling.

On ``main`` a :class:`~torchgeo.datasets.RasterDataset` can only combine tiles from
different CRSs by warping every one to a single global CRS at read time, so this measures
that baseline: all tiles warped to EPSG:6933 (EASE-Grid 2.0, equal-area, so area-weighted
random sampling stays uniform). It is the yardstick for PRs that add native-CRS reads —
rerun it on such a branch to show the speed-up.

The dataset is generated once under ``--out`` (pixels are random; only geometry/CRS
matter for warp timing) and reused on subsequent runs.

Example::

    python scripts/global_crs_bench.py --out data/global_crs
"""

import argparse
import contextlib
import math
import os
import time
from collections.abc import Callable, Iterator
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.merge
import rasterio.vrt
import rasterio.warp
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomPatchSampler

BANDS = 3

# scenario -> (lon, lat, axis): the two tiles are centered on (lon, lat) and offset along
# `axis` by (1 - overlap) * tile_width, so --overlap sets how much they overlap (hence the
# fraction of patches that span two CRSs). Apart from the controls, the pair straddles a
# zone boundary so estimate_utm_crs() splits them into different UTM zones.
SCENARIOS: dict[str, tuple[float, float, str]] = {
    'adjacent': (6.0, 48.0, 'lon'),  # zone boundary 6 deg E: UTM zones 31 / 32
    'same_zone': (9.0, 48.0, 'lon'),  # control: mid zone 32, one native zone (!= index)
    'in_index_crs': (6.0, 48.0, 'lon'),  # control: authored in the index CRS
    'equator': (33.0, 0.0, 'lat'),  # equator: N / S hemisphere (326xx / 327xx)
    'high_lat': (
        24.0,
        70.0,
        'lon',
    ),  # zone boundary 24 deg E at 70 deg N: zones 34 / 35
    'antimeridian': (180.0, 20.0, 'lon'),  # +/-180: zones 60 / 1
}
# Tiles for these scenarios are authored directly in the index CRS, so neither the warp
# baseline nor native reads reproject them: the zero-warp floor for read + merge overhead.
# Their realized overlap is approximate (the offset is in ground degrees but the tiles span
# index-CRS meters), which is immaterial here since overlap doesn't affect a zero-warp read.
INDEX_SCENARIOS = {'in_index_crs'}
# Antimeridian is excluded by default: its tiles land on opposite edges of a global CRS,
# a degenerate case for a single shared grid. Opt in with ``--scenarios antimeridian``.
DEFAULT_SCENARIOS = ['adjacent', 'same_zone', 'in_index_crs', 'equator', 'high_lat']


class _GlobalCRSBench(RasterDataset):
    """Reads the synthetic tiles; a single ``crs`` warps every tile onto one grid."""

    filename_glob = '*.tif'
    is_image = True


class Result(NamedTuple):
    """Timing result for one benchmarked scenario."""

    scenario: str
    native: str  # comma-separated native CRSs of the scenario's tiles
    samples: int
    batches: int
    elapsed: float
    warps: float  # warped windowed reads per sample
    reprojects: float  # explicit rasterio.warp.reproject calls per sample


def _utm_for(lon: float, lat: float) -> CRS:
    """Return the UTM CRS ``estimate_utm_crs()`` picks for a point."""
    return gpd.GeoSeries([Point(lon, lat)], crs='EPSG:4326').estimate_utm_crs()


def _scenario_centers(
    name: str, size: int, res: float, overlap: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two tile centers for a scenario at a given overlap fraction.

    The tiles are offset from the scenario center along its axis by
    ``(1 - overlap) * tile_width``, split evenly either side, so ``overlap`` is the
    fraction of tile width the two tiles share.

    Args:
        name: Scenario name.
        size: Tile size in pixels.
        res: Resolution in meters.
        overlap: Fraction of tile width the two tiles overlap, in [0, 1).

    Returns:
        The ``(lon, lat)`` centers of tiles A and B.
    """
    lon, lat, axis = SCENARIOS[name]
    half_offset_m = (1.0 - overlap) * size * res / 2
    if axis == 'lon':
        deg = half_offset_m / (111320 * math.cos(math.radians(lat)))
        return ((lon - deg + 540) % 360 - 180, lat), (
            (lon + deg + 540) % 360 - 180,
            lat,
        )
    deg = half_offset_m / 110540  # meters per degree of latitude
    return (lon, lat - deg), (lon, lat + deg)


def _scenario_dir(out: str, name: str, size: int, overlap: float) -> str:
    """Directory for a scenario's tiles, keyed by the params that set their geometry.

    Keying on *size* and *overlap* keeps distinct configurations from reusing each
    other's cached tiles when sweeping.
    """
    return os.path.join(out, f'{name}_s{size}_o{round(overlap * 100):03d}')


def _write_tile(
    path: str,
    lon: float,
    lat: float,
    size: int,
    res: float,
    seed: int,
    crs: CRS | None = None,
) -> tuple[CRS, Polygon]:
    """Write one tile authored in ``crs`` (no pixel reprojection).

    Args:
        path: Output GeoTIFF path.
        lon: Center longitude in degrees.
        lat: Center latitude in degrees.
        size: Tile width/height in pixels.
        res: Resolution in meters.
        seed: Seed for the random pixel values.
        crs: CRS to author the tile in (defaults to the center's UTM zone).

    Returns:
        The tile's CRS and its footprint as an EPSG:4326 polygon.
    """
    crs = crs or _utm_for(lon, lat)
    cx, cy = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
    half = size * res / 2
    # Snap the origin to the res grid, as real tiled products are.
    left = (cx - half) // res * res
    top = (cy + half) // res * res
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'count': BANDS,
        'width': size,
        'height': size,
        'crs': crs,
        'transform': from_origin(left, top, res, res),
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }
    rng = np.random.default_rng(seed)
    with rasterio.open(path, 'w', **profile) as dst:
        for band in range(1, BANDS + 1):
            dst.write(rng.integers(0, 65535, (size, size), dtype='uint16'), band)

    inv = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    corners_x = [cx - half, cx + half, cx + half, cx - half]
    corners_y = [cy - half, cy - half, cy + half, cy + half]
    lons, lats = inv.transform(corners_x, corners_y)
    return crs, Polygon(zip(lons, lats))


def generate(
    out: str,
    size: int,
    res: float,
    scenarios: list[str],
    index_crs: int,
    overlap: float,
) -> None:
    """Generate (if missing) the tiles for each scenario.

    Args:
        out: Root output directory.
        size: Tile size in pixels.
        res: Resolution in meters.
        scenarios: Scenario names to generate.
        index_crs: EPSG code used to author :data:`INDEX_SCENARIOS` tiles directly.
        overlap: Fraction of tile width the two tiles overlap, in [0, 1).
    """
    for name in scenarios:
        scenario_dir = _scenario_dir(out, name, size, overlap)
        os.makedirs(scenario_dir, exist_ok=True)
        paths = {
            label: os.path.join(scenario_dir, f'tile_{label}.tif') for label in 'AB'
        }
        if all(os.path.exists(p) for p in paths.values()):
            continue

        # Author the control scenarios in the index CRS; the rest in their UTM zone.
        tile_crs = CRS.from_epsg(index_crs) if name in INDEX_SCENARIOS else None
        # Seed off the canonical scenario order so a scenario's pixels are identical
        # regardless of which subset/order is passed via --scenarios.
        base_seed = 2 * list(SCENARIOS).index(name)
        centers = _scenario_centers(name, size, res, overlap)
        crss, feet = {}, {}
        for j, (label, center) in enumerate(zip('AB', centers)):
            crss[label], feet[label] = _write_tile(
                paths[label], *center, size, res, seed=base_seed + j, crs=tile_crs
            )
        intersects = feet['A'].intersects(feet['B'])
        different = crss['A'] != crss['B']
        print(
            f'{name:<13} {crss["A"].to_epsg()} / {crss["B"].to_epsg()}  '
            f'overlap={intersects} different_crs={different}'
        )


@contextlib.contextmanager
def _count_reprojections() -> Iterator[dict[str, int]]:
    """Count GDAL reprojection work performed while the block runs.

    Patches ``WarpedVRT.read`` (each call warps one windowed read of an off-CRS source —
    counted per read, so it survives torchgeo's VRT caching) and ``rasterio.warp.reproject``
    (explicit reprojection, e.g. inside ``merge``). Native reads open plain datasets, so
    they register no warped reads. Main wraps every source in a (possibly identity)
    ``WarpedVRT``, so it counts a read per matched source regardless of cost.
    """
    counts = {'warps': 0, 'reprojects': 0}
    vrt_read = rasterio.vrt.WarpedVRT.read

    def counted_read(
        self: rasterio.vrt.WarpedVRT, *args: object, **kwargs: object
    ) -> object:
        counts['warps'] += 1
        return vrt_read(self, *args, **kwargs)

    def counted_reproject(orig: Callable[..., object]) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            counts['reprojects'] += 1
            return orig(*args, **kwargs)

        return wrapper

    # reproject is imported by-name into several modules (e.g. rasterio.merge), so patch
    # every binding — patching only rasterio.warp would miss merge's own copy.
    modules = [m for m in (rasterio.warp, rasterio.merge) if hasattr(m, 'reproject')]
    originals = {m: m.reproject for m in modules}

    rasterio.vrt.WarpedVRT.read = counted_read  # type: ignore[method-assign]
    for m in modules:
        m.reproject = counted_reproject(originals[m])  # type: ignore[assignment]
    try:
        yield counts
    finally:
        rasterio.vrt.WarpedVRT.read = vrt_read  # type: ignore[method-assign]
        for m in modules:
            m.reproject = originals[m]  # type: ignore[assignment]


def benchmark(
    name: str,
    scenario_dir: str,
    crs: CRS,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    length: int,
    seed: int,
    warmup: int,
    repeats: int,
) -> Result:
    """Time random-patch sampling over a scenario, warped to ``crs``.

    Args:
        name: Scenario name (row label).
        scenario_dir: Directory holding the scenario's tiles.
        crs: :term:`coordinate reference system (CRS)` to warp every tile to.
        patch_size: Size of each square patch in pixels.
        batch_size: Number of patches per mini-batch.
        num_workers: Number of dataloader worker processes.
        length: Number of random patches to sample.
        seed: Seed for the sampler (for reproducible before/after comparison).
        warmup: Untimed passes to run before timing.
        repeats: Timed passes; the fastest is reported.

    Returns:
        The scenario's timing result. ``samples`` is 0 when the patch is larger than
        the warped tile (``RandomPatchSampler`` erodes the ROI by ~sqrt(2)/2 * patch to
        keep patches in bounds, which can empty a small tile at high latitude in an
        equal-area CRS) — raise ``--size`` in that case.
    """
    dataset = _GlobalCRSBench(scenario_dir, crs=crs)
    native_crss = {rasterio.open(f).crs.to_epsg() for f in dataset.files}
    sampler = RandomPatchSampler(
        dataset, size=patch_size, length=length, generator=seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=stack_samples,
    )

    def one_pass() -> tuple[int, int, float]:
        batches = samples = 0
        start = time.perf_counter()
        for batch in tqdm(dataloader, total=len(dataloader), desc=name, leave=False):
            batches += 1
            samples += batch['image'].shape[0]
        return batches, samples, time.perf_counter() - start

    # One instrumented, untimed pass counts reprojections (and doubles as a warmup); the
    # monkeypatch is off before timing so it adds no overhead there.
    with _count_reprojections() as counts:
        _, count_samples, _ = one_pass()
    warps = counts['warps'] / count_samples if count_samples else 0.0
    reprojects = counts['reprojects'] / count_samples if count_samples else 0.0

    # Warmup passes pay one-time costs (PROJ/GDAL init, OS page cache) so timing is
    # order-invariant; reporting the fastest of several repeats sheds random scheduler/
    # thermal noise. Both are needed to resolve the ~10% native-vs-warp delta.
    for _ in range(warmup):
        one_pass()
    num_batches, num_samples, elapsed = min(
        (one_pass() for _ in range(max(1, repeats))), key=lambda r: r[2]
    )

    native = ', '.join(f'EPSG:{c}' for c in sorted(native_crss))
    return Result(name, native, num_samples, num_batches, elapsed, warps, reprojects)


def print_matrix(results: list[Result], index_crs: int, patch_size: int) -> None:
    """Print all scenario results as one aligned matrix.

    Args:
        results: One :class:`Result` per benchmarked scenario.
        index_crs: EPSG code every tile was warped to (shared by all rows).
        patch_size: Patch size in pixels (for the empty-tile footnote).
    """
    headers = (
        'scenario',
        'native CRSs',
        'samples',
        'batches',
        's',
        'samples/s',
        'batches/s',
        'warps/smp',
        'reproj/smp',
    )
    aligns = '<<>>>>>>>'
    rows = [
        (
            r.scenario,
            r.native,
            str(r.samples),
            str(r.batches),
            f'{r.elapsed:.3f}',
            f'{r.samples / r.elapsed:.1f}',
            f'{r.batches / r.elapsed:.2f}',
            f'{r.warps:.2f}',
            f'{r.reprojects:.2f}',
        )
        if r.samples
        else (r.scenario, r.native, '-', '-', '-', '-', '-', '-', '-')
        for r in results
    ]
    widths = [
        max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)
    ]

    def row(cells: tuple[str, ...]) -> str:
        padded = (f'{c:{a}{w}}' for c, a, w in zip(cells, aligns, widths))
        return '| ' + ' | '.join(padded) + ' |'

    def sep() -> str:
        # Markdown alignment markers: ---: right-aligns, :--- left-aligns.
        dashes = (
            '-' * (w - 1) + ':' if a == '>' else ':' + '-' * (w - 1)
            for a, w in zip(aligns, widths)
        )
        return '| ' + ' | '.join(dashes) + ' |'

    # GitHub-flavored markdown so the table pastes straight into a PR.
    print(f'\nWarped to EPSG:{index_crs} (samples/s, higher is better):\n')
    print(row(headers))
    print(sep())
    for cells in rows:
        print(row(cells))
    if any(not r.samples for r in results):
        print(
            f'\n_`-` = patch ({patch_size} px) exceeds the warped tile; raise --size._'
        )


def main() -> None:
    """Parse CLI args, generate the dataset if needed, and benchmark each scenario."""
    parser = argparse.ArgumentParser(
        description='Benchmark sampling a global multi-CRS dataset.'
    )
    parser.add_argument(
        '--out', default='data/global_crs', help='directory to generate/read tiles in'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=2048,
        help='tile size in pixels (real-sized runs use ~3850, a quarter Landsat scene)',
    )
    parser.add_argument('--res', type=float, default=30.0, help='resolution in meters')
    parser.add_argument(
        '--crs',
        type=int,
        default=6933,
        help='EPSG code every tile is warped to (the shared global grid)',
    )
    parser.add_argument(
        '--patch-size', type=int, default=256, help='patch size in pixels'
    )
    parser.add_argument('--batch-size', type=int, default=32, help='mini-batch size')
    parser.add_argument(
        '--num-workers', type=int, default=0, help='dataloader worker processes'
    )
    parser.add_argument(
        '--length', type=int, default=256, help='number of random patches per scenario'
    )
    parser.add_argument('--seed', type=int, default=0, help='sampler seed')
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='fraction of tile width the two tiles overlap, in [0, 1)'
        ' (lower = more single-CRS patches = larger native win)',
    )
    parser.add_argument(
        '--warmup', type=int, default=1, help='untimed passes before timing'
    )
    parser.add_argument(
        '--repeats', type=int, default=3, help='timed passes; the fastest is reported'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=list(SCENARIOS),
        default=DEFAULT_SCENARIOS,
        help='which scenarios to benchmark',
    )
    args = parser.parse_args()
    if not 0.0 <= args.overlap < 1.0:
        parser.error('--overlap must be in [0, 1)')

    print('Global multi-CRS benchmark')
    print(
        f'out={args.out} size={args.size} crs=EPSG:{args.crs}'
        f' patch_size={args.patch_size} batch_size={args.batch_size}'
        f' num_workers={args.num_workers} length={args.length} overlap={args.overlap}'
    )

    generate(args.out, args.size, args.res, args.scenarios, args.crs, args.overlap)

    crs = CRS.from_epsg(args.crs)
    results = [
        benchmark(
            name,
            _scenario_dir(args.out, name, args.size, args.overlap),
            crs=crs,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            length=args.length,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for name in args.scenarios
    ]
    print_matrix(results, args.crs, args.patch_size)


if __name__ == '__main__':
    main()

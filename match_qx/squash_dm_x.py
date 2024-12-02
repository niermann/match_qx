#!/usr/bin/env python3
import numpy as np
from pathlib import Path

from pyctem.iolib import load_dm, TemDataFile
from pyctem.hl import LinearAxis, DataSet, CoreMetaData, show_position_stem
from pyctem import TemMeasurementMetaData, TemInstrumentMetaData, CameraDetectorMetaData

COPYRIGHT = """
Copyright (c) 2024 Tore Niermann

This program comes with ABSOLUTELY NO WARRANTY;

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See file 'COPYING' for details.
"""

LINESCAN3D_VERSION = 4.0


def main(path, show_result=False, dryrun=False, dtype=None):
    metadata = CoreMetaData()
    instrument_metadata = None
    data = None
    image = None

    data = load_dm(path, memmap=True)
    image_scale = getattr(data.axes[0], "scale", 1.0)
    image_unit = getattr(data.axes[0], "unit", "px")

    metadata['source'] = data.metadata.ref()
    metadata['instrument'] = TemInstrumentMetaData(getattr(data.metadata, "instrument", None))
    metadata['detector'] = CameraDetectorMetaData(getattr(data.metadata, "detector", None))

    data_view = data.get(copy=False)

    image = DataSet(data=np.sum(data_view, axis=(-2, -1), dtype=np.float32),
                    axes=data.axes[0:2], metadata=metadata.copy())

    pos0 = np.array((data.shape[1] * 0.5, 0), dtype=float)
    pos1 = np.array((data.shape[1] * 0.5, data.shape[0]), dtype=float)
    posw = data.shape[1] + 0.001
    binsize = 1.0

    print("Creating linescan:")
    print(f'\t"pos0": [{pos0[0]:.1f}, {pos0[1]:.1f}],')
    print(f'\t"pos1": [{pos1[0]:.1f}, {pos1[1]:.1f}],')
    print(f'\t"posw": {posw:.1f},')
    print(f'\t"pos_binsize": {binsize:.1f},')

    scan_axes = (
        LinearAxis(name='x', context='POSITION', unit=image_unit, scale=binsize * image_scale),
        data.axes[-2],
        data.axes[-1]
    )

    metadata['linescan3d_version'] = LINESCAN3D_VERSION
    metadata['pos0'] = pos0
    metadata['pos1'] = pos1
    metadata['posw'] = posw
    metadata['pos_binsize'] = binsize
    metadata['pos_scale'] = image_scale
    metadata['pos_unit'] = image_unit

    if dtype is None:
        dtype = np.float32
    pos_mean = DataSet(data=np.mean(data_view, axis=1, dtype=dtype), axes=scan_axes, metadata=metadata)
    #pos_var = DataSet(data=np.var(data_view, axis=1, dtype=np.float32, ddof=1), axes=scan_axes, metadata=metadata)
    #pos_count = DataSet(data=np.full(data_view.shape[0:1] + data_view.shape[2:], data.shape[1]), axes=scan_axes, metadata=metadata)

    if not dryrun:
        linescan_path = path.stem + ".squash_x.tdf"
        with TemDataFile(linescan_path, 'w') as outfile:
            outfile.write_dataset(pos_mean, name="mean")
            #outfile.write_dataset(pos_var, name="var")
            #outfile.write_dataset(pos_count, name="count")
            outfile.write_dataset(image, name="image")

    del data_view
    del data

    if show_result:
        pos_dir = (pos1 - pos0).astype(float)
        pos_dir /= np.sqrt(np.sum(pos_dir ** 2))
        pos = np.arange(pos_mean.shape[0], dtype=float)[..., np.newaxis] * binsize * pos_dir + pos0
        show_position_stem(pos_mean, pos, image.get(), title=path.stem, vmin=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create 3D linescan from 4D-STEM by average in x-direction", epilog=COPYRIGHT)
    parser.add_argument('path')
    parser.add_argument('-r', '--result', action='store_true', default=False, help="Show result")
    parser.add_argument('-n', '--dry-run', dest="dryrun", action='store_true', default=False, help="Do not save result")
    parser.add_argument('-d', '--dtype', type=str, default=None, help="Datatype of average")
    args = parser.parse_args()

    path = Path(args.path)
    main(path, show_result=args.result, dryrun=args.dryrun, dtype=args.dtype)

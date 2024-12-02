#!/usr/bin/env python3
import numpy as np
from pathlib import Path

from pyctem.iolib import load_mib, load_tiff, TemDataFile, load_tdf, Hdf5DataSet
from pyctem.hl import (show_4d_stem, LinearAxis, DataSet, CoreMetaData, show_position_stem,
                       apply_4d_stem, SumDetector, CenterOfMassDetector)
from pyctem.proc import line_scan
from pyctem.utils import decode_json
from pyctem import TemMeasurementMetaData
from tem import TemInstrumentMetaData, CameraDetectorMetaData

COPYRIGHT = """
Copyright (c) 2024 Tore Niermann

This program comes with ABSOLUTELY NO WARRANTY;

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See file 'COPYING' for details.
"""


def main(path, show_result=False, dryrun=False):
    metadata = CoreMetaData()
    instrument_metadata = None
    data = None
    image = None

    from pyctem.iolib import load_dm
    data = load_dm(merlin_path, memmap=True)
    image_scale = getattr(data.axes[1], "scale", 1.0)
    image_unit = getattr(data.axes[1], "unit", "px")

    metadata['source'] = data.metadata.ref()
    metadata['instrument'] = TemInstrumentMetaData(getattr(data.metadata, "instrument", None))
    metadata['detector'] = CameraDetectorMetaData(getattr(data.metadata, "detector", None))

    data_view = data.get(copy=False)

    image = DataSet(data=np.sum(data_view, axis=(-2, -1), dtype=np.float32),
                    axes=data.axes[0:2], metadata=metadata.copy())

    pos0 = np.array((data.shape[1] * 0.5, data.shape[0])), dtype=float)
    pos1 = np.array((data.shape[1] * 0.5, data.shape[0])), dtype=float)
    posw = data.shape[1] + 0.001
    binsize = 1.0

    print("Creating linescan:")
    print(f'\t"pos0": [{pos0[0]:.1f}, {pos0[1]:.1f}],')
    print(f'\t"pos1": [{pos1[0]:.1f}, {pos1[1]:.1f}],')
    print(f'\t"posw": {posw:.1f},')
    print(f'\t"pos_binsize": {binsize:.1f},')

    pos_scan = line_scan(data_view, pos0, pos1, posw, bin_size=binsize)
    scan_axes = (
        LinearAxis(name='x', context='POSITION', unit=image_unit, scale=binsize * image_scale,
        data.axes[-2],
        data.axes[-1]
    )

    pos_mean = DataSet(data=np.mean(data_view, axis=0, dtype=np.float32), axes=scan_axes, metadata=metadata)
    pos_var = DataSet(data=np.var(data_view, axis=0, dtype=np.float32, ddof=1), axes=scan_axes, metadata=metadata)
    pos_count = DataSet(data=np.full(pos_scan[2], axes=scan_axes, metadata=metadata)

    if not dryrun:
        with TemDataFile(linescan_path, 'w') as outfile:
            metadata['linescan3d_version'] = LINESCAN3D_VERSION
            metadata['pos0'] = pos0
            metadata['pos1'] = pos1
            metadata['posw'] = posw
            metadata['pos_binsize'] = binsize
            metadata['pos_scale'] = image_scale
            metadata['pos_unit'] = image_unit

            outfile.write_dataset(pos_mean, name="mean")
            outfile.write_dataset(pos_var, name="var")
            outfile.write_dataset(pos_count, name="count")
            outfile.write_dataset(image, name="image")

    if show_result:
        pos_dir = (pos1 - pos0).astype(float)
        pos_dir /= np.sqrt(np.sum(pos_dir ** 2))
        pos = np.arange(pos_mean.shape[0], dtype=float)[..., np.newaxis] * binsize * pos_dir + pos0
        show_position_stem(pos_mean, pos, image.get(), title=linescan_path.stem, vmin=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create 3D linescan from 4D-STEM position space",
                                     epilog=COPYRIGHT + "\n\nExample parameter file:\n" + EXAMPLE_PARAMETER_FILE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paramfile')
    parser.add_argument('-s', '--show4d', action='store_true', default=False, help="Show 4D dataset")
    parser.add_argument('-l', '--linescan', action='store_true', default=False, help="Show linescan interactively")
    parser.add_argument('-r', '--result', action='store_true', default=False, help="Show result")
    parser.add_argument('--lazy', action='store_true', default=True, help="Use lazy 4D STEM import")
    parser.add_argument('--no-lazy', action='store_false', dest="lazy", help="No not use lazy 4D STEM import")
    parser.add_argument('-n', '--dry-run', dest="dryrun", action='store_true', default=False, help="Do not save result")
    parser.add_argument('-m', '--method', type=str, default=False, help="Method to use for derivation of image")
    args = parser.parse_args()

    param_file = Path(args.paramfile)
    with open(param_file, 'rt') as file:
        param_source = file.read()
    param = decode_json(param_source, allow_comments=True, allow_trailing_commas=True)
    main(param, param_file.stem, show_4d=args.show4d, show_linescan=args.linescan, show_result=args.result,
         image_method=args.method, lazy=args.lazy, dryrun=args.dryrun)

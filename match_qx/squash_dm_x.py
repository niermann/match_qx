#!/usr/bin/env python3
import time
import numpy as np
from pathlib import Path

from pyctem.iolib import TemDataFile
from pyctem.iolib.dm import parse_image_header, DigitalMicrographFile
from pyctem.hl import LinearAxis, DataSet, CoreMetaData
from pyctem import TemInstrumentMetaData, CameraDetectorMetaData
from pyctem.utils import print_progress

COPYRIGHT = """
Copyright (c) 2024 Tore Niermann

This program comes with ABSOLUTELY NO WARRANTY;

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See file 'COPYING' for details.
"""

LINESCAN3D_VERSION = 4.0


def main(path, dtype=None, var=False):
    metadata = CoreMetaData()

    with DigitalMicrographFile(path, 'r', memmap=True) as tags:
        dm_image = tags['ImageList'][1]
        dm_image_data = dm_image['ImageData']

        header = parse_image_header(dm_image)

        # Read image
        # Bitcasting by use of view is needed for complex types, which are returned by as_array as 2 float records.
        image_array = dm_image_data['Data']
        dm_dtype = np.dtype(image_array.byteorder + header.dtype)
        data = image_array.as_array(copy=False).view(dm_dtype).reshape(header.shape)

        image_scale = getattr(header.axes[0], "scale", 1.0)
        image_unit = getattr(header.axes[0], "unit", "px")

        metadata['source'] = header.metadata.ref()
        metadata['instrument'] = TemInstrumentMetaData(getattr(header.metadata, "instrument", None))
        metadata['detector'] = CameraDetectorMetaData(getattr(header.metadata, "detector", None))

        chunk_size = (16 << 20) // header.shape[-1] // header.shape[-2] // header.shape[-3]

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
            header.axes[-2],
            header.axes[-1]
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

        linescan_path = path.stem + ".squash_x.tdf"
        with TemDataFile(linescan_path, 'w') as outfile:
            outfile.write_dataset(DataSet(data=np.full(header.shape[0:1], header.shape[1]), axes=scan_axes[0:1], metadata=metadata.copy()), name="count")

            image = outfile.create_empty_dataset(header.shape[0:2], dtype=np.float32, name="image", axes=header.axes[0:2], metadata=metadata.copy())
            pos_mean = outfile.create_empty_dataset(header.shape[0:1] + header.shape[2:], dtype=dtype, name="mean", axes=scan_axes, metadata=metadata.copy())
            if var:
                pos_var = outfile.create_empty_dataset(header.shape[0:1] + header.shape[2:], dtype=np.float32, name="var", axes=scan_axes, metadata=metadata.copy())

            print("Squashing...")
            y = 0
            start = time.time()
            while y < data.shape[0]:
                y_end = min(data.shape[0], y + chunk_size)
                print_progress(y, data.shape[0], elapsed=time.time() - start)
                data_view = data[y:y_end]
                image[y:y_end] = np.sum(data_view, axis=(-2, -1), dtype=np.float32)
                pos_mean[y:y_end] = np.mean(data_view, axis=1, dtype=dtype)
                if var:
                    pos_var[y:y_end] = np.var(data_view, axis=1, dtype=np.float32, ddof=1)
                y = y_end
            print_progress(data.shape[0], data.shape[0], elapsed=time.time() - start, newline=True)

        del data
        del data_view


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create 3D linescan from 4D-STEM by averaging a Digital Micrograph file in x-direction", epilog=COPYRIGHT)
    parser.add_argument('path')
    parser.add_argument('-d', '--dtype', type=str, default=None, help="Datatype of average")
    parser.add_argument('--no-var', action="store_false", dest='var', default=True, help="Don't calculate the variance")
    args = parser.parse_args()

    path = Path(args.path)
    main(path, dtype=args.dtype, var=args.var)

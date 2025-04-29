#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyctem.iolib import TemDataFile
from pyctem.hl import LinearAxis, DataSet, show_position_stem
from pyctem.proc import line_scan
from pyctem.utils import decode_json


COPYRIGHT = """
Copyright (c) 2024 Tore Niermann

This program comes with ABSOLUTELY NO WARRANTY;

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See file 'COPYING' for details.
"""


EXAMPLE_PARAMETER_FILE = """{
    // File with positional linescan (created by linescan_pos_3d.py) 
    "linescan3d_file": "linescan_file.tdf",

    // [qx, qy] Position of starting point of profile (in diffraction pattern pixels)
    //"diff0": [64, 128], 

    // [qx, qy] Position of ending point of profile (in diffraction pattern pixels)
    //"diff1": [192, 128],
    
    // Width of profile (in diffraction pattern pixels)
    //"diffw": 10.0,
    
    // Size of profile bin (in diffraction pattern pixels)
    //"diff_binsize": 1.0,
    
    // Name of output file (TDF format) 
    // If omitted, the output filename is concatenated from stem4d_file and the name of the parameter file.
    // Output file will contain four datasets:
    //   "mean"   2D Dataset with reciprocal and spatial linescan (averaged perpendicular to linescan in diffraction space). Dimensions [q, x] 
    //   "var"    Variance of (reciprocal) linescan. Dimensions [q, x] 
    //   "count"  Number of point contributing to (reciprocal) linescan. Dimensions [q, x] 
    //"output_file": "some-output-file.tdf"
}"""

LINESCAN2D_VERSION = 1.2


def main(param, param_file_stem, show_3d=False, show_linescan=False, show_result=False, log_diff=False, cmap="gray",
         vmax=None, dryrun=False):
    source_path = Path(param["linescan3d_file"])

    with TemDataFile(source_path, 'r') as source:
        pos_mean = source.read('mean')
        source_version = pos_mean.metadata.get("linescan3d_version", 0)
        if source_version < 2.0:
            raise ValueError("Linescan3D file invalid or outdated version (must be at least version 2.0)")
        image = source.read("image")

    output_path = Path(param.get("output_file", '.'.join((source_path.stem, param_file_stem, "tdf"))))

    diff0 = np.array(param.get('diff0', (pos_mean.shape[2] * 0.25, pos_mean.shape[1] * 0.5)), dtype=float)
    diff1 = np.array(param.get('diff1', (pos_mean.shape[2] * 0.75, pos_mean.shape[1] * 0.5)), dtype=float)
    diffw = float(param.get('diffw', 1))
    binsize = float(param.get('diff_binsize', 1))

    if show_3d:
        pos0 = np.array(pos_mean.metadata["pos0"], dtype=float)
        pos1 = np.array(pos_mean.metadata["pos1"], dtype=float)
        posw = pos_mean.metadata["posw"]
        pos_binsize = pos_mean.metadata["pos_binsize"]
        image_scale = pos_mean.metadata["pos_scale"]
        image_unit = pos_mean.metadata["pos_unit"]
        pos_dir = (pos1 - pos0)
        pos_dir /= np.sqrt(np.sum(pos_dir ** 2))
        pos = pos_mean.axis_range(0)[..., np.newaxis] * pos_dir + pos0 * image_scale
        show_position_stem(pos_mean, pos, image, title=source_path.stem, vmin=1, diff_cmap=cmap, vmax=vmax)
    pos_mean_view = pos_mean.get(copy=False)
    diff_scale = np.sqrt(getattr(pos_mean.axes[1], "scale", 1.0) * getattr(pos_mean.axes[2], "scale", 1.0))
    diff_unit = getattr(pos_mean.axes[1], "unit", "px")

    if show_linescan:
        # TODO: have some sensible import path (relative import?)
        from linescan_gui import DraggableLineScan
        sum_diff = np.sum(pos_mean_view, axis=0)
        if log_diff:
            tmp = np.empty(sum_diff.shape, dtype=float)
            mask = sum_diff > 0
            tmp[mask] = np.log(sum_diff[mask])
            tmp[~mask] = tmp[mask].min()
            sum_diff = tmp
        drag_gui = DraggableLineScan(sum_diff, diff0, diff1, diffw, title=output_path.name, cmap=cmap)
        drag_gui.run()
        diff0 = drag_gui.pos[0]
        diff1 = drag_gui.pos[1]
        diffw = drag_gui.width

    print("Creating linescan:")
    print(f'\t"diff0": [{diff0[0]:.1f}, {diff0[1]:.1f}],')
    print(f'\t"diff1": [{diff1[0]:.1f}, {diff1[1]:.1f}],')
    print(f'\t"diffw": {diffw:.1f},')
    print(f'\t"diff_binsize": {binsize:.1f},')

    diff_scan = line_scan(pos_mean_view, diff0, diff1, diffw, axes=(2, 1), bin_size=binsize)
    scan_axes = (
        LinearAxis(name='q', context='DIFFRACTION', unit=diff_unit, scale=binsize * diff_scale),
        pos_mean.axes[0]
    )

    if not dryrun:
        with TemDataFile(output_path, 'w') as outfile:
            metadata = pos_mean.metadata.copy()
            metadata.uuid = None
            metadata['linescan2d_version'] = LINESCAN2D_VERSION
            metadata['linescan3d_source'] = pos_mean.metadata.ref()
            metadata['diff0'] = diff0
            metadata['diff1'] = diff1
            metadata['diffw'] = diffw
            metadata['diff_binsize'] = binsize

            diff_mean = DataSet(data=diff_scan[0], axes=scan_axes, metadata=metadata)
            diff_var = DataSet(data=diff_scan[1], axes=scan_axes, metadata=metadata)
            diff_count = DataSet(data=diff_scan[2], axes=scan_axes, metadata=metadata)

            outfile.write_dataset(diff_mean, name="mean")
            outfile.write_dataset(diff_var, name="var")
            outfile.write_dataset(diff_count, name="count")

    # Show result and save as SVG image
    fig, ax = plt.subplots()
    pos_scale = getattr(scan_axes[1], "scale", 1.0)
    pos_offset = getattr(scan_axes[1], "offset", 0.0)
    diff_scale = getattr(scan_axes[0], "scale", 1.0)
    diff_offset = getattr(scan_axes[0], "offset", 0.0)
    extent = [-0.5 * pos_scale + pos_offset, (diff_scan[0].shape[1] - 0.5) * pos_scale + pos_offset,
              (diff_scan[0].shape[0] - 0.5) * diff_scale + diff_offset, -0.5 * diff_scale + diff_offset]
    aspect = abs(extent[1] - extent[0]) / abs(extent[3] - extent[2])
    ax.imshow(diff_scan[0], extent=extent, aspect=aspect, cmap=cmap, vmax=vmax)
    ax.set_xlabel(f"{scan_axes[1].name} ({getattr(scan_axes[1], 'unit', '')})")
    ax.set_ylabel(f"{scan_axes[0].name} ({getattr(scan_axes[0], 'unit', '')})")
    ax.set_title(output_path.stem)

    if not dryrun:
        fig.savefig(output_path.with_suffix(".svg"))
    if show_result:
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create qx-plane from 4D-STEM position space linescan",
                                     epilog=COPYRIGHT + "\n\nExample parameter file:\n" + EXAMPLE_PARAMETER_FILE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paramfile')
    parser.add_argument('-s', '--show3d', action='store_true', default=False, help="Show 3D dataset")
    parser.add_argument('-l', '--linescan', action='store_true', default=False, help="Show linescan interactively")
    parser.add_argument('-r', '--result', action='store_true', default=False, help="Show result")
    parser.add_argument('-c', '--cmap', type=str, default="gray", help="Colormap")
    parser.add_argument('--vmax', type=float, default=None, help="Max. intensity for diffraction patterns")
    parser.add_argument('--linear', action='store_true', default=False,
                        help="Show data linear instead of logarithmically")
    parser.add_argument('-n', '--dry-run', dest="dryrun", action='store_true', default=False, help="Do not save result")
    args = parser.parse_args()

    param_file = Path(args.paramfile)
    with open(param_file, 'rt') as file:
        param_source = file.read()
    param = decode_json(param_source, allow_comments=True, allow_trailing_commas=True)
    main(param, param_file.stem, show_3d=args.show3d, show_linescan=args.linescan, show_result=args.result,
         log_diff=not args.linear, cmap=args.cmap, vmax=args.vmax, dryrun=args.dryrun)

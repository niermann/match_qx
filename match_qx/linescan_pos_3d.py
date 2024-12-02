#!/usr/bin/env python3
import numpy as np
from pathlib import Path

from pyctem.iolib import load_mib, load_tiff, TemDataFile, load_tdf, Hdf5DataSet
from pyctem.hl import (show_4d_stem, LinearAxis, DataSet, CoreMetaData, show_position_stem,
                       apply_4d_stem, SumDetector, CenterOfMassDetector)
from pyctem.proc import line_scan
from pyctem.utils import decode_json
from pyctem import TemMeasurementMetaData, TemInstrumentMetaData, CameraDetectorMetaData

COPYRIGHT = """
Copyright (c) 2024 Tore Niermann

This program comes with ABSOLUTELY NO WARRANTY;

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See file 'COPYING' for details.
"""

EXAMPLE_PARAMETER_FILE = """{
    // 4D-STEM data file
    // Supported file types are:
    //  *.mib       Original Merlin Data file, requires valid image in 'image_file' or 'image_area_size' parameter.
    //  *.tdf       TEM Data File (PyCTEM file format)
    //  *.hdf5/h5   HDF5 File (if more than a single dataset is within the file, parameter 'stem4d_dataset' is required.
    //              Also .h5 files create by NION Swift are supported
    "stem4d_file": "raw_data_aluminum.hdf5",
    
    // Name of 4D-STEM dataset within file 'stem4d_file' (optional).
    // Can be omitted if only a single dataset is present in the file. Otherwise, defaults to 'merlin.data'.
    //"stem4d_dataset": "merlin.data",
    
    // Image file (for additional metadata and interactive mode; optional)
    // A correlative signal to 4D-STEM measurement (e.g. HAADF signal)
    // Supported file types are:
    //  *.tiff      TIFF data (tested with JEOL scan data files)
    //  *.tdf       TEM Data File (PyCTEM file format), requires parameter 'image_dataset' to identify dataset.
    //"image_file": "some_image.tiff",
     
    // Name of image dataset within file 'image_file' (mandatory for some file types).
    //"image_dataset": "some_image_name",
    
    // Optional sampling (nm/px) for scan dimensions. Otherwise, taken from 4D-STEM or image metadata. 
    //"image_scale": 0.12345,

    // Optional sampling (nm^-1 / px) for diffraction dimensions. Otherwise, taken from 4D-STEM or image metadata. 
    //"diff_scale": 0.54321,
    
    // [x, y] Size of scan dimensions in pixels. Ignored if non MIB-file 4D-STEM data, otherwise mandatory.
    //"image_area_size": [256, 256],
     
    // [x, y] Position of starting point of profile (in pixels)
    //"pos0": [64, 128], 

    // [x, y] Position of ending point of profile (in pixels)
    //"pos1": [192, 128],
    
    // Width of profile (in pixels)
    //"posw": 10.0,
    
    // Size of profile bin (in pixels)
    //"pos_binsize": 1.0,
    
    // Name of output file (TDF format) 
    // If omitted, the output filename is concatenated from stem4d_file and the name of the parameter file.
    // Output file will contain four datasets:
    //   "image"  Scan-Image used for linescan creation
    //   "mean"   3D Dataset with spatial linescan (averaged perpendicular to linescan). Dimensions [x, qy, qx] 
    //   "var"    Variance of spatial linescan. Dimensions [x, qy, qx] 
    //   "count"  Number of point contributing to spatial linescan. Dimensions [x, qy, qx] 
    //"output_file": "some-output-file.tdf"
    
    // Calibrations for diffraction patterns (in 1/nm / px)
    // First table indexed with acceleration voltage in kV, nested table by camera length index
    //"camera_length_calibration": {
    //   "300": { "13": 0.0265384, "12": 0.0339306 }
    //},
    
    // Optional correction factor for calibration (is multiplied to value from table above) 
    //"calibration_correction": 1.0,
}"""

LINESCAN3D_VERSION = 4.0


def parse_nion_metadata(nion_meta):
    from json import loads
    nion_meta = loads(nion_meta)

    metadata = TemMeasurementMetaData()
    metadata.uuid = nion_meta["uuid"]
    metadata.timestamp = nion_meta["created"]

    try:
        metadata["unit"] = nion_meta["intensity_calibration"]["units"]
    except KeyError:
        pass

    source = nion_meta["metadata"]["hardware_source"]
    metadata.instrument.acceleration_voltage_kV = source["high_tension"] * 1e-3

    axes = []
    for a in metadata["dimensional_calibration"]:
        axes.append(LinearAxis(offset=a.get("offset", 0.0), scale=a.get("scale", 1.0), unit=a.get("unit", "px")))

    return axes, metadata


def get_diff_axes(param, voltage, detect_angle):
    diff_scale = param["camera_length_calibration"][str(int(voltage))][str(int(detect_angle))] * \
                 param.get("calibration_correction", 1.0)
    diff_axes = (LinearAxis(name='qy', context='DIFFRACTION', unit='1/nm', scale=diff_scale),
                 LinearAxis(name='qx', context='DIFFRACTION', unit='1/nm', scale=diff_scale))
    return diff_axes


def main(param, param_file_stem, show_4d=False, show_linescan=False, show_result=False, image_method=None, lazy=True,
         dryrun=False):
    merlin_path = Path(param["stem4d_file"])

    try:
        image_path = Path(param["image_file"])
    except KeyError:
        image_path = None

    metadata = CoreMetaData()
    instrument_metadata = None
    data = None
    image = None

    if merlin_path.suffix == '.tdf':
        merlin_dataset = param.get("stem4d_dataset")
        tdf_file = TemDataFile(merlin_path, "r")
        if not merlin_dataset:
            uuids = tdf_file.uuids(cls=TemDataFile.CLASS_DATASET)
            if len(uuids) == 1:
                merlin_dataset = uuids[0]
            elif tdf_file.contains("merlin.data"):
                merlin_dataset = "merlin.data"
            else:
                raise ValueError("No 'stem4d_dataset' parameter given.")
        data = tdf_file.read(merlin_dataset, cls=TemDataFile.CLASS_DATASET, lazy=lazy)

        image_dataset = param.get("image_dataset")
        if (not image_path or image_path == merlin_path) and image_dataset:
            image = tdf_file.read(image_dataset, cls=TemDataFile.CLASS_DATASET)

    if (image is None) and image_path:
        if image_path.suffix == '.tdf':
            image_dataset = param.get("image_dataset")
            image = load_tdf(image_path, name=image_dataset)
        elif image_path.suffix in ['.tiff', '.tif']:
            image = load_tiff(image_path)
        elif image_path.suffix == '.ndata':
            with np.lib.npyio.NpzFile(image_path) as file:
                image = file["data"]
                nion_meta = file["metadata.json"]
            axes, metadata = parse_nion_metadata(nion_meta)
            metadata.filename = image_path
            image = DataSet(data=image, axes=axes, metadata=metadata)
        else:
            raise ValueError(f"Unknown file format: {image_path}")

    try:
        image_scale = float(param["image_scale"])
    except (KeyError, ValueError, TypeError):
        image_scale = None
    if image_scale:
        image_axes = (LinearAxis(name="y", context="POSITION", unit="nm", scale=image_scale),
                      LinearAxis(name="x", context="POSITION", unit="nm", scale=image_scale))
    elif data is not None:
        image_axes = data.axes[0:2]
    else:
        image_axes = (LinearAxis(name="y", context="POSITION", unit="px"),
                      LinearAxis(name="x", context="POSITION", unit="px"))

    try:
        diff_scale = float(param["diff_scale"])
    except (KeyError, ValueError, TypeError):
        diff_scale = None
    if diff_scale:
        diff_axes = (LinearAxis(name="qy", context="DIFFRACTION", unit="1/nm", scale=diff_scale),
                     LinearAxis(name="qx", context="DIFFRACTION", unit="1/nm", scale=diff_scale))
    elif data is not None:
        diff_axes = data.axes[2:4]
    else:
        diff_axes = (LinearAxis(name='qy', context='DIFFRACTION', unit='px'),
                     LinearAxis(name='qx', context='DIFFRACTION', unit='px'))

    if data is not None:
        data.axes = image_axes + diff_axes

    if merlin_path.suffix == '.mib':
        assert data is None
        if image is not None:
            image_area_size = param.get("image_area_size", (image.shape[1], image.shape[0]))
            image_shape = (image_area_size[1], image_area_size[0])

            if not image_scale:
                if image_shape != image.shape:
                    image_axes = LinearAxis(image.axes[0], size=0), LinearAxis(image.axes[1], size=0)
                    image = None
                else:
                    image_axes = image.axes

            if not diff_scale:
                try:
                    voltage = int(image.metadata.instrument.acceleration_voltage_kV)
                    detect_angle = int(image.metadata.instrument.camera_length_index)
                    diff_axes = get_diff_axes(param, voltage, detect_angle)
                except (KeyError, AttributeError, ValueError, TypeError):
                    pass

            data = load_mib(merlin_path, memmap=lazy, index_shape=image_shape, axes=image_axes + diff_axes)
            instrument_metadata = getattr(image.metadata, "instrument", None)
        elif "image_area_size" in param:
            image_area_size = param["image_area_size"]
            image_shape = (image_area_size[1], image_area_size[0])
            data = load_mib(merlin_path, memmap=lazy, index_shape=image_shape, axes=image_axes + diff_axes)
        else:
            raise ValueError("'image_area_size' must be set, when 'image_file' is omitted.")
    elif merlin_path.suffix in ['.hdf5', '.h5']:
        assert data is None

        import h5py
        hdf5_file = h5py.File(merlin_path, "r")
        merlin_dataset = param.get("stem4d_dataset")
        if not merlin_dataset:
            names = list(hdf5_file.keys())
            if len(names) != 1 or not isinstance(hdf5_file[names[0]], h5py.Dataset):
                raise ValueError("No 'stem4d_dataset' parameter given.")
            merlin_dataset = names[0]

        hdf5_data = hdf5_file[merlin_dataset]
        try:
            nion_meta = hdf5_data.attrs["properties"]
            axes, metadata = parse_nion_metadata(nion_meta)
            metadata.filename = image_path
            if not image_scale:
                image_axes = axes[0:2]
            if not diff_scale:
                diff_axes = axes[2:4]
        except KeyError:
            metadata = TemMeasurementMetaData(filename=merlin_path, name=merlin_dataset)
        if lazy:
            data = Hdf5DataSet(hdf5_data, axes=image_axes + diff_axes, metadata=metadata)
        else:
            data = DataSet(data=hdf5_data[...], axes=image_axes + diff_axes, metadata=metadata)
            del hdf5_data
            hdf5_file.close()
            del hdf5_file
    elif merlin_path.suffix in ['.dm4', '.dm3']:
        assert data is None
        from pyctem.iolib import load_dm
        data = load_dm(merlin_path, memmap=True)
    else:
        raise ValueError("Not a supported 4DSTEM data file format")

    metadata['source'] = data.metadata.ref()
    if image is not None:
        metadata['image'] = image.metadata.ref()
    if instrument_metadata:
        metadata['instrument'] = TemInstrumentMetaData(instrument_metadata)
    else:
        metadata['instrument'] = TemInstrumentMetaData(getattr(data.metadata, "instrument", None))
    metadata['detector'] = CameraDetectorMetaData(getattr(data.metadata, "detector", None))

    if (not isinstance(data.axes[2], LinearAxis) or not isinstance(data.axes[3], LinearAxis)
            or getattr(data.axes[2], "unit", "px") == "px" or getattr(data.axes[3], "unit", "px") == "px"):
        try:
            voltage = int(metadata['instrument'].acceleration_voltage_kV)
            detect_angle = int(metadata['instrument'].camera_length_index)
            diff_axes = get_diff_axes(param, voltage, detect_angle)
            data.axes = data.axes[0:2] + diff_axes
        except (AttributeError, KeyError, TypeError):
            pass

    if (image is None) or image_method:
        if (not image_method) or image_method == "sum":
            image = apply_4d_stem(data, {'sum': SumDetector()}, verbose=1)["sum"]
        elif image_method in ["comx", "comy"]:
            com = apply_4d_stem(data, {'com': CenterOfMassDetector()}, verbose=1)["com"]
            image = DataSet(data=com[Ellipsis, ord(image_method[3]) - ord("x")], axes=com.axes[:-1],
                            metadata=com.metadata)
        else:
            raise ValueError("Unknown image method")

    image_scale = np.sqrt(getattr(image.axes[0], "scale", 1.0) * getattr(image.axes[1], "scale", 1.0))
    image_unit = getattr(image.axes[0], "unit", "")

    linescan_path = Path(param.get("output_file", '.'.join((merlin_path.stem, param_file_stem, "tdf"))))
    data_view = data.get(copy=False)

    pos0 = np.array(param.get('pos0', (data.shape[1] * 0.25, data.shape[0] * 0.5)), dtype=float)
    pos1 = np.array(param.get('pos1', (data.shape[1] * 0.75, data.shape[0] * 0.5)), dtype=float)
    posw = float(param.get('posw', 1))
    binsize = float(param.get('pos_binsize', 1))

    if show_4d:
        show_4d_stem(data, image, title=merlin_path, vmin=1)

    if show_linescan:
        # TODO: have some sensible import path (relative import?)
        from linescan_gui import DraggableLineScan
        drag_gui = DraggableLineScan(image.get(), pos0, pos1, posw, title=linescan_path.name)
        drag_gui.run()
        pos0 = drag_gui.pos[0]
        pos1 = drag_gui.pos[1]
        posw = drag_gui.width

    print("Creating linescan:")
    print(f'\t"pos0": [{pos0[0]:.1f}, {pos0[1]:.1f}],')
    print(f'\t"pos1": [{pos1[0]:.1f}, {pos1[1]:.1f}],')
    print(f'\t"posw": {posw:.1f},')
    print(f'\t"pos_binsize": {binsize:.1f},')

    pos_scan = line_scan(data_view, pos0, pos1, posw, bin_size=binsize)
    scan_axes = (
        LinearAxis(name='x', context='POSITION', unit=image_unit, scale=binsize * image_scale),
        data.axes[-2],
        data.axes[-1]
    )

    pos_mean = DataSet(data=pos_scan[0], axes=scan_axes, metadata=metadata)
    pos_var = DataSet(data=pos_scan[1], axes=scan_axes, metadata=metadata)
    pos_count = DataSet(data=pos_scan[2], axes=scan_axes, metadata=metadata)

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

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from pathlib import Path

from pyctem.iolib import load_mib, load_tiff, TemDataFile, load_tdf, Hdf5DataSet
from pyctem.hl import (show_4d_stem, LinearAxis, DataSet, CoreMetaData, show_position_stem,
                       apply_4d_stem, SumDetector, CenterOfMassDetector)
from pyctem.proc import line_scan
from pyctem.utils import decode_json
from pyctem import TemMeasurementMetaData

EPILOG = """
Copyright (c) 2024 Tore Niermann

This program comes with ABSOLUTELY NO WARRANTY;

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See file 'COPYING' for details.
"""

LINESCAN3D_VERSION = 4.0


class DraggableLineScan:
    lock = None  # only one can be animated at a time

    def __init__(self, image, pos0, pos1, width, title=None):
        self.pos = [np.array(pos0, dtype=float), np.array(pos1, dtype=float)]
        self.width = float(width)
        self.points = [patches.Circle(self.pos[0], 3, color='r', linewidth=0),
                       patches.Circle(self.pos[1], 3, color='r', linewidth=0)]
        self.title = title

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.ax.imshow(image, cmap="gray")

        ax_width = plt.axes((0.25, 0.1, 0.65, 0.03))
        self.width_slider = Slider(ax=ax_width, label="Width", valmin=0, valmax=300, valinit=self.width)

        for p in self.points:
            self.ax.add_patch(p)

        self.press = None
        self.patch = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.width_slider.on_changed(self.on_width)
        self.update()

    def on_width(self, value):
        self.width = value
        self.update()

    def update(self):
        ppdir = np.array((self.pos[0][1] - self.pos[1][1], self.pos[1][0] - self.pos[0][0]), dtype=float)
        ppdir /= np.sqrt(np.sum(ppdir ** 2))

        v = [self.pos[0] + self.width * 0.5 * ppdir,
             self.pos[0] - self.width * 0.5 * ppdir,
             self.pos[1] - self.width * 0.5 * ppdir,
             self.pos[1] + self.width * 0.5 * ppdir,
             self.pos[0] + self.width * 0.5 * ppdir]

        if self.patch:
            self.patch.remove()

        delta = self.pos[1] - self.pos[0]
        self.patch = self.ax.add_patch(patches.Polygon(v, fill=False, edgecolor="white"))
        title = (self.title + '\n') if self.title else ''
        self.ax.set_title(title + f'Length: {np.sqrt(np.sum(delta ** 2)):.1f} px,' +
                          f' Angle: {np.degrees(np.arctan2(delta[1], delta[0])):.1f} deg')
        plt.draw()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if self.lock is None:
            return

        point = self.lock
        for n, p in enumerate(self.points):
            if p is point:
                pt_idx = n
                break
        else:
            return

        pos = np.array((event.xdata, event.ydata), dtype=float)
        point.center = pos
        self.pos[pt_idx] = pos

        self.ax.draw_artist(point)
        self.update()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.lock is not None:
            return

        for point in self.points[::-1]:
            if point.contains(event)[0]:
                break
        else:
            return

        self.lock = point
        point.set_color('yellow')

        self.ax.draw_artist(point)
        plt.draw()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        if self.lock is None:
            return

        point = self.lock
        point.set_color('r')
        self.lock = None

        self.ax.draw_artist(point)
        self.update()

    def run(self):
        plt.show()


def get_diff_axes(param, voltage, detect_angle):
    diff_scale = param["camera_length_calibration"][voltage][detect_angle] * param.get("calibration_correction", 1.0)
    diff_axes = (LinearAxis(name='qy', context='DIFFRACTION', unit='1/nm', scale=diff_scale),
                 LinearAxis(name='qx', context='DIFFRACTION', unit='1/nm', scale=diff_scale))
    return diff_axes


def main(param, param_file_stem, show_4d=False, show_linescan=False, show_result=False, image_method=None, lazy=True,
         dryrun=False):
    work_dir = Path(param.get("work_dir", '.'))
    merlin_path = work_dir / param["stem4d_file"]

    image_path = param.get("image_file")
    if image_path:
        image_path = work_dir / image_path

    metadata = CoreMetaData()
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
        elif "image_area_size" in param:
            image_area_size = param["image_area_size"]
            image_shape = (image_area_size[1], image_area_size[0])
            data = load_mib(merlin_path, memmap=lazy, index_shape=image_shape, axes=image_axes + diff_axes)
        else:
            raise ValueError("'image_area_size' must be set, when 'image_file' is omitted.")
    elif merlin_path.suffix in ['.hdf5', '.h5']:
        import h5py

        assert data is None
        hdf5_file = h5py.File(merlin_path, "r")
        merlin_dataset = param.get("stem4d_dataset")
        if not merlin_dataset:
            names = list(hdf5_file.keys())
            if len(names) != 1 or not isinstance(hdf5_file[names[0]], h5py.Dataset):
                raise ValueError("No 'stem4d_dataset' parameter given.")
            merlin_dataset = names[0]

        hdf5_data = hdf5_file[merlin_dataset]
        metadata = TemMeasurementMetaData(filename=merlin_path, name=merlin_dataset)
        if lazy:
            data = Hdf5DataSet(hdf5_data, axes=image_axes + diff_axes, metadata=metadata)
        else:
            data = DataSet(data=hdf5_data[...], axes=image_axes + diff_axes, metadata=metadata)
            del hdf5_data
            hdf5_file.close()
            del hdf5_file

    metadata['source'] = data.metadata.ref()
    if image is not None:
        metadata['image'] = image.metadata.ref()
    try:
        metadata['instrument'] = data.metadata.instrument
    except AttributeError:
        try:
            metadata['instrument'] = image.metadata.instrument
        except AttributeError:
            pass

    if (not isinstance(data.axes[2], LinearAxis) or not isinstance(data.axes[3], LinearAxis)
            or getattr(data.axes[2], "unit", "px") == "px" or getattr(data.axes[3], "unit", "px") == "px"):
        try:
            voltage = int(data.metadata.instrument.acceleration_voltage_kV)
            detect_angle = int(data.metadata.instrument.camera_length_index)
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

    if not dryrun:
        with TemDataFile(linescan_path, 'w') as outfile:
            metadata['linescan3d_version'] = LINESCAN3D_VERSION
            metadata['pos0'] = pos0
            metadata['pos1'] = pos1
            metadata['posw'] = posw
            metadata['pos_binsize'] = binsize
            metadata['pos_scale'] = image_scale
            metadata['pos_unit'] = image_unit

            pos_scan = line_scan(data_view, pos0, pos1, posw, bin_size=binsize)

            scan_axes = (
                LinearAxis(name='x', context='POSITION', unit=image_unit, scale=binsize * image_scale),
                data.axes[-2],
                data.axes[-1]
            )

            pos_mean = DataSet(data=pos_scan[0], axes=scan_axes, metadata=metadata)
            pos_var = DataSet(data=pos_scan[1], axes=scan_axes, metadata=metadata)
            pos_count = DataSet(data=pos_scan[2], axes=scan_axes, metadata=metadata)

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


    parser = argparse.ArgumentParser(description="Create 3D linescan from 4D-STEM position space", epilog=EPILOG)
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

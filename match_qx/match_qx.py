#!/usr/bin/env python3
import time
import os
from abc import abstractmethod, ABC
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from typing import Union, Optional, Dict, Mapping, Iterable, Tuple, Any, List, NoReturn, Set
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interpn

from pyctem import BeamList, CoreMetaData, SuperCell
from pyctem.hl import LinearAxis, DataSet, Axis, BaseDataSet, SampledAxis, DataSetLike, as_dataset, BeamIndexAxis
from pyctem.utils import DTypeLike, ArrayLike, csquare, decode_json, cross_3d, calc_wavelength
from pyctem.iolib import load_tdf, TemDataFile


COPYRIGHT = """
Copyright (c) 2024 Tore Niermann

This program comes with ABSOLUTELY NO WARRANTY;

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See file 'COPYING' for details.
"""

EXAMPLE_PARAMETER_FILE = """{
    // File with experiment (as returned by linescan_diff_2d)
    "experimental": "dislocation_A.qx.tdf", 
    
    // Name of dataset (needed, if more than one dataset in experiment file)
    //"experimental_dataset": "mean",

    // Whether X / Q axes of experiment should be flipped (defaults to false)
    //"experimental_flip_q": false,
    //"experimental_flip_x": false,
    
    // Rescale X axis in calculation by this factor (defaults to 1.0). Despite the parameter name the 
    // calculated QX plot is rescaled
    //"experimental_rescale_x": 1.0,
    
    // File with calculation
    "calculation": "hw_gan5_c_0002.tdf",
    
    // Initial parameters                       
    "initial": {                                                             
        "semi_conv(1/nm)": 1.66,
        "tilt(1/nm)": 1.97,
        "depth(nm)": 50,
        "thickness(nm)": 126,
        "x_shift(nm)": -2.34,
        "q_blur(1/nm)": 0.166,
    },
    
    // Brute force parameters and their range (Min and Max, use null for range found in calculation)
    "brute_force_parameters": {
        "thickness(nm)": [120.0, 140.0],
        "depth(nm)": [40.0, 80.0],
    },
    
    // Labels for brute force plots
    //"brute_force_labels": [
    //    "$\\mathrm{Thickness}\\,t\\,(\\mathrm{nm})$",
    //    "$\\mathrm{Depth}\\,d\\,(\\mathrm{nm})$" 
    //],
    
    // Only brute force depths less equal to thickness (defaults to true)
    //"constrain_depth_thickness": true,
    
    // Which parameters to automatically optimize
    "optimized_parameters": ["tilt(1/nm)", "x_shift(nm)", "q_shift(1/nm)"],
    
    // MTF to use along q direction
    "mtf": [["GAUSSIAN", 0.738033, 102.968], ["LORENTZIAN", 0.029029, 0.117078]],
    
    // Whether background is subtracted (defaults to False)
    //"subtract_background": false,
    
    // Acceleration voltage in kV (otherwise taken from experiment)
    "acceleration_voltage(kV)": 300.0
}"""


ParameterRange = Union[Tuple[Optional[float], Optional[float]], None]


def make_axis_name(axis: Axis, n: int) -> str:
    """Return name for given Axis *axis* with index *n*"""
    name = axis.name if axis.name else f"axis-{n}"
    unit = getattr(axis, "unit", None)
    if unit:
        name += f"({unit})"
    return name


class DataSetMetadataWrapper(BaseDataSet):
    """
    Override metadata of original dataset

    :param source: Source dataset
    :param axes: Overriden axes
    :param metadata: Overriden metadata
    """
    def __init__(self, source: DataSet, axes: Tuple[Axis, ...] = None, metadata: Optional[CoreMetaData] = None):
        if axes is None:
            axes = ()
        if len(axes) > source.ndim:
            raise ValueError("Too many axes in *axes*.")
        axes += source.axes[len(axes):]

        if metadata is None:
            metadata = source.metadata

        super().__init__(shape=source.shape, dtype=source.dtype, axes=axes, metadata=metadata)
        self._source = source

    def get(self, *args, **kw):
        return self._source.get(*args, **kw)

    def set(self, *args, **kw):
        self._source.set(*args, **kw)


class QXRenderer:
    """
    Creates QX-plots from Multibeam-Data.

    :param beamlist: Beam-List, also defines crystal coordinates
    :param shape: Shape of output
    :param x_axis: Describes X-axis of output
    :param q_axis: Describes Q-axis of output
    :param q_dir_hkl: Direction of tilt axis in crystal coordinates
    :param voltage: Acceleration voltage in kV
    :param dtype: Data type of output
    :param method: Interpolation method (see scipy.interpolate.interpn)
    """
    def __init__(self, beamlist: BeamList, shape: Tuple[int, int], x_axis: LinearAxis, q_axis: LinearAxis,
                 q_dir_hkl: ArrayLike, voltage: float, dtype: DTypeLike = None, method="linear"):
        self._beamlist = beamlist

        if len(shape) != 2:
            raise ValueError("Expected *shape* to have two components.")
        self._shape = tuple(shape)

        if x_axis.unit != 'nm' or x_axis.context != 'POSITION':
            raise ValueError("Expected *x_axis* to have unit 'nm' and context 'POSITION'.")
        self._x_axis = x_axis

        if q_axis.unit != '1/nm' or q_axis.context != 'DIFFRACTION':
            raise ValueError("Expected *q_axis* to have unit '1/nm' and context 'DIFFRACTION'.")
        self._q_axis = q_axis

        q_dir_hkl = np.array(q_dir_hkl)
        if q_dir_hkl.shape != (3,):
            raise ValueError("Expected 'q_dir_hkl' to have shape (3,).")
        q_dir = self._beamlist.coord.map_rcpr_to_cartesian(q_dir_hkl)
        self._q_dir = q_dir / np.sqrt(np.sum(q_dir ** 2))

        self.dtype = np.dtype(dtype if dtype is not None else np.float32)
        self.method = method
        self.wavelength = calc_wavelength(voltage)

    @property
    def beamlist(self) -> BeamList:
        return self._beamlist

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the output"""
        return self._shape

    @property
    def axes(self) -> Tuple[Axis, ...]:
        """Axes of the output"""
        return (self._x_axis, self._q_axis)

    @property
    def q_dir(self) -> np.dtype:
        """Direction of Q-axis in crystal space"""
        return self._q_dir

    def __call__(self, semi_conv: float, tilt: float, source: DataSet, defocus: float = 0.0, curvature: float = 0.0) -> DataSet:
        """
        Create QX plot from 'source_data'.

        :param semi_conv: semi convergence angle (1/nm)
        :param tilt: Tilt offset (1/nm)
        :param source: Data source with axes: (beam-index, x-axis, tilt-axis)
        :param defocus: Defocus (nm)
        :param curvature: Curvature (1/nm2)
        """
        semi_conv = float(semi_conv)
        defocus = float(defocus)
        curvature = float(curvature)
        if semi_conv < 0.5 * abs(self._q_axis.scale):
            raise ValueError("Convergence angle must be larger than Q-axis scaling")

        if source.ndim != 3:
            raise ValueError("Expected 'source_data' to be 3 dimensional")
        if source.shape[0] != len(self._beamlist):
            raise ValueError("Expected dimension #0 to correspond to beam list length.")
        if source.axes[0].context != 'INDEX':
            raise ValueError("Expected axis #0 to have *context* 'INDEX'.")
        if getattr(source.axes[1], "unit", None) != "nm" or source.axes[1].context != "POSITION":
            raise ValueError("Expected axis #1 to have *unit* 'nm' and *context* 'POSITION.")
        if getattr(source.axes[2], "unit", None) != "1/nm":
            raise ValueError("Expected axis #2 to have *unit* '1/nm'.")
        if not isinstance(source.axes[1], LinearAxis) or not isinstance(source.axes[2], LinearAxis):
            raise ValueError("Expected axis #1 and #2 to be LinearAxis instances.")

        source_x_range = source.axis_range(1)
        source_t_range = -source.axis_range(2)

        if source_t_range[1] < source_t_range[0]:
            source_t_sign = -1.0
            source_t_range = -source_t_range
        else:
            source_t_sign = 1.0

        if source_x_range[1] < source_x_range[0]:
            source_x_sign = -1.0
            source_x_range = -source_x_range
        else:
            source_x_sign = 1.0

        metadata = source.metadata.copy()
        metadata['parameter']['semi_conv(1/nm)'] = semi_conv
        metadata['parameter']['tilt(1/nm)'] = tilt

        result = np.zeros(self._shape, dtype=f'c{2 * self.dtype.itemsize}')
        for n, hkl in enumerate(self.beamlist):
            g = self.beamlist.coord.map_rcpr_to_cartesian(hkl)

            g_shift = np.dot(g, self._q_dir)
            q_lo = (g_shift - semi_conv - self._q_axis.offset) / self._q_axis.scale
            q_hi = (g_shift + semi_conv - self._q_axis.offset) / self._q_axis.scale
            if self._q_axis.scale < 0:
                q_lo, q_hi = q_hi, q_lo
            q_lo = max(q_lo, 0)
            q_hi = min(q_hi, self.shape[1] - 1)
            q_ilo = int(np.floor(q_lo))
            q_ihi = int(np.ceil(q_hi))
            w_lo = 1.0 - (q_lo - q_ilo)
            w_hi = 1.0 - (q_ihi - q_hi)

            if q_ilo >= q_ihi:
                continue

            t = np.arange(q_ilo, q_ihi + 1, dtype=float) * self._q_axis.scale + self._q_axis.offset - g_shift - tilt
            x = self._x_axis.range(self._shape[0])
            x, t = np.meshgrid(x, t, indexing='ij')
            if defocus != 0.0:
                x += defocus * self.wavelength * t
            if curvature != 0.0:
                t += curvature * x

            disk = interpn((source_x_range, source_t_range), source[n],
                           (x * source_x_sign, t * source_t_sign), method=self.method, bounds_error=False,
                           fill_value=0.0)
            disk[:, 0] *= w_lo
            disk[:, -1] *= w_hi
            result[:, q_ilo:q_ihi + 1] += disk
        return DataSet(data=csquare(result), axes=self.axes, metadata=metadata)


def create_samples_from_range(amin: float, amax: float, step: float, dtype=None):
    if dtype is None:
        dtype = float
    count = (amax - amin) // step
    if abs(amin + count * step - amax) < 1e-4:
        count += 1
    return np.arange(count, dtype=dtype) * step + amin


class QXCalculation(ABC):
    @property
    @abstractmethod
    def metadata(self) -> CoreMetaData:
        pass

    @property
    @abstractmethod
    def beamlist(self) -> BeamList:
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[Optional[int], ...]:
        """Shape of the output"""
        pass

    @property
    @abstractmethod
    def axes(self) -> Tuple[Axis, ...]:
        """Axes of the output"""
        pass

    @property
    @abstractmethod
    def q_dir_hkl(self) -> Tuple[int, int, int]:
        """(HKL) of q-direction"""
        pass

    @abstractmethod
    def other_axes(self) -> Tuple[Axis, ...]:
        """Tuple with other Axis"""
        pass

    @abstractmethod
    def other_shape(self) -> Tuple[Optional[int], ...]:
        """Tuple with dimension of other indices"""
        pass

    @abstractmethod
    def other_names(self) -> Tuple[str, ...]:
        """Names of other axes"""
        pass

    @abstractmethod
    def __call__(self, other: Optional[Mapping[str, int]] = None) -> DataSet:
        """Return DataSet with (index, x, tilt) dimensions for given values of other_axes."""
        pass


class QXBlochWaveCalculation(QXCalculation):
    """
    Creates QX-plots from bloch wave calculation.

    :param crystal: SuperCell
    :param beamlist: BeamList
    :param q_dir_hkl: HKL of tilt direction
    :param tilt_range: min, max, step of tilt-range
    :param voltage: Acceleration voltage (kV)
    """
    def __init__(self, crystal: SuperCell, beamlist: BeamList, q_dir_hkl: Tuple[int, int, int],
                 tilt_range: Tuple[float, float, float], voltage: float, wavevector: ArrayLike,
                 foilnormal: Optional[ArrayLike] = None, formfactor_db=None):
        from pyctem.sim import BlochWaveCalculation, calc_structure_matrix

        self.crystal = crystal
        self._beamlist = beamlist
        self._q_dir_hkl = q_dir_hkl
        self._metadata = crystal.metadata
        self._tilt_samples = create_samples_from_range(*tilt_range)
        self.voltage = voltage

        self._shape = (len(beamlist), 2, len(self._tilt_samples))
        self._axes = (BeamIndexAxis(beamlist, name="beam"),
                      LinearAxis(name="x", context="POSITION", unit="nm", scale=1.0, offset=0.0),
                      SampledAxis(name="tilt", unit="1/nm", samples=self._tilt_samples).to_linear())

        self._other_axes = LinearAxis(name="thickness", unit="nm"),
        self._other_shape = None,
        self._other_names = "thickness(nm)",

        wavevector = np.array(wavevector)
        if foilnormal is None:
            foilnormal = -wavevector
        foilnormal = np.array(foilnormal)
        tilt_dir = crystal.coord.map_rcpr_to_cartesian(q_dir_hkl)
        tilt_dir /= np.sqrt(np.sum(tilt_dir ** 2))
        potential = calc_structure_matrix(crystal, beamlist, formfactor_db=formfactor_db, voltage=voltage)
        initial = np.zeros(len(beamlist), dtype=complex)
        initial[beamlist.index_hkl([0, 0, 0])] = 1.0

        self._bw = []
        self._exc = []
        for tilt in self._tilt_samples:
            simulator = BlochWaveCalculation(beamlist, potential, wavevector + tilt * q_dir_hkl, foilnormal, voltage)
            simulator.calculate()
            self._bw.append(simulator)
            self._exc.append(simulator.get_excitations(initial))

    @property
    def metadata(self) -> CoreMetaData:
        return self._metadata

    @property
    def beamlist(self) -> BeamList:
        return self._beamlist

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """Shape of the output"""
        return self._shape

    @property
    def axes(self) -> Tuple[Axis, ...]:
        """Axes of the output"""
        return self._axes

    @property
    def q_dir_hkl(self) -> Tuple[int, int, int]:
        """(HKL) of q-direction"""
        return self._q_dir_hkl

    def other_axes(self) -> Tuple[Axis, ...]:
        """Tuple with other Axis"""
        return self._other_axes

    def other_shape(self) -> Tuple[Optional[int], ...]:
        """Tuple with dimension of other indices"""
        return self._other_shape

    def other_names(self) -> Tuple[str, ...]:
        """Names of other axes"""
        return self._other_names

    def __call__(self, other: Optional[Mapping[str, int]] = None) -> DataSet:
        """Return DataSet with (index, tilt, x) dimensions for given values of other_axes."""
        out = np.empty(self._shape, dtype=complex)

        thickness = other['thickness(nm)']
        for n in range(len(self._tilt_samples)):
            out[:, :, n] = self._bw[n].get_amplitudes(self._exc[n], thickness)[:, np.newaxis]
        metadata = CoreMetaData()
        metadata["parameter"] = other
        return DataSet(data=out, axes=self._axes, metadata=metadata)


def import_crystal(name: str, base_dir: Optional[Path] = None) -> SuperCell:
    import pyctem.iolib

    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
    load_cif = getattr(pyctem.iolib, 'load_cif', None)

    filename, dataset = name.split('?', 1)
    file = base_dir / Path(filename).expanduser()

    crystal = None
    if file.is_file():
        if file.suffix == '.tdf':
            with TemDataFile(file, "r") as tdf_file:
                crystal = tdf_file.read(dataset, cls=TemDataFile.CLASS_SUPERCELL)
        elif file.suffix == '.cif' and load_cif:
            crystal = load_cif(file)

    if crystal is None:
        from pyctem.geometry import Material
        import pyctem.data

        full_name = "%sMaterial" % name
        material_class = getattr(pyctem.data, full_name)
        if not issubclass(material_class, Material):
            raise TypeError("Expected %s to be a subclass of Material." % material_class.__name__)
        material = material_class()
        if material.name() != name:
            raise ValueError("Expected material %s's name to be %s" % (material_class.__name__, name))
        crystal = material.build_crystal()

    return crystal


def create_bloch_wave_calculation(param: Dict[str, Any], voltage: float, verbose: int = 0,
                                  base_dir: Optional[Path] = None):
    from pyctem.data import get_formfactor_database
    from pyctem import create_beamlist

    voltage = float(voltage)
    wavelength = calc_wavelength(voltage)
    formfactor_db = get_formfactor_database(param.get("formfactor_db"))
    tilt_range = param["tilt_range"]

    # Load crystal
    crystal = import_crystal(param["crystal"], base_dir=base_dir)
    coord = crystal.coord

    # Prepare directions
    zoneaxis_uvw = param['zoneaxis_uvw']
    zoneaxis_dir = coord.map_real_to_cartesian(zoneaxis_uvw)
    zoneaxis_dir /= np.sqrt(np.sum(zoneaxis_dir ** 2))

    foilnormal_uvw = param.get("foilnormal")
    if foilnormal_uvw:
        foilnormal = coord.map_real_to_cartesian(np.array(foilnormal_uvw, dtype=float))
        foilnormal /= np.sqrt(np.sum(foilnormal ** 2))
    else:
        foilnormal = zoneaxis_dir

    # Create beam list for crystal #0
    systematic_row = param.get("systematic_row")
    if systematic_row is not None:
        systematic_row = np.array(systematic_row, dtype=int)
        if abs(np.dot(zoneaxis_uvw, systematic_row)) > 1e-5:
            raise ValueError(f"Systematic row not perpendicular to zone axis.")

    wavevector = zoneaxis_dir / wavelength

    max_rcpr_distance = float(param.get("max_rcpr_distance", 30.0))
    min_potential = float(param.get("min_potential", 0.01))
    max_exc_error = float(param.get("max_exc_error", 0.5))
    beamlist = create_beamlist(crystal, wavevector, systematic_row=systematic_row,
                               foilnormal=foilnormal, max_rcpr_distance=max_rcpr_distance,
                               min_potential=min_potential, max_exc_error=max_exc_error)

    tilt_dir_hkl = param.get("tilt_dir_hkl", systematic_row)
    if tilt_dir_hkl is None:
        raise ValueError("Parameter 'tilt_dir_hkl' required if no systematic row is given.")
    tilt_dir_hkl = np.array(tilt_dir_hkl, dtype=float)

    if verbose >= 1:
        if systematic_row is not None:
            print(f"\tSystematic row (hkl): {systematic_row}")
        if zoneaxis_uvw is not None:
            print(f"\tZoneaxis [uvw]:       {zoneaxis_uvw}")
        print(f"\tFoil-normal:          {foilnormal}")
        if tilt_dir_hkl is not None:
            print(f"\tTilt direction (hkl): {tilt_dir_hkl}")
        print(f"\tWavevector (1/nm):    {wavevector}")
        print(f"\tBeam direction:       {wavevector / np.sqrt(np.sum(wavevector ** 2))}")

    return QXBlochWaveCalculation(crystal, beamlist, q_dir_hkl=tilt_dir_hkl, tilt_range=tilt_range, voltage=voltage,
                                  wavevector=wavevector, foilnormal=foilnormal, formfactor_db=formfactor_db)


class QXCalculationSelector(QXCalculation):
    """
    Creates QX-plots from calculation data.

    :param amplitudes: Amplitudes data set (..., tilt, thickness, index, y, x)
    :param x_index: Name or index of axis to use for position
    :param tilt_index: Name or index of axis to use for tilt
    :param beam_index: Name or index of axis to use for beam index
    """
    def __init__(self, amplitudes: DataSet, beamlist: BeamList, x_index: Union[str, int] = 'x', tilt_index: Union[str, int] = 'tilt',
                 beam_index: Union[str, int] = 'beam'):
        self._amplitudes = amplitudes
        self._beamlist = beamlist
        axes = self._amplitudes.axes

        self._x_index = self._get_axis(x_index)
        self._tilt_index = self._get_axis(tilt_index)
        self._beam_index = self._get_axis(beam_index)

        self._q_dir_hkl = amplitudes.metadata["tilt_dir(hkl)"]
        self._other_indices = tuple(n for n in range(amplitudes.ndim) if n not in [self._x_index, self._beam_index, self._tilt_index])
        self._other_names = tuple(make_axis_name(axes[n], n) for n in self.other_indices())

        self._output_indices = (self._beam_index, self._x_index, self._tilt_index)

        order = np.argsort(self._output_indices).tolist()
        self._axes_order = tuple(order.index(n) for n in range(3))

    def _get_axis(self, index_or_name: Union[str, int]) -> int:
        if isinstance(index_or_name, str):
            return self._amplitudes.index_axis(name=index_or_name)

        index = int(index_or_name)
        if index < 0:
            index += self._amplitudes.ndim
        if index >= self._amplitudes.ndim:
            raise IndexError(f"Invalid axis: {index_or_name}")

        return index

    @property
    def x_index(self) -> int:
        """Index of X-Axis"""
        return self._x_index

    @property
    def tilt_index(self) -> int:
        """Index of Tilt-Axis"""
        return self._tilt_index

    @property
    def beam_index(self) -> int:
        """Index of Beam-Axis"""
        return self._beam_index

    @property
    def dtype(self) -> Tuple[Axis, ...]:
        """Data type of the output"""
        return self._amplitudes.dtype

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """Shape of the output"""
        return tuple(self._amplitudes.shape[n] for n in self._output_indices)

    @property
    def axes(self) -> Tuple[Axis, ...]:
        """Axes of the output"""
        return tuple(self._amplitudes.axes[n] for n in self._output_indices)

    @property
    def beamlist(self) -> BeamList:
        return self._beamlist

    @property
    def metadata(self) -> CoreMetaData:
        return self._amplitudes.metadata

    @property
    def q_dir_hkl(self) -> Tuple[int, int, int]:
        return self._q_dir_hkl

    def other_indices(self) -> Tuple[int, ...]:
        """Tuple with other axis indices"""
        return self._other_indices

    def other_axes(self) ->Tuple[Axis, ...]:
        """Tuple with other Axis"""
        return tuple(self._amplitudes.axes[n] for n in self.other_indices())

    def other_shape(self) -> Tuple[Axis, ...]:
        """Tuple with dimension of other indices"""
        return tuple(self._amplitudes.shape[n] for n in self.other_indices())

    def other_names(self) -> Tuple[str, ...]:
        """Names of other axes"""
        return self._other_names

    def __call__(self, other: Optional[Mapping[str, int]] = None) -> DataSet:
        """Return DataSet with (index, tilt, x) dimensions for given values of other_axes."""
        if other is None:
            other = {}
        index = [slice(None)] * self._amplitudes.ndim
        other_values = {}
        for n, name, axis, dim in zip(self.other_indices(), self.other_names(), self.other_axes(), self.other_shape()):
            value = other.get(name, 0)
            val_range = axis.range(dim)
            val_index = int(np.argmin(abs(val_range - value)))
            index[n] = val_index
            other_values[name] = val_range[val_index]

        data = self._amplitudes.get(tuple(index), copy=False)
        data = np.transpose(data, self._axes_order)
        metadata = CoreMetaData()
        metadata["calculation"] = self._amplitudes.metadata.ref()
        metadata["parameter"] = other_values
        return DataSet(data=data, axes=self.axes, metadata=metadata)


def crop_margin(data: DataSet, margin: Union[int, Tuple[int, ...]]) -> DataSet:
    """
    Removes outer pixels.

    :param data: Dataset to crop
    :param margin: Either a common crop margin to all axes, or for each axis
    """
    if isinstance(margin, int):
        margin = [margin] * data.ndim
    if len(margin) != data.ndim:
        raise ValueError("There must be a margin for each dimension.")
    if any(n < 0 for n in margin):
        raise ValueError("Margins must be non-negative")
    axes = tuple(LinearAxis(a, offset=a.offset + m * a.scale) for a, m in zip(data.axes, margin))
    index = tuple(slice(m, -m) if m > 0 else slice(None) for m in margin)
    return DataSet(data=data.get(index, copy=False), axes=axes, metadata=data.metadata)


def qx_blur_q(source: DataSet, sigma: float) -> DataSet:
    """
    Blur dataset by Gaussian in Q direction

    :param source: Source data (x, q)
    :param sigma: Amount of blur (1/nm)
    """
    from scipy.ndimage import gaussian_filter1d

    q_axis = source.axes[-1]
    if getattr(q_axis, "unit", None) != "1/nm" or q_axis.context != "DIFFRACTION":
        raise ValueError("Expected last axes to have *unit* '1/nm' and *context* 'DIFFRACTION'")
    q_scale = getattr(q_axis, "scale", 1.0)

    sigma = abs(float(sigma))
    data = source.get(copy=False)
    if sigma > 1e-3:
        px_sigma = sigma / q_scale
        data = gaussian_filter1d(data, sigma=px_sigma)
    else:
        sigma = 0.0

    metadata = source.metadata.copy()
    metadata["parameter"]["q_blur(1/nm)"] = sigma
    return DataSet(data=data, axes=source.axes, metadata=metadata)


def apply_mtf(source: DataSet, mtf_param: Iterable[Tuple], axis: int = -1, binning: int = 1) -> DataSet:
    """
    Apply MTF in direction of the given axis.

    :param source: Source data (x, q)
    :param mtf_param: MTF parameterization
    :param axis: Axis to apply MTF to
    :param binning: Binning
    """
    from pyctem.utils import parameterized_mtf, get_float_dtype

    if axis < 0:
        axis += source.ndim

    tmp = np.fft.fft(source.get(copy=False), axis=axis)
    q = np.fft.fftfreq(tmp.shape[axis])
    mtf = parameterized_mtf(q, mtf_param)

    bc = tuple(slice(None) if n == axis else np.newaxis for n in range(source.ndim))
    tmp *= mtf[bc]

    data = np.fft.ifft(tmp, axis=axis).real.astype(get_float_dtype(source.dtype))
    metadata = source.metadata.copy()
    metadata["parameter"]["mtf_param"] = tuple(mtf_param)
    metadata["parameter"]["mtf_binning"] = binning
    return DataSet(data=data, axes=source.axes, metadata=metadata)


def l2_loss(experiment: DataSet, calculation: DataSet, weights: Optional[ArrayLike] = None):
    sr = (experiment.get() - calculation.get()) ** 2
    if weights is not None:
        sr *= weights
    return np.sum(sr)


def attenuate(attenuation: float, source: DataSet) -> DataSet:
    """
    Attenuates dataset *source* by *attenuation*

    :param attenuation: Attenuation factor
    :param source: Dataset to attenuate
    """
    attenuation = float(attenuation)
    metadata = source.metadata.copy()
    metadata["parameter"]["attenuation"] = attenuation
    data = source.get(copy=True)
    data *= attenuation
    return DataSet(data=data, axes=source.axes, metadata=metadata)


def optimize_attenuation(experiment: DataSet, calculation: DataSet) -> float:
    """Returns optimal (L2 minimal) attenuation factor for *calculation* to match *experiment*"""
    a = calculation.get(copy=True)
    b = experiment.get(copy=True)
    return np.sum(a * b) / np.sum(a ** 2 + 1e-5)


class MatchQXPipeline:
    """
    Encapsulate matching of QX plots.

    :param experimental: Experimental dataset, axes (x-axis, q-axis)
    :param calculation: Calculation class, axes (x-axis, q-axis)
    :param beamlist: Beamlist for calculation
    :param background: Background estimation,
    :param x_axis: Override experimental X-axis
    :param q_axis: Override experimental Q-axis
    :param q_margin: Q-Margin for calculated data
    :param mtf: MTF for MTF blurring
    :param voltage: Acceleration voltage (kV), taken from experimental if omitted
    :param weights: Weights for loss function, axes (x-axis, q-axis)
    """
    def __init__(self, experimental: DataSet, calculation: QXCalculation, voltage: float,
                 background: Optional[ArrayLike] = None,
                 x_axis: Optional[LinearAxis] = None, q_axis: Optional[LinearAxis] = None,
                 q_margin: int = 3, mtf: Optional[Tuple] = None,
                 weights: Optional[ArrayLike] = None):
        if experimental.ndim != 2:
            raise ValueError("Expected *experimental* to be two dimensional")
        self.experimental = experimental
        self._shape = experimental.shape
        self.weights = weights

        self._q_margin = int(q_margin)
        self._mtf_param = mtf

        self.calculation = calculation
        self.voltage = voltage
        if background is not None:
            self.background = np.array(background, dtype=float)
            self.experimental = DataSet(data=self.experimental.get() - self.background, axes=self.experimental.axes, metadata=self.experimental.metadata)
        else:
            self.background = None

        if x_axis is None:
            x_axis = self.experimental.axes[0]
        if x_axis.unit != 'nm' or x_axis.context != 'POSITION':
            raise ValueError("Expected *x_axis* to have unit 'nm' and context 'POSITION'.")
        self._x_axis = x_axis

        if q_axis is None:
            q_axis = self.experimental.axes[1]
        if q_axis.unit != '1/nm' or q_axis.context != 'DIFFRACTION':
            raise ValueError("Expected *q_axis* to have unit '1/nm' and context 'DIFFRACTION'.")
        self._q_axis = q_axis

        # Pipeline
        self._q_shift = None
        self._x_shift = None
        self._make_renderer(0.0, 0.0)

        # Parameters
        self._default_param = {
            "semi_conv(1/nm)" : 1.0,
            "tilt(1/nm)": 0.0,
            "q_blur(1/nm)": 0.0,
            "q_shift(1/nm)": 0.0,
            "x_shift(nm)": 0.0,
            "defocus(nm)": 0.0,
            "curvature(1/nm2)": 0.0,
        }
        self._param_scales = {
            "semi_conv(1/nm)": 0.1,
            "tilt(1/nm)": 1.0,
            "q_blur(1/nm)": 0.1,
            "q_shift(1/nm)": 0.1,
            "x_shift(nm)": 1.0,
            "defocus(nm)": 1.0,
            "curvature(1/nm2)": 0.001,
        }
        for name, axis in zip(self.calculation.other_names(), self.calculation.other_axes()):
            if isinstance(axis, SampledAxis):
                axis = axis.try_linear()
            self._param_scales[name] = getattr(axis, "scale", 1.0)
            self._default_param[name] = 0.0

    def default_parameters(self):
        """Return default parameter values"""
        return self._default_param

    def parameter_keys(self):
        """Return keys iterable for parameters"""
        return self._default_param.keys()

    def parameter_scale(self, key):
        """Return scale for parameters"""
        return self._param_scales[key]

    def other_keys(self):
        """Return keys needed for calculation selection"""
        return self.calculation.other_names()

    def other_axes(self):
        """Return axes of keys needed for calculation selection"""
        return self.calculation.other_axes()

    def other_shape(self):
        """Return shape of keys needed for calculation selection"""
        return self.calculation.other_shape()

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Weighting of experimental data"""
        return self._weights

    @weights.setter
    def weights(self, value: Optional[ArrayLike]):
        if value is not None:
            value = np.array(value)
            if value.shape != self._shape:
                raise ValueError("Weights array must have same shape as experimental array.")
            self._weights = value
        else:
            self._weights = None

    def _make_renderer(self, q_shift: float, x_shift: float):
        self._q_shift = q_shift
        self._x_shift = x_shift
        q_offset = getattr(self.experimental.axes[1], "offset", 0.0)
        x_offset = getattr(self.experimental.axes[0], "offset", 0.0)
        self._x_axis = LinearAxis(self.experimental.axes[0], offset=x_offset + x_shift)
        self._q_axis = LinearAxis(self.experimental.axes[1], offset=q_offset + q_shift)

        q_axis = LinearAxis(self._q_axis, offset=self._q_axis.offset - self._q_axis.scale * self._q_margin)
        shape = (self.experimental.shape[0], self.experimental.shape[1] + 2 * self._q_margin)

        # TODO: Preliminary fix for interpolation problem is setting method to "nearest" (instead of "linear")
        self._renderer = QXRenderer(self.calculation.beamlist, shape, x_axis=self._x_axis, q_axis=q_axis,  #method="nearest",
                                    q_dir_hkl=self.calculation.q_dir_hkl, voltage=self.voltage)

    def evaluate(self, param):
        q_shift = param["q_shift(1/nm)"]
        x_shift = param["x_shift(nm)"]
        if (self._q_shift is None) or (self._x_shift is None) or abs(q_shift - self._q_shift) > 1e-6 or abs(x_shift - self._x_shift) > 1e-6:
            self._make_renderer(q_shift, x_shift)

        other_index = {k: int(param[k]) for k in self.other_keys()}
        calculation = self.calculation(other_index)

        semi_conv = param["semi_conv(1/nm)"]
        tilt = param["tilt(1/nm)"]
        defocus = param["defocus(nm)"]
        curvature = param["curvature(1/nm2)"]
        render = self._renderer(semi_conv, tilt, calculation, defocus=defocus, curvature=curvature)

        if self._mtf_param:
            render = apply_mtf(render, self._mtf_param)

        q_blur = param["q_blur(1/nm)"]
        render = qx_blur_q(render, q_blur)
        render = crop_margin(render, (0, self._q_margin))

        att = optimize_attenuation(self.experimental, render)
        final = attenuate(att, render)

        loss = l2_loss(self.experimental, final, self.weights)
        return final, loss


def format_label(name: str, unit: str) -> str:
    """Create nicely TeX'ed axis labels"""
    unit = unit.replace('%', '\\%')
    name = name.replace('%', '\\%')
    name = name.replace('_', '\\_')
    if unit == '1/nm':
        unit = '\\mathrm{nm}^{-1}'
    else:
        unit = f'\\mathrm{{{unit}}}'
    return f"$\\mathrm{{{name}}}\\,({unit})$"


class WidgetLayouter:
    """Helper class to accommodate a dynamic number of sliders in interactive matplotlib figure."""
    def __init__(self, num=3):
        import matplotlib.pyplot as plt
        plt.subplots_adjust(bottom=0.05 + num * 0.05)
        self._offset = 0.05 + num * 0.05
        self._items = {}

    def add_slider(self, label, min_value, max_value, valinit=None, on_changed=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        if label in self._items:
            raise ValueError(f"Duplicate widget name '{label}'.")

        ax = plt.axes((0.25, self._offset, 0.65, 0.03))
        self._offset -= 0.05
        slider = Slider(ax, label, min_value, max_value, valinit=valinit)
        if callable(on_changed):
            slider.on_changed(on_changed)

        self._items[label] = slider
        return slider

    def items(self) -> Iterable:
        """Returns name and instance"""
        return self._items.items()


def optimize_parameter(pipeline: MatchQXPipeline, initial_param: Dict[str, float], keys: Iterable[str],
                       verbose: int = 1, raise_on_failure: bool = False, maxiter: int = 100) -> Tuple[Dict[str, float], float]:
    """
    Optimize parameters for pipeline using Brent or BFGS.

    :param pipeline: Evaluation pipeline
    :param initial_param: Start values
    :param keys: Parameters to optimize
    :param verbose: Verbosity
    :param raise_on_failure: Whether a ValueError is raised, when optimization failed
    :param maxiter: Max number of iterations
    :returns: optimized parameters, loss value at optimum
    """
    param = pipeline.default_parameters()
    param.update(initial_param)

    if isinstance(keys, str):
        keys = (keys,)
    else:
        keys = tuple(keys)

    from scipy.optimize import minimize, minimize_scalar, bracket

    if len(keys) == 1:
        key = keys[0]
        scale = pipeline.parameter_scale(key)

        def eval_scalar(x):
            param[key] = x * scale
            _, loss = pipeline.evaluate(param)
            #print(x, loss)
            return loss

        initial = param[key] / scale
        try:
            bracket_res = bracket(eval_scalar, initial - 0.5, initial + 0.5)
            res = minimize_scalar(eval_scalar, bracket=bracket_res[0:3], method="Brent",
                                  options={'disp': verbose, 'xtol': 1e-3, 'maxiter': maxiter})
        except (ValueError, RuntimeError) as exc:
            if raise_on_failure:
                raise ValueError(f"Optimization of '{key}' failed: {exc}\n")
            _, res_loss = pipeline.evaluate(param)
        else:
            if raise_on_failure and not res.success:
                raise ValueError(f"Optimization of '{key}' failed:\n{res.message}")
            param[key] = res.x * scale
            res_loss = res.fun
    else:
        x0 = np.empty(len(keys), dtype=float)
        scale = np.empty(len(keys), dtype=float)
        for n, k in enumerate(keys):
            scale[n] = pipeline.parameter_scale(k)
            x0[n] = param[k] / scale[n]

        def eval_mvar(x):
            for n, key in enumerate(keys):
                param[key] = x[n] * scale[n]
            _, loss = pipeline.evaluate(param)
            #print(x, loss)
            return loss

        res = minimize(eval_mvar, x0, method="BFGS", options={
            'disp': verbose, 'eps': 1e-4, 'xrtol': 1e-3, 'maxiter': maxiter
        })

        if raise_on_failure and not res.success:
            raise ValueError(f"Optimization of {keys} failed:\n{res.message}")

        for n, key in enumerate(keys):
            param[key] = res.x[n] * scale[n]
        res_loss = res.fun

    return param, res_loss


class MatchQXPlot:
    """
    Plot showing result of QX matching

    :param pipeline: Matching pipeline
    :param param: Parameters
    :param aspect: Aspect ratio of plot
    :param cmap: Colormap (gray by default)
    :param interpolation: Interpolation method in images.
    :param rescale_x: Rescaling of experimental X-axis
    :param rescale_loss: Rescaling of Losses
    :param optimized_parameters: Parameters to optimize on request
    """
    def __init__(self, pipeline: MatchQXPipeline, param: Optional[dict] = None,
                 aspect: Optional[float] = None, cmap: str ="gray", interpolation: Optional[str] = None,
                 rescale_x: float = 1.0, rescale_loss: float = 1.0,
                 optimized_parameters: Iterable[str] = None):
        self.pipeline = pipeline
        self.optimized_parameters = frozenset(optimized_parameters) if optimized_parameters else frozenset()

        self.aspect = aspect
        self.cmap = cmap
        self.rescale_x = rescale_x
        self.rescale_loss = rescale_loss
        self.interpolation = interpolation

        self.param = dict(pipeline.default_parameters())
        if param:
            self.param.update(param)

        self.title = []
        if self.experimental.metadata.filename:
            self.title.append("Experimental: " + Path(self.experimental.metadata.filename).stem)
        if self.calculation.metadata.filename:
            self.title.append("Calculation: " + Path(self.calculation.metadata.filename).stem)

    @property
    def experimental(self):
        return self.pipeline.experimental

    @property
    def calculation(self):
        return self.pipeline.calculation

    def get_extent(self, calc: DataSet):
        x_scale = getattr(calc.axes[0], "scale", 1.0)
        x_offset = getattr(calc.axes[0], "offset", 0.0)
        q_scale = getattr(calc.axes[1], "scale", 1.0)
        q_offset = getattr(calc.axes[1], "offset", 0.0)
        return (
            q_offset - 0.5 * q_scale,
            q_offset + (calc.shape[1] - 0.5) * q_scale,
            (x_offset + (calc.shape[0] - 0.5) * x_scale) / self.rescale_x,
            (x_offset - 0.5 * x_scale) / self.rescale_x
        )

    def _init_plot(self, additional_plots=None, **fig_kw):
        import matplotlib.pyplot as plt

        calc, loss = self.pipeline.evaluate(self.param)
        extent = self.get_extent(calc)

        if self.aspect is None:
            aspect = abs(extent[1] - extent[0]) / abs(extent[2] - extent[3])
        else:
            aspect = self.aspect

        if additional_plots:
            cols = len(additional_plots)
            fig, ax = plt.subplots(nrows=1, ncols=2 + cols, **fig_kw)
            ax[1].sharex(ax[0])
            ax[1].sharey(ax[0])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=3, sharex="all", sharey="all", **fig_kw)

        plt.suptitle('\n'.join(self.title + [f"Loss: {loss * self.rescale_loss}"]))

        background = self.pipeline.background if self.pipeline.background is not None else 0.0
        vmin = np.amin(self.experimental.get(copy=False) + background)
        vmax = np.amax(self.experimental.get(copy=False) + background)
        im0 = ax[0].imshow(self.experimental.get(copy=False) + background, extent=extent, aspect=aspect, cmap=self.cmap,
                           vmin=vmin, vmax=vmax, interpolation=self.interpolation)
        ax[0].set_title("Experimental")

        im1 = ax[1].imshow(calc.get(copy=False) + background, extent=extent, aspect=aspect, cmap=self.cmap,
                           vmin=vmin, vmax=vmax, interpolation=self.interpolation)
        ax[1].set_title("Calculation")

        additional_states = []
        if additional_plots:
            for n, plot in enumerate(additional_plots):
                additional_states.append(plot.setup(fig, ax[2 + n], self.param))
            im2 = None
            max_ax = 2
        else:
            res = self.experimental.get(copy=False) - calc.get(copy=False)
            rmax = np.amax(abs(res))
            im2 = ax[2].imshow(res, extent=extent, aspect=aspect, cmap="RdBu",
                               vmin=-rmax, vmax=rmax, interpolation=self.interpolation)
            ax[2].set_title("Residual")
            max_ax = 3

        for a in ax[:max_ax]:
            a.set_xlabel(format_label(self.experimental.axes[1].name, getattr(self.experimental.axes[1], "unit", "")))
            a.set_ylabel(format_label(self.experimental.axes[0].name, getattr(self.experimental.axes[0], "unit", "")))
            a.tick_params(axis='x', labelbottom=True)
            a.tick_params(axis='y', labelleft=True)

        return fig, ax, (im0, im1, im2), additional_states

    def run_plot(self, additional_plots=None, additional_updates=None, **fig_kw):
        import matplotlib.pyplot as plt

        additional_plots = list(additional_plots) if additional_plots else ()
        additional_updates = list(additional_updates) if additional_updates else []
        fig, ax, im, additional_states = self._init_plot(additional_plots=additional_plots, **fig_kw)
        for plot, state in zip(additional_plots, additional_states):
            additional_updates.append((plot.update, state))

        other_shape = self.pipeline.other_shape()
        other_num = sum(1 for s in other_shape if (s is None) or (s > 1))
        layouter = WidgetLayouter(5 + other_num)
        other_range = {}

        def update():
            background = self.pipeline.background if self.pipeline.background is not None else 0.0
            calc, loss = self.pipeline.evaluate(self.param)
            fig.suptitle('\n'.join(self.title + [f"Loss: {loss * self.rescale_loss}"]))

            extent = self.get_extent(calc)
            ax[0].set_xlim(extent[0], extent[1])
            ax[0].set_ylim(extent[2], extent[3])
            for img in im:
                if img is not None:
                    img.set_extent(extent)

            calc_data = calc.get(copy=False)
            im[1].set_data(calc_data + background)

            if im[2] is not None:
                res = self.experimental.get(copy=False) - calc_data
                im[2].set_data(res)

            for update_func, state in additional_updates:
                update_func(state, self.param)

            fig.canvas.draw_idle()

        def changed(_):
            for name, slider in layouter.items():
                value = slider.val
                try:
                    val_range = other_range[name]
                    index = int(np.argmin(abs(val_range - value)))
                    if abs(value - val_range[index]) > 1e-5:
                        slider.set_val(val_range[index])
                        return
                except KeyError:
                    pass
                self.param[name] = value
            update()

        def optimize(_):
            self.param, _ = optimize_parameter(self.pipeline, self.param, self.optimized_parameters)
            for name, slider in layouter.items():
                value = self.param[name]
                if abs(value - slider.val) > 1e-5:
                    slider.set_val(value)
            #update()

        layouter.add_slider("x_shift(nm)", self.param["x_shift(nm)"] - 10.0,
                            self.param["x_shift(nm)"] + 10.0, self.param["x_shift(nm)"], on_changed=changed)
        layouter.add_slider("q_shift(1/nm)", self.param["q_shift(1/nm)"] - 1.0,
                            self.param["q_shift(1/nm)"] + 1.0, self.param["q_shift(1/nm)"], on_changed=changed)
        layouter.add_slider("q_blur(1/nm)", 0.0, +0.5, self.param["q_blur(1/nm)"], on_changed=changed)
        layouter.add_slider("semi_conv(1/nm)", self.experimental.axes[1].scale * 2.0, 3.0,
                            valinit=self.param["semi_conv(1/nm)"], on_changed=changed)
        layouter.add_slider("tilt(1/nm)", -3.0, +3.0, valinit=self.param["tilt(1/nm)"], on_changed=changed)
        layouter.add_slider("defocus(nm)", -500.0, +500.0, valinit=self.param["defocus(nm)"], on_changed=changed)
        layouter.add_slider("curvature(1/nm2)", -0.05, +0.05, valinit=self.param["curvature(1/nm2)"], on_changed=changed)

        for name, axis, dim in zip(self.pipeline.other_keys(), self.pipeline.other_axes(), other_shape):
            if dim == 1:
                continue
            minmax = axis.minmax(0, dim)
            value = self.param[name]
            layouter.add_slider(name, minmax[0], minmax[1], valinit=value, on_changed=changed)
            other_range[name] = axis.range(dim)

        if self.optimized_parameters:
            from matplotlib.widgets import Button
            opt_ax = fig.add_axes([0.8, 0.925, 0.2, 0.075])
            opt_btn = Button(opt_ax, 'Optimize')
            opt_btn.on_clicked(optimize)

        plt.show()

    def save_plot(self, filename, additional_plots=None, **fig_kw):
        import matplotlib.pyplot as plt

        fig, ax, im, additional_states = self._init_plot(additional_plots=additional_plots, **fig_kw)

        param_keys = ("x_shift(nm)", "q_shift(1/nm)", "q_blur(1/nm)", "semi_conv(1/nm)", "tilt(1/nm)", "defocus(nm)", "curvature(1/nm2)") \
            + tuple(self.pipeline.other_keys())

        fig.subplots_adjust(bottom=0.05 + len(param_keys) * 0.03)
        text_ax = plt.axes([0.15, 0.05, 0.70, len(param_keys) * 0.03])
        text_ax.axis('off')
        text_ax.text(0.5, 0.5, '\n'.join(f'{k}: ' for k in param_keys),
                     horizontalalignment='right', verticalalignment='center', transform=text_ax.transAxes)
        text_ax.text(0.5, 0.5, '\n'.join(f"{self.param[k]:.2f}" for k in param_keys),
                     horizontalalignment='left', verticalalignment='center', transform=text_ax.transAxes)

        fig.savefig(filename)
        plt.close(fig)

    def save_sweep(self, filename, key, value_range, title=None, additional_plots=None, dpi=100, ffmpeg_kw=None, **fig_kw):
        from matplotlib.animation import FFMpegWriter

        fig, ax, im, additional_states = self._init_plot(additional_plots=additional_plots, dpi=dpi, **fig_kw)
        fig.subplots_adjust(left=0.05, right=0.95)
        param = dict(self.param)

        background = self.pipeline.background if self.pipeline.background is not None else 0.0

        if ffmpeg_kw is None:
            ffmpeg_kw = {}
        writer = FFMpegWriter(**ffmpeg_kw)
        with writer.saving(fig, filename, dpi):
            for n, value in enumerate(value_range):
                param[key] = value

                calc, loss = self.pipeline.evaluate(param)
                if title:
                    fig.suptitle(title.format(value=value))
                else:
                    fig.suptitle('\n'.join(self.title + [f"{key}: {value}", f"Loss: {loss * self.rescale_loss}"]))

                extent = self.get_extent(calc)
                ax[0].set_xlim(extent[0], extent[1])
                ax[0].set_ylim(extent[2], extent[3])
                for img in im:
                    if img is not None:
                        img.set_extent(extent)

                calc_data = calc.get(copy=False)
                im[1].set_data(calc_data + background)

                if im[2] is not None:
                    res = self.experimental.get(copy=False) - calc_data
                    im[2].set_data(res)

                if additional_plots:
                    for plot, state in zip(additional_plots, additional_states):
                        plot.update(state, param)

                print(f"{filename}: frame {n}/{len(value_range)} - {key}={value}...")
                writer.grab_frame()


def _evaluate_brute_force(task):
    """Helper method for brute force evaluation in multiprocessing environment"""
    import multiprocessing as mp
    index, pipeline, param, optimized_parameters, verbose = task
    if verbose > 1:
        print(mp.current_process().name, "START", index)
    if optimized_parameters:
        param, loss = optimize_parameter(pipeline, param, optimized_parameters, verbose=min(verbose, 1))
    else:
        loss = pipeline.evaluate(param)
    if verbose > 1:
        print(mp.current_process().name, "DONE", index, loss)
    return index, param, loss


def brute_force(pipeline: MatchQXPipeline, initial_param: Dict[str, float], axes: Iterable[SampledAxis],
                optimized_parameters: Optional[Tuple[str, ...]] = None, constrain_depth_thickness: bool = False,
                verbose: int = 1, pool=None, prefix: Optional[Tuple[str]] = ()) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Perform brute force evaluation of parameters along given axes"""
    axes = tuple(axes)
    shape = tuple(len(a.samples) for a in axes)
    result_param = np.full(shape, initial_param, dtype=object)
    result_loss = np.full(shape, float("NaN"), dtype=float)

    param_name = tuple(f"{a.name}({a.unit})" for a in axes)

    tasks = []
    for index in np.ndindex(*shape):
        param = dict(initial_param)
        for n, p, a in zip(index, param_name, axes):
            param[p] = a.samples[n]
        if constrain_depth_thickness and ("depth(nm)" in param) and ("thickness(nm)" in param) and param["depth(nm)"] > param["thickness(nm)"]:
            continue
        tasks.append((index, pipeline, param, optimized_parameters, max(0, verbose - 1)))

    if pool is not None:
        task_iter = pool.imap_unordered(_evaluate_brute_force, tasks)
    else:
        task_iter = map(_evaluate_brute_force, tasks)

    for index, param, loss in task_iter:
        if verbose > 0:
            text = ', '.join(prefix + tuple(f'{k}={param[k]:.2f}' for k in param_name))
            print(text, "->", loss)
        result_param[index] = dict(param)
        result_loss[index] = loss

    return result_param, result_loss


def create_sub_sample_axes(calculation: QXCalculation, key: str, min_max_step: Optional[Tuple[Optional[float], ...]] = None):
    name, unit = split_name_and_unit(key)

    try:
        dim, axis = next(pair for pair in zip(chain(calculation.shape, calculation.other_shape()),
                                              chain(calculation.axes, calculation.other_axes()))
                         if pair[1].name == name)
    except StopIteration:
        raise ValueError(f"No parameter '{name}' found.")
    if axis.unit != unit:
        raise ValueError(f"Expected unit '{unit}' on axis: {axis}")

    if dim is None:
        if not min_max_step or any(m is None for m in min_max_step):
            raise ValueError(f"Expected min, max, step for continuous parameter '{key}'")
        samples = create_samples_from_range(*min_max_step)
    else:
        full_samples = axis.range(dim)
        if min_max_step is None:
            min_max_step = None, None, None
        amin = min_max_step[0] if min_max_step[0] is not None else np.amin(full_samples)
        amax = min_max_step[1] if min_max_step[1] is not None else np.amax(full_samples)
        if len(min_max_step) > 2 and min_max_step[2]:
            step = min_max_step[2]
        else:
            step = np.mean(np.diff(full_samples))
        asamples = create_samples_from_range(amin, amax, step)
        isamples = np.argmin(abs(asamples[:, np.newaxis] - full_samples), axis=1)
        samples = full_samples[np.unique(isamples)]
    return SampledAxis(name=name, unit=unit, samples=samples)


def background_support(dataset: DataSet, beamlist: BeamList, q_dir_hkl: ArrayLike, q_shift: float = 0.0,
                       margin: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    q_axis = LinearAxis(dataset.axes[1], offset=dataset.axes[1].offset + q_shift)
    data_mean = np.mean(dataset.get(), axis=0)

    q_dir = beamlist.coord.map_rcpr_to_cartesian(q_dir_hkl)
    q_dir = q_dir / np.sqrt(np.sum(q_dir ** 2))

    g_shifts = []
    for n, hkl in enumerate(beamlist):
        g = beamlist.coord.map_rcpr_to_cartesian(hkl)
        if np.sum(cross_3d(g, q_dir) ** 2) > 1e-3:
            print("Excluded {hkl}")
            continue
        g_shifts.append(np.dot(g, q_dir))
    g_shifts = np.sort(g_shifts)

    support = 0.5 * (g_shifts[1:] + g_shifts[:-1])

    g_support = []
    background = []
    for g in (support - q_axis.offset) / q_axis.scale:
        i = int(np.floor(g))
        while margin <= i <= (dataset.shape[1] - margin - 1):
            b = data_mean[i]
            bl = data_mean[i - 1]
            br = data_mean[i + 1]
            if b > bl:
                i -= 1
            elif b > br:
                i += 1
            else:
                g_support.append(i * q_axis.scale + q_axis.offset)
                background.append(b)
                break

    g_support = np.array(g_support, dtype=float)
    background = np.array(background, dtype=float)
    return np.array(g_support, dtype=float), np.array(background, dtype=float)


def gaussian_background_constrained(q_range: ArrayLike, q_support: ArrayLike, support_data: ArrayLike) -> np.ndarray:
    from scipy.optimize import least_squares

    support_q2 = np.asanyarray(q_support) ** 2
    p0 = support_data.max(), 0.5 / np.sqrt(support_q2.max())

    def func(x2, params):
        return params[0] * np.exp(-params[1] * x2)

    def res_func(params, x2_data, y_data,  penalty1=1e4, penalty2=1e2):
        test = func(x2_data, params)
        res = y_data - test
        p1 = abs(test) - test
        p2 = abs(res) - res
        return np.sqrt(np.mean(res ** 2)) + penalty1 * p1 + penalty2 * p2

    result = least_squares(res_func, p0, args=(support_q2, support_data), method="lm")
    return func(q_range ** 2, result.x)


def get_finite_data(data: DataSetLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determines min/max bounds of finite values in *data*.

    :param data: Dataset to test
    :returns: cropped data, lo/hi-data indexes, min/max values along axes
    """
    data = as_dataset(data)

    # Get Min/Max finite bounds
    lo_bounds = []
    hi_bounds = []
    for nd in range(data.ndim):
        lo = 0
        hi = data.shape[nd]
        finite = np.any(np.isfinite(data), axis=tuple(n for n in range(data.ndim) if n != nd))
        while not finite[lo]:
            lo += 1
        while not finite[hi - 1]:
            hi -= 1
        lo_bounds.append(lo)
        hi_bounds.append(hi)
    bounds = np.array(tuple(zip(lo_bounds, hi_bounds)), dtype=int)
    crop_data = data[tuple(slice(lo, hi) for lo, hi in bounds)].copy()

    minmax = []
    for nd, axis in enumerate(data.axes):
        axis = axis.try_linear()
        if isinstance(axis, LinearAxis):
            half = 0.5 * axis.scale
            lo = axis.offset + lo_bounds[nd] * axis.scale - half
            hi = axis.offset + hi_bounds[nd] * axis.scale - half
        else:
            samples = axis.range(data.shape[nd])
            half = 0.5 * np.mean(samples[1:] - samples[:-1])
            lo = samples[lo_bounds[nd]] - half
            hi = samples[hi_bounds[nd] - 1] + half
        minmax.append((lo, hi))
    minmax = np.array(minmax, dtype=float)
    return crop_data, bounds, minmax


class BruteForcePlot:
    """Show results of brute force evaluation"""
    def __init__(self, loss: DataSet, colorbar: bool = True, cmap: str = "viridis",
                 axis_labels: Optional[Iterable[str]] = None, rescale_loss: float = 1.0,
                 minloss: Optional[float] = None, maxloss: Optional[float] = None):
        self._loss = loss
        self._minloss = minloss
        self._maxloss = maxloss
        self._colorbar = colorbar
        self._cmap = cmap
        self._rescale_loss = rescale_loss

        if axis_labels is None:
            axis_labels = (format_label(a.name, a.unit) for a in loss.axes)
        self._axis_labels = tuple(axis_labels)

    def setup(self, fig, ax, param: Optional[Mapping[str, float]] = None):
        loss_data, bounds, minmax = get_finite_data(self._loss)
        loss_data *= self._rescale_loss

        if self._loss.ndim == 1:
            samples = self._loss.axes[0].range(self._loss.shape[0])
            ax.plot(samples[bounds[0, 0]:bounds[0, 1]], loss_data)
            ax.set_xlabel(self._axis_labels[0])
            ax.set_xlim(minmax[0, 0], minmax[0, 1])
            ax.set_ylim(self._minloss, self._maxloss)

            if param is not None:
                key = make_axis_name(self._loss.axes[0], 0)
                lines = ax.axvline(param[key], color='k')
            else:
                lines = None
        elif self._loss.ndim == 2:
            extent = [minmax[1, 0], minmax[1, 1], minmax[0, 1], minmax[0, 0]]
            aspect = abs(extent[1] - extent[0]) / abs(extent[2] - extent[3])
            vmax = self._maxloss if self._maxloss is not None else np.nanquantile(loss_data, 0.95)

            im = ax.imshow(loss_data, extent=extent, aspect=aspect, vmax=vmax, cmap=self._cmap, vmin=self._minloss)
            ax.set_xlabel(self._axis_labels[1])
            ax.set_ylabel(self._axis_labels[0])

            if param is not None:
                key_x = make_axis_name(self._loss.axes[1], 1)
                key_y = make_axis_name(self._loss.axes[0], 0)
                lines = ax.scatter([param[key_x]], [param[key_y]], marker='x', color='white')
            else:
                lines = None

            if self._colorbar:
                cax = inset_axes(ax, width='5%', height='100%', loc="center right", borderpad=-1.2)
                cbar = fig.colorbar(im, cax=cax)
                #cbar.set_ticks([])
        else:
            lines = None

        return fig, lines

    def update(self, state, param):
        fig, lines = state
        if self._loss.ndim == 1:
            if (param is not None) and (state is not None):
                key = make_axis_name(self._loss.axes[0], 0)
                value = param[key]
                lines.set_xdata([value, value])

        elif self._loss.ndim == 2:
            if (param is not None) and (state is not None):
                key_x = make_axis_name(self._loss.axes[1], 1)
                key_y = make_axis_name(self._loss.axes[0], 0)
                lines.set_offsets((param[key_x], param[key_y]))

        fig.canvas.draw_idle()


def split_name_and_unit(key: str) -> Tuple[str, str]:
    items = key.split('(', 1)
    if len(items) == 0:
        return items[0], ''
    if not items[1].endswith(')'):
        raise ValueError("No closing parenthesis found.")
    return items[0], items[1][:-1]


def escape_key(key):
    if "1/nm2" in key:
        stored_key = key.replace("1/nm2", "nm-2")
    elif "1/nm" in key:
        stored_key = key.replace("1/nm", "nm-1")
    else:
        stored_key = key
    return stored_key


def load_brute_force_new(bff: TemDataFile, param_keys: Iterable[str], prefix: str = "") \
        -> Tuple[Dict[str, np.ndarray], DataSet]:
    param_keys = set(param_keys)
    result_loss = bff.read(prefix + "loss")

    result_param = {key: np.zeros(result_loss.shape, dtype=float) for key in param_keys}
    missing_key = []
    for key in param_keys:
        stored_key = prefix + escape_key(key)
        if stored_key in bff.names():
            result_param[key][...] = bff.read(stored_key).get(copy=False)
        else:
            missing_key.append(key)

    if missing_key:
        print("Not identical parameter keys. Missing stored keys:")
        for key in missing_key:
            print(f"\t{key}")

    return result_param, result_loss


def load_brute_force(bff: TemDataFile, param_keys: Iterable[str], prefix: str = "") -> Tuple[np.ndarray, DataSet]:
    param_keys = set(param_keys)
    new_param, result_loss = load_brute_force_new(bff, param_keys, prefix)

    result_param = np.empty(result_loss.shape, dtype=object)
    for index in np.ndindex(*result_loss.shape):
        result_param[index] = {
            key: value[index] for key, value in new_param.items()
        }

    return result_param, result_loss


def store_brute_force(bff: TemDataFile, result_param: ArrayLike, result_loss: DataSet, prefix: str = "") -> NoReturn:
    bff.write_dataset(result_loss, name=prefix + "loss")
    for key in result_param.flat[0].keys():
        dataset = DataSet(shape=result_param.shape, dtype=float, axes=result_loss.axes,
                          metadata=CoreMetaData(result_loss.metadata, uuid=None, name=None))
        for index in np.ndindex(*result_param.shape):
            dataset[index] = result_param[index][key]
        stored_key = escape_key(key)
        bff.write_dataset(dataset, name=prefix + stored_key)


def get_loss_scale(data: ArrayLike, relative_losses: Optional[str]):
    if not relative_losses:
        return 1.0

    try:
        scale = float(relative_losses)
    except ValueError:
        if relative_losses == 'n':
            rescale_loss = 1.0 / data.size
        elif relative_losses == 'ss':
            rescale_loss = 1.0 / np.sum(data) ** 2
        elif relative_losses == 'ssn':
            print("Data-scale: ", np.sum(data) ** 2)
            rescale_loss = 1.0 / np.sum(data) ** 2 / data.size
        elif relative_losses == 'sstot':
            rescale_loss = 1.0 / np.sum((data - np.mean(data)) ** 2)
        else:
            raise ValueError(f"Unknown loss rescaling: {relative_losses}")
    else:
        rescale_loss = 1 / scale / data.size
    return rescale_loss


def find_minima(loss: BaseDataSet, border_margin: float = 0.0, kth=None):
    thickness_axis = loss.index_axis(name="thickness")
    thickness = loss.axes[thickness_axis].range(loss.shape[thickness_axis])
    thickness = np.expand_dims(thickness, axis=tuple(m for m in range(loss.ndim) if m != thickness_axis))

    border = np.zeros(loss.shape, dtype=bool)
    for n, a in enumerate(loss.axes):
        if not a.name.startswith("depth"):
            continue
        depth = a.range(loss.shape[n])
        depth = np.expand_dims(depth, axis=tuple(m for m in range(loss.ndim) if m != n))
        border |= (depth < border_margin)
        border |= (depth > (thickness - border_margin))
        del depth
    del thickness

    valid = np.isfinite(loss)
    if not np.any(valid):
        return []

    minima = valid.copy()
    minima[:, 1:] &= (~valid[:, :-1]) | (loss[:, :-1] >= loss[:, 1:])
    minima[:, :-1] &= (~valid[:, 1:]) | (loss[:, 1:] >= loss[:, :-1])
    minima[1:, :] &= (~valid[:-1, :]) | (loss[:-1, :] >= loss[1:, :])
    minima[:-1, :] &= (~valid[1:, :]) | (loss[1:, :] >= loss[:-1, :])
    minima[1:, 1:] &= ~valid[:-1, :-1] | (loss[:-1, :-1] >= loss[1:, 1:])
    minima[1:, :-1] &= ~valid[:-1, 1:] | (loss[:-1, 1:] >= loss[1:, :-1])
    minima[:-1, 1:] &= ~valid[1:, :-1] | (loss[1:, :-1] >= loss[:-1, 1:])
    minima[:-1, :-1] &= ~valid[1:, 1:] | (loss[1:, 1:] >= loss[:-1, :-1])
    minima &= ~border

    all_minima = np.argwhere(minima)
    all_minima_value = loss[np.where(minima)]
    all_minima_order = np.argsort(all_minima_value)[slice(0, kth)]
    return all_minima[all_minima_order], all_minima_value[all_minima_order]


def main(filename: str, experimental: DataSet, calculation: QXCalculation, initial_param: Dict[str, float],
         brute_force_parameters: Mapping[str, ParameterRange],
         interactive: bool = True,
         constrain_depth_thickness: bool = True,
         expunge_cached_brute_force: bool = True,
         experimental_rescale_x: float = 1.0, subtract_background: bool = False,
         nodes: int = 1, verbose: int = 1,
         optimized_parameters: Optional[Iterable[str]] = None,
         relative_losses: Optional[str] = None,
         mtf: Optional[Tuple] = None, save_final: bool = True, voltage: Optional[float] = None,
         cmap: str = "gray", bcmap: str = "viridis", with_brute: bool = False,
         width: Optional[float] = None, height: Optional[float] = None, dpi: Optional[float] = None,
         brute_force_labels: Optional[Iterable[str]] = None, sweep: bool = False,
         minloss: Optional[float] = None, maxloss: Optional[float] = None,
         line_sections: bool = False):
    import matplotlib
    rc_params = matplotlib.rc_params()
    dpi = float(dpi) if dpi is not None else rc_params["figure.dpi"]
    width = float(width) if width is not None else rc_params["figure.figsize"][0] * dpi
    height = float(height) if height is not None else rc_params["figure.figsize"][1] * dpi

    if subtract_background:
        q_range = experimental.axis_range(1) + initial_param.get("q_shift(1/nm)", 0.0)
        g_support, b_support = background_support(experimental, calculation.beamlist, q_dir_hkl=calculation.q_dir_hkl,
                                                  q_shift=initial_param.get("q_shift(1/nm)", 0.0))
        background = gaussian_background_constrained(q_range, g_support, b_support)

        if verbose > 1:
            plt.plot(q_range, np.mean(experimental.get(), axis=0))
            plt.scatter(g_support, b_support)
            plt.plot(q_range, background)
            plt.show()
    else:
        background = None

    pipeline = MatchQXPipeline(experimental, calculation, voltage, background=background, mtf=mtf)
    if optimized_parameters:
        optimized_parameters = tuple(optimized_parameters)
    else:
        optimized_parameters = ("tilt(1/nm)", "x_shift(nm)", "q_shift(1/nm)")
        if not mtf:
            optimized_parameters = optimized_parameters + ("q_blur(1/nm)",)

    all_initial_param = dict(pipeline.default_parameters())
    all_initial_param.update(initial_param)
    additional_plots = None
    additional_updates = None
    result_param = None
    result_loss = None

    rescale_loss = get_loss_scale(pipeline.experimental.get(copy=False), relative_losses)

    if brute_force_parameters:
        brute_force_axes = tuple(create_sub_sample_axes(calculation, key, value) for key, value in
                                 brute_force_parameters.items())
        brute_force_name = Path(filename).with_suffix('.brute_force.tdf')

        if not expunge_cached_brute_force:
            try:
                with TemDataFile(brute_force_name, 'r') as bff:
                    result_param, result_loss = load_brute_force(bff, all_initial_param)
            except (ValueError, KeyError, FileNotFoundError):
                pass

        linear_axes = tuple(a.try_linear() for a in brute_force_axes)
        if (result_param is None) or (result_loss is None) or (linear_axes != result_loss.axes):
            start = time.monotonic()

            if nodes != 1:
                import multiprocessing as mp
                mp.set_start_method("spawn")
                pool = mp.Pool(nodes)
            else:
                pool = None

            try:
                result_param, result_loss = brute_force(pipeline, all_initial_param, brute_force_axes,
                                                        optimized_parameters=optimized_parameters,
                                                        constrain_depth_thickness=constrain_depth_thickness,
                                                        verbose=verbose, pool=pool)
            except:
                if pool:
                    pool.close()
                raise

            elapsed = time.monotonic() - start

            metadata = CoreMetaData()
            metadata["experimental"] = experimental.metadata.ref()
            metadata["calculation"] = calculation.metadata.ref()
            metadata["optimized_param"] = optimized_parameters
            metadata["initial_param"] = all_initial_param
            metadata["subtract_background"] = subtract_background
            metadata["brute_force_time(s)"] = elapsed
            if verbose >= 1:
                print("Brute force time (s):", elapsed)

            result_loss = DataSet(data=result_loss, axes=linear_axes, metadata=metadata)
            with TemDataFile(brute_force_name, "w") as bff:
                store_brute_force(bff, result_param, result_loss)

        argmin = np.unravel_index(np.nanargmin(result_loss), result_loss.shape)
        optimal_param = result_param[argmin]

        fig, ax = plt.subplots()
        bf_plot = BruteForcePlot(result_loss, cmap=bcmap, colorbar=True, rescale_loss=rescale_loss,
                                 minloss=minloss, maxloss=maxloss)
        bf_state = bf_plot.setup(fig, ax, optimal_param)

        title = []
        if experimental.metadata.filename:
            title.append("Experimental: " + Path(experimental.metadata.filename).stem)
        if calculation.metadata.filename:
            title.append("Calculation: " + Path(calculation.metadata.filename).stem)

        plt.suptitle('\n'.join(title))

        brute_force_svgname = Path(filename).with_suffix('.brute_force.svg')
        fig.savefig(brute_force_svgname)

        # line plots through MSE map
        if line_sections:
            plt.close(fig)
            loss_data, bounds, minmax = get_finite_data(result_loss)
            print(bounds, minmax, loss_data.shape)
            for n, a in enumerate(result_loss.axes):
                _fig, _ax = plt.subplots()
                index = tuple(argmin[m] if m != n else slice(bounds[n, 0], bounds[n, 1]) for m in range(result_loss.ndim))
                data = result_loss.get(index) * rescale_loss
                mask = np.isnan(data)
                _ax.plot(result_loss.axis_range(n)[bounds[n, 0]:bounds[n, 1]], np.ma.masked_array(data, mask=mask))
                _ax.set_xlabel(format_label(a.name, a.unit))
                #_ax.set_xlim(minmax[n, 0], minmax[n, 1])
                _ax.set_ylabel("Mean squared error")
                _ax.set_ylim(minloss, maxloss)
                _ax.set_title(filename)

                section_name = Path(filename).with_suffix(f'.{a.name}-section.svg')
                _fig.savefig(section_name)
                plt.close(_fig)

        if with_brute or not interactive:
            plt.close(fig)
            del bf_plot
            del bf_state

            if with_brute:
                additional_plots = [BruteForcePlot(result_loss, cmap=bcmap, colorbar=True, axis_labels=brute_force_labels,
                                                   rescale_loss=rescale_loss, minloss=minloss, maxloss=maxloss)]
        else:
            additional_updates = [(bf_plot.update, bf_state)]
            fig.canvas.draw_idle()
    else:
        optimal_param, _ = optimize_parameter(pipeline, all_initial_param, optimized_parameters)

    plot = MatchQXPlot(pipeline, optimal_param, cmap=cmap, rescale_x=experimental_rescale_x, rescale_loss=rescale_loss,
                       optimized_parameters=optimized_parameters)

    if sweep and brute_force_parameters:
        ffmpeg_kw = {'fps': 3}
        depth_range = brute_force_axes[1].samples
        depth_sweep_name = Path(filename).with_suffix('.depth.mp4')
        if constrain_depth_thickness:
            depth_range = depth_range[depth_range <= optimal_param['thickness(nm)']]
        plot.save_sweep(depth_sweep_name, "depth(nm)", depth_range, ffmpeg_kw=ffmpeg_kw,
                        title=f"Thickness: {optimal_param['thickness(nm)']:.0f} nm, Depth: {{value:.0f}} nm",
                        figsize=(width / dpi, height / dpi), dpi=dpi, additional_plots=additional_plots)

        thickness_range = brute_force_axes[0].samples
        thickness_sweep_name = Path(filename).with_suffix('.thickness.mp4')
        if constrain_depth_thickness:
            thickness_range = thickness_range[thickness_range >= optimal_param['depth(nm)']]
        plot.save_sweep(thickness_sweep_name, "thickness(nm)", thickness_range, ffmpeg_kw=ffmpeg_kw,
                        title=f"Thickness: {{value:.0f}} nm, Depth: {optimal_param['depth(nm)']:.0f} nm",
                        figsize=(width / dpi, height / dpi), dpi=dpi, additional_plots=additional_plots)

    if interactive:
        plot.run_plot(figsize=(width / dpi, height / dpi), dpi=dpi, additional_plots=additional_plots, additional_updates=additional_updates)

    experimental_minmax = experimental.minmax()
    print("X-Range (nm):", (experimental_minmax[0][0] + plot.param["x_shift(nm)"]) / experimental_rescale_x,
                           (experimental_minmax[1][0] + plot.param["x_shift(nm)"]) / experimental_rescale_x)
    print("Q-Range (1/nm):", experimental_minmax[0][1] + plot.param["q_shift(1/nm)"], experimental_minmax[1][1] + plot.param["q_shift(1/nm)"])

    print("Parameters:")
    for k, v in plot.param.items():
        print(f"\t{k}: {v}")

    if (result_loss is not None) and (result_param is not None):
        print()
        print("Brute Force Minimum:")

        argmin = tuple(np.argmin(abs(result_loss.axis_range(nd) - plot.param[make_axis_name(axis, nd)]))
                       for nd, axis in enumerate(result_loss.axes))
        print("\tOptimum:", result_loss[argmin] * rescale_loss)

        bf_topk_index, bf_topk_loss = find_minima(result_loss, kth=3)
        for m in range(3):
            print(f"\tBF-Optimum #{m}: {bf_topk_loss[m] * rescale_loss} @",
                  *tuple(f"{a.name} = {float(a.offset + a.scale * bf_topk_index[m][n]):.1f} ({a.unit})"
                         for n, a in enumerate(result_loss.axes)))

        argmin = np.unravel_index(np.nanargmin(result_loss), result_loss.shape)
        optimal_param = result_param[argmin]

        for nd, axis in enumerate(result_loss.axes):
            key = make_axis_name(axis, nd)
            name, unit = split_name_and_unit(key)
            print()
            print(f"\t{key}: {result_loss.axis_range(nd)[argmin[nd]]:.2f}")

            lo_index = tuple(index if n != nd else index - 1 for n, index in enumerate(argmin))
            hi_index = tuple(index if n != nd else index + 1 for n, index in enumerate(argmin))
            try:
                lo = result_loss[lo_index] * rescale_loss
                hi = result_loss[hi_index] * rescale_loss
            except IndexError:
                pass
            else:
                deriv = (hi - lo) / (2.0 * axis.scale)
                print(f"\tdLoss/d{name}: {deriv:f}")
                deriv2 = (hi + lo - 2.0 * result_loss[argmin] * rescale_loss) / axis.scale ** 2
                print(f"\td2Loss/d{name}2: {deriv2:f}")

    if save_final:
        final_image_name = Path(filename).with_suffix('.result.svg')
        plot.save_plot(final_image_name, figsize=(width / dpi, height / dpi), dpi=dpi, additional_plots=additional_plots)


EPILOG = """Example parameter file (JSON file format):
{
    // File with experiment
    "experimental": "alpha1_10cm_4C_0002_stelle1.pos_ls2.diff_ls2.tdf", 
    
    // Name of dataset (if more than one dataset in file)
    "experimental_dataset": "mean",

    // Whether X / Q axes of experiment should be flipped (defaults to false)
    "experimental_flip_q": false,
    "experimental_flip_x": true,
    
    // Rescale X axis by this factor (defaults to 1.0)
    "experimental_rescale_x": 1.0,
    
    // File with calculation
    "calculation": "hw_gan5_c_0002.tdf",
    
    // Initial parameters                       
    "initial": {                                                             
        "semi_conv(1/nm)": 1.66,
        "tilt(1/nm)": 1.97,
        "depth(nm)": 50,
        "thickness(nm)": 126,
        "x_shift(nm)": -2.34,
        "q_blur(1/nm)": 0.166,
    },
    
    // Brute force parameters and their range (Min and Max, defaults to full calculation range)
    "brute_force_parameters": {
        "thickness(nm)": [120.0, 140.0],
        "depth(nm)": [40.0, 80.0],
    },
    
    // Labels for brute force plots
    // "brute_force_labels": [
    //    "$\\mathrm{Thickness}\\,t\\,(\\mathrm{nm})$",
    //    "$\\mathrm{Depth}\\,d\\,(\\mathrm{nm})$" 
    //],
    
    // Only brute force depths less equal to thickness (defaults to true)
    "constrain_depth_thickness": true,
    
    // Which parameters to automatically optimize
    "optimized_parameters": ["tilt(1/nm)", "x_shift(nm)", "q_shift(1/nm)", "q_blur(1/nm)"],
    
    // MTF to use along q direction
    "mtf": [["GAUSSIAN", 0.738033, 102.968], ["LORENTZIAN", 0.029029, 0.117078]],
    
    // Whether background is subtracted (defaults to False)
    "subtract_background": false,
    
    // Acceleration voltage in kV (otherwise taken from experiemnt)
    "acceleration_voltage(kV)": 300.0
}
"""


class ParsedParameters:
    """Class encapsulating parameters taken from parameter file."""

    brute_force_parameters: Dict[str, ParameterRange]
    calculation_path: Path
    experimental: DataSet
    experimental_rescale_x: float
    experimental_flip_q: bool
    experimental_flip_x: bool
    initial_param: Dict[str, float]
    subtract_background: bool
    optimized_parameters: Optional[Iterable[str]]
    constrain_depth_thickness: bool
    mtf: Any
    voltage: Optional[float]
    brute_force_labels: Optional[Iterable[str]]

    def __init__(self, param: Dict, enable_brute_force: Optional[bool] = None, base_dir: Optional[Path] = None, verbose: int = 0):
        base_dir = Path.cwd() if base_dir is None else Path(base_dir)

        if "brute_force_parameters" in param:
            # New interface
            self.brute_force_parameters = param["brute_force_parameters"]
        elif enable_brute_force or param.get("brute_force_enable", False):
            # Old interface (only thickness and depth)
            self.brute_force_parameters = {
                "thickness(nm)": param.get("brute_force_thickness_range"),
                "depth(nm)": param.get("brute_force_depth_range")
            }
        else:
            self.brute_force_parameters = {}
        if (enable_brute_force is not None) and not enable_brute_force:
            self.brute_force_parameters = {}

        experimental_path = base_dir / Path(param['experimental']).expanduser()
        experimental_name = param.get('experimental_name')
        with TemDataFile(experimental_path, "r") as experimental_file:
            uuids = experimental_file.uuids(cls=TemDataFile.CLASS_DATASET)
            if not experimental_name:
                if len(uuids) == 1:
                    experimental_name = uuids[0]
                elif 'mean' in experimental_file.names(cls=TemDataFile.CLASS_DATASET):
                    experimental_name = "mean"
                else:
                    raise ValueError("Multiple dataset in experimental file? Need parameter 'experimental_name'.")
            experimental = load_tdf(experimental_path, experimental_name)

        # Get experimental axes
        if args.strict:
            x_axis = LinearAxis(experimental.axes[1])
            q_axis = LinearAxis(experimental.axes[0])
        else:
            x_axis = LinearAxis(experimental.axes[1], context="POSITION", unit="nm")
            q_axis = LinearAxis(experimental.axes[0], context="DIFFRACTION", unit="1/nm")

        self.experimental_rescale_x = param.get("experimental_rescale_x", 1.0)
        self.experimental_flip_q = param.get('experimental_flip_q', False)
        self.experimental_flip_x = param.get('experimental_flip_x', False)

        # Get experimental data
        x_axis = LinearAxis(x_axis, scale=x_axis.scale * self.experimental_rescale_x, offset=x_axis.offset * self.experimental_rescale_x)
        experimental_axes = (x_axis, q_axis)
        experimental_data = experimental.get(copy=True).T
        if self.experimental_flip_q:  # Flip Q-axis
            experimental_data = experimental_data[:, ::-1]
        if self.experimental_flip_x:  # Flip X-axis
            experimental_data = experimental_data[::-1, :]

        margin_x = param.get("margin_x", None)
        if not isinstance(margin_x, list):
            margin_x = (margin_x,) * 2
        n_lo = 0
        n_hi = experimental_data.shape[0]
        if margin_x[0] is not None:
            n_lo += int(np.floor(float(margin_x[0] * self.experimental_rescale_x) / abs(x_axis.scale)))
        if margin_x[1] is not None:
            n_hi -= int(np.floor(float(margin_x[1] * self.experimental_rescale_x) / abs(x_axis.scale)))
        if n_lo >= experimental_data.shape[0]:
            n_lo = experimental_data.shape[0] - 1
        if n_hi <= n_lo:
            n_hi = n_lo + 1
        x_axis = LinearAxis(x_axis, offset=n_lo * x_axis.scale + x_axis.offset)
        experimental_data = experimental_data[n_lo:n_hi, :]

        self.initial_param = param.get('initial', {})
        self.initial_param.setdefault("x_shift(nm)", 0.0)
        self.initial_param["x_shift(nm)"] += n_lo * x_axis.scale

        self.experimental = DataSet(data=experimental_data.copy(), axes=experimental_axes, metadata=experimental.metadata)
        self.experimental.metadata["flipped_q"] = self.experimental_flip_q
        self.experimental.metadata["flipped_x"] = self.experimental_flip_x
        self.experimental.metadata["rescaled_x"] = self.experimental_rescale_x
        self.experimental.metadata["margin_x"] = margin_x

        self.voltage = param.get('acceleration_voltage(kV)')
        self.voltage = self.experimental.metadata["instrument"]["acceleration_voltage(kV)"] if self.voltage is None else float(self.voltage)

        if 'calculation' in param:
            if 'bloch_waves' in param:
                raise ValueError("The parameters 'calculation' and 'bloch_waves' are mutually exclusive.")

            calc_file = TemDataFile(base_dir / Path(param['calculation']).expanduser(), "r")
            amplitudes = calc_file.read("amplitudes", lazy=True)
            beamlist = calc_file.read(amplitudes.metadata["beamlist_uuid"])
            self.calculation = QXCalculationSelector(amplitudes, beamlist)
        elif 'bloch_waves' in param:
            bloch_waves_param = param["bloch_waves"]
            self.calculation = create_bloch_wave_calculation(bloch_waves_param, self.voltage, base_dir=base_dir, verbose=verbose)
        else:
            raise ValueError("Either the parameter 'calculation' or 'bloch_waves' must be given.")

        self.subtract_background = param.get('subtract_background', False)
        self.optimized_parameters = param.get('optimized_parameters')
        self.constrain_depth_thickness = param.get('constrain_depth_thickness', True)
        self.mtf = param.get('mtf')

        self.brute_force_labels = param.get('brute_force_labels')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Match QX plots",
                                     epilog=COPYRIGHT + "\n\nExample parameter file:\n" + EXAMPLE_PARAMETER_FILE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paramfile')
    parser.add_argument('-i', '--interactive', action='store_true', default=False, help="Match interactively")
    parser.add_argument('--with-brute', action='store_true', default=False, dest="with_brute", help="Show result of brute force plot in interactive plot")
    parser.add_argument('-r', '--relative', type=str, nargs='?', action='store', default=None, const="ss", help="Show losses rescaled to 'ss', or 'sstot'")
    parser.add_argument('-n', '--nodes', type=int, default=1, help="Nodes to use for brute forcing")
    parser.add_argument('-b', '--brute', action='store_true', default=None, dest="brute", help="Enable brute force search (override param file)")
    parser.add_argument('--no-brute', action='store_false', default=None, dest="brute",
                        help="Disable brute force search (override param file)")
    parser.add_argument('--no-final', action='store_false', default=True, dest="final",
                        help="Disable saving of final plot")
    parser.add_argument('-c', '--cmap', type=str, default="gray", help="Colormap for intensity plots")
    parser.add_argument('--bcmap', type=str, default="viridis", help="Colormap for Brute Force match")
    parser.add_argument('--width', type=float, default=None, help="Width of final match plot")
    parser.add_argument('--height', type=float, default=None, help="Height of final match plot")
    parser.add_argument('--dpi', type=float, default=None, help="DPI of final image")
    parser.add_argument('-l', '--line-sections', dest='line_sections', action='store_true', default=False, help="Store profile lines through brute force minimum")
    parser.add_argument('-x', '--expunge', action='store_true', default=False, help="Expunge brute force cache")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbose mode")
    parser.add_argument('--no-strict', dest="strict", action='store_false', default=True, help="No strict axis checking on experiment")
    parser.add_argument('-s', '--sweep', action='store_true', default=False, help="Store brute force sweeps")
    parser.add_argument('--minloss', type=float, default=None, help="Minimum loss to show")
    parser.add_argument('--maxloss', type=float, default=None, help="Maximum loss to show")
    args = parser.parse_args()

    param_file = Path(args.paramfile)
    with open(param_file, 'rt') as file:
        param_source = file.read()
    param = decode_json(param_source, allow_comments=True, allow_trailing_commas=True)

    pp = ParsedParameters(param, enable_brute_force=args.brute, base_dir=param_file.parent, verbose=args.verbose)
    main(param_file.stem, pp.experimental, pp.calculation, initial_param=pp.initial_param,
         interactive=args.interactive, verbose=args.verbose, nodes=args.nodes,
         subtract_background=pp.subtract_background,
         optimized_parameters=pp.optimized_parameters,
         experimental_rescale_x=pp.experimental_rescale_x,
         brute_force_parameters=pp.brute_force_parameters,
         constrain_depth_thickness=pp.constrain_depth_thickness,
         mtf=pp.mtf, save_final=args.final, voltage=pp.voltage,
         relative_losses=args.relative,
         with_brute=args.with_brute, expunge_cached_brute_force=args.expunge,
         cmap=args.cmap, bcmap=args.bcmap, width=args.width, height=args.height, dpi=args.dpi,
         brute_force_labels=pp.brute_force_labels, sweep=args.sweep,
         minloss=args.minloss, maxloss=args.maxloss, line_sections=args.line_sections)

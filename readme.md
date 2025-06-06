match_qx: Scripts for generation and evaluation of (q,x)-Plots
==============================================================

This repository contain tools to process 4D STEM images into (q,x)-plots and compare these
plots with simulations. These tools were used in the following article:

    T. Niermann, L. Niermann, M. Lehmann
    "Three dimensional classiﬁcation of dislocations from single projections"
    Nature Communication 15 (2024) 1356
    DOI: https://doi.org/10.1038/s41467-024-45642-z
    
If you use these tools, please cite the above article. For further questions, please
contact Tore Niermann (tore.niermann@tu-berlin.de).

Requirements (other versions of the packages might also work, but are not tested):

* Python >= 3.10
* numpy >= 1.26
* matplotlib >= 3.8
* pyctem == 4.3.1


Command line scripts
--------------------

The scripts must be run in the "match_qx" sub-directory, since they import other
modules from that directory (alternatively, add the "match_qx" sub-directory to your _PYTHON_PATH_).

All scripts require _JSON_ parameter files as argument. Within the parameter file, they allow
_JavaScript_-style comments and trailing commas in _array_ or _object_ element enumerations.   

Be aware that these scripts when run in interactive mode, requires interactive matplotlib figures. Make sure
you run them with a _matplotlib_ backend, which supports this.

### linescan_pos_3d.py

Script to create profile within the positional dimension of a 4D STEM dataset.

The script requires a _JSON_ parameter file as argument. Use the "-h" option to obtain a commented
version of the parameter file and a description of further command line options. The "-l" option
can be used to run an interactive session.

### linescan_diff_2d.py

Script to create profile within the diffraction dimensions of a dataset already reduced by _linescan_pos_3d.py_.

The script requires a _JSON_ parameter file as argument. Use the "-h" option to obtain a commented
version of the parameter file and a description of further command line options. The "-l" option
can be used to run an interactive session.

### match.py

Script to match experimental and calculated (q,x)-Plots. For matching three parameter sets are important:

* __initial__: these are the assumed parameters for matching. They are also used as starting point of the minimization 
* __optimized_parameters__: This set of parameters is optimized using a minimization method   
* __brute_force_parameters__: These can be used to thoroughly test sets of parameters. Within this sense
    this is a brute force optimization, as the whole parameter space is searched. If a brute force search is run, the 
    set of _optimized_parameters_ is optimized for each point in the brute force search space. Initial and other
    parameters are taken from _initial_.

Supported parameters are identified by their name and unit, however for each parameter only the
described unit is excepted. Beside the following parameters, parameters can also refer to additional dimensions present in the calculation, 
like "thickness(nm)". Additionally, the following parameters are supported by the script:
* "semi_conv(1/nm)": Semiconvergence angle (as reciprocal length) of the probe. Default: 1.0 1/nm.
* "tilt(1/nm)": Incident angle of the central beam (as reciprocal length). Default: 0.0 1/nm.
* "q_blur(1/nm)": RMS-Width of an optional Gaussian blur along the Q-axis. A value of 0.0 refers to no blurring (default). 
* "q_shift(1/nm)": Shift of the experimental data with respect to calculated data along the Q-axis. Default 0.0 1/nm.  
* "x_shift(nm)": Shift of the experimental data with respect to calculated data along the X-axis. Default 0.0 nm.
* "defocus(nm)": Additional defocus of the calculated data. Default: 0.0 nm. 
* "curvature(1/nm2)": Additional curvature of the calculated data. Default: 0.0 1/nm^2.

The script requires a _JSON_ parameter file as argument. Use the "-h" option to obtain a commented
version of the parameter file. The "-i" option can be used to run an interactive session. For details for the
specification of the detector's modulation transfer function ("mtf" parameter), 
see https://holoaverage.readthedocs.io/en/latest/parameters.html#modulation-transfer-function

_match_qx_ offers the following command line options:

* -h, --help: Display help text and example parameter file
* -i, --interactive: Run interactive matching session 
* --with-brute: Include brute force plot into result window (instead of separate window)
* -r, --relative=method: How the errors (losses) are displayed. By default the absolute squared difference is shown. 
    Supported _methods_ are:
    * "n": normalize losses by number of points in QX-Plot, i.e. display the mean squared error. This option is also used if no _method_ is given
    * "ss": normalize losses by sum of experiment intensity in QX-Plot squared
    * "ssn": normalize losses by sum of experiment intensity squared and number of points in QX-Plot
    * "sstot": normalize losses by total variance in experimental intensity
    * number: normalize losses by number and number of points  
* -n, --nodes=number: How many processes (nodes) to use for brute force matching (defaults to 1)
* -b, --brute, --no-brute: Enforce or disable brute force search
* --no-final: Disable saving of final match as plot
* -c, --cmap=name: Matplotlib name of colormap used for display of intensity plots (defaults to gray)
* --bcmap=name: Matplotlib name of colormap used for display of brute force searches (defaults to viridis)
* --width=number: Width of final match plot in inch
* --height=number: Height of final match plot in inch
* --dpi=number: DPI used for match plot
* -l, --line-sections: Additionally store line sections through minimum in brute force plot
* -x, --expunge: Expunge brute force search cache
* -v, --verbose: Be verbose, use multiple times for more output
* --no-strict: Be tolerant with axes metadata in experiment
* -s, --sweep: Store brute force sweeps as movies. This currently only works if "depth(nm)" and "thickness(nm)" are the 
    brute force parameters (in this order) and requires _ffmpeg_ to be properly installed and configured with matplotlib.  
* --minloss=value: Minimum value on loss axis
* --maxloss=value: Maximum value on loss axis

Examples
--------

Exemplary parameter files are in directory "param-files". 

Acknowledgements
----------------

This software results from projects funded by the Deutsche Forschungsgemeinschaft
(DFG, German Research Foundation) within projects 492463633 and 403371556.

License
-------

Copyright (C) 2024 Tore Niermann (tore.niermann@tu-berlin.de)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/#GPL>.

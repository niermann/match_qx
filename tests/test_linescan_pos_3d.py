import numpy as np
from src.linescan_pos_3d import main
from pyctem.iolib import TemDataFile
from pathlib import Path


USE_CACHED_RESULTS = True
cached_mib_and_tiff = USE_CACHED_RESULTS
cached_unicorn = USE_CACHED_RESULTS
cached_raw_data = USE_CACHED_RESULTS
cached_raw_data2 = USE_CACHED_RESULTS
cached_raw_data3 = USE_CACHED_RESULTS


# Diff-scale (1/nm)/px [Voltage] [DetectAngleNumber]
# See GrandArm/Calibrations page on EM-Wiki
CAMERA_LENGTH_CALIBRATION = {
    200: {
        9: 0.031204,
        10: 0.029216,
        11: 0.027185,
        12: 0.025514,
        13: 0.020053,
        14: 0.016502,
        15: 0.013921,
        16: 0.010480,
        17: 0.005418,
        18: 0.004303
    },
    300: {
        0: 0.3139505,
        1: 0.2619862,
        2: 0.2162937,
        3: 0.1853549,
        4: 0.1429535,
        5: 0.0840736,
        6: 0.0730317,
        7: 0.0587714,
        8: 0.0490568,
        9: 0.0417861,
        10: 0.0385048,
        11: 0.0362937,
        12: 0.0339306,
        13: 0.0265384,
        14: 0.0216729,
        15: 0.0182095,
        16: 0.0139851,
        17: 0.0069423,
        18: 0.0055985,
    }
}

MIB_AND_TIFF_PARAMETERS = {
    "image_file": "files/STEM_20220718_1149_38_ADF1_ImagePanel1.tif",
    "stem4d_file": "files/alpha1_10cm_4C_0002_stelle1.mib",
    "pos0": [201.6, 224.3],
    "pos1": [204.1, 109.8],
    "posw": 49.5,
    "pos_binsize": 1,
    "camera_length_calibration": CAMERA_LENGTH_CALIBRATION,
    "calibration_correction": 4.101
}

UNICORN_PARAMETERS = {
    "stem4d_file": "files/Stelle1_200_1M-1.tdf",
    "output_file": "Stelle1_200_1M-1.pos.tdf",
    "pos0": [176.6, 135.3],
    "pos1": [127.5, 194.5],
    "posw": 21.6,
    "pos_binsize": 1,
    "camera_length_calibration": CAMERA_LENGTH_CALIBRATION,
    "calibration_correction": 4.101
}

RAW_DATA_PARAMETERS = {
    "stem4d_file": "files/raw_data_dislocation_A.hdf5",
    "output_file": "dislocation_A.pos.tdf",
    "image_scale": 0.78125,
    "diff_scale": 0.074677,
    "pos0": [201.6, 224.3],
    "pos1": [204.1, 109.8],
    "posw": 49.5,
    "pos_binsize": 1
}

RAW_DATA2_PARAMETERS = {
    "stem4d_file": "files/raw_data_dislocation_B.hdf5",
    "output_file": "dislocation_B.pos.tdf",
    "image_scale": 0.78125,
    "diff_scale": 0.074677,
    "pos0": [86.0, 121.7],
    "pos1": [35.2, 51.5],
    "posw": 13.0,
    "pos_binsize": 1
}

RAW_DATA3_PARAMETERS = {
    "stem4d_file": "files/raw_data_aluminum.hdf5",
    "output_file": "aluminum.pos.tdf",
    "image_scale": 0.78125,
    "diff_scale": 0.074677,
    "pos0": [176.6, 135.3],
    "pos1": [127.5, 194.5],
    "posw": 21.6,
    "pos_binsize": 1
}


def prepare_mib_and_tiff():
    global cached_mib_and_tiff
    output_file = Path("alpha1_10cm_4C_0002_stelle1.pos.tdf")
    if not cached_mib_and_tiff or not output_file.is_file():
        if not Path(MIB_AND_TIFF_PARAMETERS["stem4d_file"]).is_file() or \
                not Path(MIB_AND_TIFF_PARAMETERS["image_file"]).is_file():
            return None     # Skip test due to missing files

        cached_mib_and_tiff = True
        main(MIB_AND_TIFF_PARAMETERS, "pos", image_method="sum")
    return output_file


def prepare_unicorn():
    global cached_unicorn
    output_file = Path(UNICORN_PARAMETERS["output_file"])
    if not cached_unicorn or not output_file.is_file():
        if not Path(UNICORN_PARAMETERS["stem4d_file"]).is_file():
            return None  # Skip test due to missing files

        cached_unicorn = True
        main(UNICORN_PARAMETERS, "overriden-param-file")
    return output_file


def prepare_raw_data():
    global cached_raw_data
    output_file = Path(RAW_DATA_PARAMETERS["output_file"])
    if not cached_raw_data or not output_file.is_file():
        if not Path(RAW_DATA_PARAMETERS["stem4d_file"]).is_file():
            return None  # Skip test due to missing files

        cached_raw_data = True
        main(RAW_DATA_PARAMETERS, "pos", lazy=True)
    return output_file


def prepare_raw_data2():
    global cached_raw_data2
    output_file = Path(RAW_DATA2_PARAMETERS["output_file"])
    if not cached_raw_data2 or not output_file.is_file():
        if not Path(RAW_DATA2_PARAMETERS["stem4d_file"]).is_file():
            return None  # Skip test due to missing files

        cached_raw_data2 = True
        main(RAW_DATA2_PARAMETERS, "pos", lazy=True)
    return output_file


def prepare_raw_data3():
    global cached_raw_data3
    output_file = Path(RAW_DATA3_PARAMETERS["output_file"])
    if not cached_raw_data3 or not output_file.is_file():
        if not Path(RAW_DATA3_PARAMETERS["stem4d_file"]).is_file():
            return None  # Skip test due to missing files

        cached_raw_data3 = True
        main(RAW_DATA3_PARAMETERS, "pos", lazy=True, image_method="sum")
    return output_file


def test_mib_and_tiff():
    output_file = prepare_mib_and_tiff()
    if not output_file:
        print("Skipped test 'test_mib_and_tiff' due to missing data files.")
        return

    with TemDataFile(output_file, "r") as file:
        # Test image
        image = file.read("image")
        assert image.shape == (256, 256)
        assert image.axes[0].unit == "nm"
        assert abs(image.axes[0].scale - 0.78125) < 1e-5
        assert image.axes[1].unit == "nm"
        assert abs(image.axes[1].scale - 0.78125) < 1e-5

        # Test linescan
        linescan = file.read("mean")
        assert linescan.shape == (115, 256, 256)
        assert linescan.axes[0].unit == "nm"
        assert abs(linescan.axes[0].scale - 0.78125) < 1e-5
        assert linescan.axes[1].unit == "1/nm"
        assert abs(linescan.axes[1].scale - 0.074677) < 1e-5
        assert linescan.axes[2].unit == "1/nm"
        assert abs(linescan.axes[2].scale - 0.074677) < 1e-5


def test_unicorn():
    output_file = prepare_unicorn()
    if not output_file:
        print("Skipped test 'test_unicorn' due to missing data files.")
        return

    with TemDataFile(output_file, "r") as file:
        # Test image
        image = file.read("image")
        assert image.shape == (256, 256)
        assert image.axes[0].unit == "nm"
        assert abs(image.axes[0].scale - 0.78125) < 1e-5
        assert image.axes[1].unit == "nm"
        assert abs(image.axes[1].scale - 0.78125) < 1e-5

        # Test linescan
        linescan = file.read("mean")
        assert linescan.shape == (77, 256, 256)
        assert linescan.axes[0].unit == "nm"
        assert abs(linescan.axes[0].scale - 0.78125) < 1e-5
        assert linescan.axes[1].unit == "1/nm"
        assert abs(linescan.axes[1].scale - 0.074677) < 1e-5
        assert linescan.axes[2].unit == "1/nm"
        assert abs(linescan.axes[2].scale - 0.074677) < 1e-5


def test_raw_data():
    output_file = prepare_raw_data()
    if not output_file:
        print("Skipped test 'test_raw_data' due to missing data files.")
        return

    with TemDataFile(output_file, "r") as file:
        # Test image
        image = file.read("image")
        assert image.shape == (256, 256)
        assert image.axes[0].unit == "nm"
        assert abs(image.axes[0].scale - 0.78125) < 1e-5
        assert image.axes[1].unit == "nm"
        assert abs(image.axes[1].scale - 0.78125) < 1e-5

        # Test linescan
        linescan = file.read("mean")
        assert linescan.shape == (115, 256, 256)
        assert linescan.axes[0].unit == "nm"
        assert abs(linescan.axes[0].scale - 0.78125) < 1e-5
        assert linescan.axes[1].unit == "1/nm"
        assert abs(linescan.axes[1].scale - 0.074677) < 1e-5
        assert linescan.axes[2].unit == "1/nm"
        assert abs(linescan.axes[2].scale - 0.074677) < 1e-5


def test_raw_data2():
    output_file = prepare_raw_data2()
    if not output_file:
        print("Skipped test 'test_raw_data2' due to missing data files.")
        return

    with TemDataFile(output_file, "r") as file:
        # Test image
        image = file.read("image")
        assert image.shape == (256, 256)
        assert image.axes[0].unit == "nm"
        assert abs(image.axes[0].scale - 0.78125) < 1e-5
        assert image.axes[1].unit == "nm"
        assert abs(image.axes[1].scale - 0.78125) < 1e-5

        # Test linescan
        linescan = file.read("mean")
        assert linescan.shape == (87, 256, 256)
        assert linescan.axes[0].unit == "nm"
        assert abs(linescan.axes[0].scale - 0.78125) < 1e-5
        assert linescan.axes[1].unit == "1/nm"
        assert abs(linescan.axes[1].scale - 0.074677) < 1e-5
        assert linescan.axes[2].unit == "1/nm"
        assert abs(linescan.axes[2].scale - 0.074677) < 1e-5


def test_raw_data3():
    output_file = prepare_raw_data3()
    if not output_file:
        print("Skipped test 'test_raw_data3' due to missing data files.")
        return

    with TemDataFile(output_file, "r") as file:
        # Test image
        image = file.read("image")
        assert image.shape == (256, 256)
        assert image.axes[0].unit == "nm"
        assert abs(image.axes[0].scale - 0.78125) < 1e-5
        assert image.axes[1].unit == "nm"
        assert abs(image.axes[1].scale - 0.78125) < 1e-5

        # Test linescan
        linescan = file.read("mean")
        assert linescan.shape == (77, 256, 256)
        assert linescan.axes[0].unit == "nm"
        assert abs(linescan.axes[0].scale - 0.78125) < 1e-5
        assert linescan.axes[1].unit == "1/nm"
        assert abs(linescan.axes[1].scale - 0.074677) < 1e-5
        assert linescan.axes[2].unit == "1/nm"
        assert abs(linescan.axes[2].scale - 0.074677) < 1e-5


def test_dislocation_a():
    original_filename = prepare_mib_and_tiff()
    published_filename = prepare_raw_data()
    if not original_filename or not published_filename:
        print("Skipped test 'test_dislocation_a' due to missing data files.")
        return

    with (TemDataFile(original_filename, "r") as original_file,
          TemDataFile(published_filename,"r") as published_file):
        original_image = original_file.read("image")
        published_image = published_file.read("image")
        assert np.allclose(original_image.get(copy=False), published_image.get(copy=False))

        original_mean = original_file.read("mean")
        published_mean = published_file.read("mean")
        assert np.allclose(original_mean.get(copy=False), published_mean.get(copy=False))


def test_aluminum():
    original_filename = prepare_unicorn()
    published_filename = prepare_raw_data3()
    if not original_filename or not published_filename:
        print("Skipped test 'test_aluminum' due to missing data files.")
        return

    with (TemDataFile(original_filename, "r") as original_file,
          TemDataFile(published_filename,"r") as published_file):
        original_image = original_file.read("image")
        published_image = published_file.read("image")
        assert np.allclose(original_image.get(copy=False), published_image.get(copy=False))

        original_mean = original_file.read("mean")
        published_mean = published_file.read("mean")
        assert np.allclose(original_mean.get(copy=False), published_mean.get(copy=False))

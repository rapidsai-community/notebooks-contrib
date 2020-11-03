import json
import glob
import multiprocessing
import pynvml
import os
import tarfile
import urllib

# Global variables

# Links to mortgage data files
MORTGAGE_YEARLY_1GB_SPLITS_URL = "https://rapidsai-data.s3.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_yearly/"
MORTGAGE_YEARLY_2GB_SPLITS_URL = "https://rapidsai-data.s3.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_yearly_2gb/"


def get_data(data_dir, start_year, end_year, use_1GB_splits):
    """
    Utility to download and extract mortgage data to specied data_dir.
    Only specific years of data between `start_year` and `end_year` will be downloaded
    to the specified directory
    """
    if use_1GB_splits:
        data_url = MORTGAGE_YEARLY_1GB_SPLITS_URL
    else:
        data_url = MORTGAGE_YEARLY_2GB_SPLITS_URL
    for year in range(start_year, end_year + 1):
        if not os.path.isfile(data_dir + "acq/Acquisition_" + str(year) + "Q4.txt"):
            print(f"Downloading data for year {year}")
            filename = "mortgage_" + str(year)
            filename += "_1gb.tgz" if use_1GB_splits else "_2GB.tgz"
            urllib.request.urlretrieve(data_url + filename, data_dir + filename)
            print(f"Download complete")
            print(f"Decompressing and extracting data")

            tar = tarfile.open(data_dir + filename, mode="r:gz")
            tar.extractall(path=data_dir)
            tar.close()
            print(f"Done extracting year {year}")

    if not os.path.isfile(data_dir + "names.csv"):
        urllib.request.urlretrieve(data_url + "names.csv", data_dir + "names.csv")


def _read_data_spec(filename=os.path.dirname(__file__) + "/Data_Spec.json"):
    """
    Read the Data_Spec json
    """
    with open(filename) as f:
        data_spec = json.load(f)

    try:
        spec_list = data_spec["SpecInfo"]
    except KeyError:
        raise ValueError(f"SpecInfo missing in Data spec file: {filename}")
    return spec_list


def determine_dataset(total_mem, min_mem, part_count=None):
    """
    Determine params and dataset to use
    based on Data spec sheet and available memory
    """
    start_year = None  # start year for etl proessing
    end_year = None  # end year for etl processing (inclusive)

    use_1GB_splits = True
    if min_mem >= 31.5e9:
        use_1GB_splits = False

    spec_list = _read_data_spec()
    # Assumption that spec_list has elements with mem_requirement
    # in Descending order

    # TODO: Code duplication. Consolidate into one
    if part_count:
        part_count = int(part_count)
        for i, spec in enumerate(spec_list):
            spec_part_count = (
                spec["Part_Count"][1] if use_1GB_splits else spec["Part_Count"][0]
            )
            if part_count > spec_part_count:
                start_year = (
                    spec_list[i - 1]["Start_Year"] if i > 0 else spec["Start_Year"]
                )
                end_year = spec_list[i - 1]["End_Year"] if i > 0 else spec["End_Year"]
                break
        if not start_year:
            start_year = spec_list[-1]["Start_Year"]
            end_year = spec_list[-1]["End_Year"]

    else:
        for spec in spec_list:
            spec_part_count = (
                spec["Part_Count"][1] if use_1GB_splits else spec["Part_Count"][0]
            )
            if total_mem >= spec["Total_Mem"]:
                start_year = spec["Start_Year"]
                end_year = spec["End_Year"]
                part_count = spec_part_count
                break

    return (start_year, end_year, part_count, use_1GB_splits)


def memory_info():
    """
    Assumes identical GPUs in a node
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    pynvml.nvmlShutdown()
    return gpu_mem


def get_num_files(start_year, end_year, perf_dir):
    """
    Get number of files to read given start_year
    end_year and path to performance files
    """
    count = 0
    for year in range(start_year, end_year + 1):
        count += len(glob.glob(perf_dir + f"/*{year}*"))
    return count


def get_cpu_cores():
    return multiprocessing.cpu_count()

"""
Libera Data Processing Template

This template demonstrates the standard workflow for processing Libera science data.
It follows the 8-step process outlined in the Libera processing guidelines and serves
as a starting point for scientists to implement their specific algorithms.

Steps implemented:
1. Read and use the Input Manifest
2. Read and store input data from manifest files
3. Calculate science data variables (PLACEHOLDER - INSERT YOUR SCIENCE HERE)
4. Store science data with metadata for NetCDF formatting
5. Write data to output folder with proper timestamps
6. Create output manifest
7. Add data files to output manifest
8. Write output manifest to output folder

Author: [Your Name]
Date: [Date]
Version: 1.0.0
"""

# Standard library imports
import argparse
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

# Third-party imports
import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, S3Path

# Libera utilities imports
from libera_utils.io.filenaming import (
    ManifestFilename,
)
from libera_utils import Manifest
from libera_utils import smart_open
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.logutil import configure_task_logging

# Configure logging
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI application."""
    now = datetime.now(UTC)
    args = parse_cli_args()
    configure_task_logging(f"example_algorithm_{now}")

    logger.debug(f"CLI args: {args}")
    # Extract the manifest file path from command line arguments
    if not args.manifest:
        raise ValueError("Manifest file path must be provided as a command line argument")
    manifest_path = AnyPath(args.manifest)

    output_manifest_path = algorithm(manifest_path)
    logger.info(f"Processing complete. Output manifest: {output_manifest_path}")


def parse_cli_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing the manifest file path and options
    """
    parser = argparse.ArgumentParser(
        prog="libera-l2-unfiltered-radiances",
        description="Libera science data processing for unfiltering radiances"
    )

    parser.add_argument(
        "manifest",
        type=str,
        help="Absolute path to the input manifest file"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    return parser.parse_args()


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """
    Main processing algorithm implementing the 8-step Libera processing workflow.

    This function orchestrates the entire data processing pipeline from reading
    input manifests to writing output manifests with processed data products.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file containing data file information

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file

    Raises
    ------
    Exception
        If any file cannot be opened or processed
    """

    # Step 1: Read and use the Input Manifest
    logger.info("Step 1: Reading the input manifest file")
    input_manifest = Manifest.from_file(manifest_path)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")

    # Step 2: Read and store ALL input data from manifest files
    logger.info("Step 2: Reading all input data from manifest files")
    all_input_data = read_all_input_data(input_manifest)

    # Step 3: Calculate science data variables (YOUR SCIENCE GOES HERE)
    logger.info("Step 3: Calculating science data variables")
    processed_data = calculate_science_data(all_input_data)

    # Set the output location to write to in the output dropbox
    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Steps 4-5: Store data with metadata and write to output folder
    logger.info("Steps 4-5: Creating and writing data product")
    output_data_file_path = create_and_write_data_product(
        processed_data=processed_data,
        output_path=dropbox_path
    )

    # Step 6: Create output manifest
    logger.info("Step 6: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    # Step 7: Add data files to output manifest
    logger.info("Step 7: Adding data files to output manifest")
    output_manifest.add_files(output_data_file_path)

    # Step 8: Write output manifest to output dropbox folder
    logger.info("Step 8: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath


def read_all_input_data(input_manifest: Manifest) -> dict[str, xr.Dataset]:
    """
    Step 2: Read and store ALL input data from manifest files.

    This function opens and validates all input NetCDF files from the manifest
    and stores them in a dictionary keyed by filename.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest containing file information

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary with filenames as keys and loaded datasets as values

    Raises
    ------
    Exception
        If any file cannot be opened or is invalid
    """
    logger.info("Step 2: Reading all input data from manifest files")
    all_data = {}

    for i, file_info in enumerate(input_manifest.files):
        logger.info(f"Reading file {i + 1}/{len(input_manifest.files)}: {file_info.filename}")

        try:
            with smart_open(file_info.filename) as file_handle:
                # Load the NetCDF dataset
                dataset = xr.open_dataset(file_handle)
                all_data[file_info.filename] = dataset
                logger.info(f"Successfully loaded dataset with variables: {list(dataset.variables)}")

        except Exception as e:
            logger.error(f"Failed to open file {file_info.filename}: {e}")
            raise

    logger.info(f"Successfully loaded {len(all_data)} datasets")
    return all_data


def calculate_science_data(all_input_data: dict[str, xr.Dataset]) -> dict:
    """
    Step 3: Calculate your specific science data variables.

    *** THIS IS WHERE YOUR SCIENCE ALGORITHM GOES ***

    This function validates that all input files can be opened and creates
    constant arrays for output variables with a fixed size of 123 samples.

    Parameters
    ----------
    all_input_data : dict[str, xr.Dataset]
        Dictionary of all loaded input datasets keyed by filename

    Returns
    -------
    dict
        Dictionary containing processed science variables with sequential constants
        Keys should match the variable names in your product definition YAML

    Notes
    -----
    This function:
    - Validates that all input files were successfully opened
    - Creates constant arrays of size 123 for all output variables
    - Uses sequential values across all variables (Option B)
    """
    logger.info("Step 3: Calculating science data variables")

    # Validate all input files were opened successfully
    logger.info(f"Validating {len(all_input_data)} input datasets")
    for filename, dataset in all_input_data.items():
        if dataset is None:
            raise ValueError(f"Failed to load dataset from {filename}")
        logger.debug(f"Dataset from {filename} has variables: {list(dataset.variables)}")

    # Define output size
    output_size = 123
    logger.info(f"Creating output arrays of size {output_size}")

    # Create time stamps (current time repeated)
    time_data = np.full((output_size,), datetime(2028, 1, 1))

    # Create sequential constant arrays for each variable
    calibrated_radiance = np.full_like(time_data, 123)

    radiance_uncertainty = np.full_like(time_data, 0.01)

    latitude = np.full_like(time_data, 45.0)  # Example constant latitude

    longitude = np.full_like(time_data, -75.0)  # Example constant longitude

    quality_flags = np.full_like(time_data, 0)  # Example constant quality flags

    # Package results
    processed_data = {
        'radiometer_time': time_data,
        'unfiltered_radiance': calibrated_radiance,
        'latitude': latitude,
        'longitude': longitude,
        'quality_flags': quality_flags
    }

    logger.info("Science calculations completed")
    logger.info(f"Created variables: {list(processed_data.keys())}")
    logger.info(f"Each variable has {output_size} samples")

    return processed_data


def create_and_write_data_product(
        processed_data: dict,
        output_path: str | Path | S3Path
) -> AnyPath:
    """
    Steps 4-5: Store science data with metadata and write to output folder.

    This function creates a properly formatted NetCDF data product using the
    DataProductConfig class from libera_utils, which handles metadata management
    and file formatting according to Libera standards.

    Parameters
    ----------
    processed_data : dict
        Dictionary of processed science variables from calculate_science_data()
    output_path : str | Path | S3Path
        The location to write the output file

    Returns
    -------
    AnyPath
        Path to the written data product file

    Notes
    -----
    This function:
    - Loads the product definition from the data folder
    - Creates a DataProductConfig object with proper metadata
    - Adds the processed science data to each variable
    - Writes the data product with proper Libera naming conventions
    """
    logger.info("Steps 4-5: Creating and writing data product")

    # Get the product definition file path from the data folder
    script_dir = Path(__file__).parent
    product_config_file = script_dir / "l2-unfiltered-radiance-product-definition.yml"

    if not product_config_file.exists():
        raise FileNotFoundError(
            f"Product definition file not found: {product_config_file}\n"
            "Please ensure example_product_definition.yml is in the data folder."
        )

    # Step 4: Create DataProductConfig with metadata
    logger.info("Step 4: Setting up data product configuration")

    # Add processed data to each variable defined in the configuration
    logger.info("Adding processed data to variables")

    # Step 5: Write the data product file
    logger.info("Step 5: Writing data product to environment specified file")
    logger.info(f"Saving to {output_path}")

    # Set the time range for the data product
    current_time = datetime.now(UTC)
    start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = current_time.replace(hour=23, minute=59, second=59, microsecond=999999)

    output_file_path = write_libera_data_product(
        data_product_definition=product_config_file,
        data=processed_data,
        output_path=output_path,
        time_variable="radiometer_time",
    )

    logger.info(f"Data product written to: {output_file_path}")

    return output_file_path


if __name__ == "__main__":
    main()
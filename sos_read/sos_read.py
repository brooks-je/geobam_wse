"""SoS Read - Downloads Confluence SoS file from S3 bucket."""

import logging
import pathlib
import sys

import boto3
import botocore

logging.basicConfig(
    format="%(asctime)s - %(module)s - : %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)


def download_sos(bucket_key, sos_filepath):
    """Downloads the SoS to the TMP_PATH and returns the full path reference
    to the downloaded SoS file.

    Arguments
    bucket_key: str - SoS bucket plus key, "confluence-dev1-sos/unconstrained/0001"
    sos_filepath: str - Full path to download SoS file to locally
    """

    bucket_key_split = bucket_key.split('/')
    bucket = bucket_key_split[0]
    sos_filename = pathlib.Path(sos_filepath).name
    if len(bucket_key_split) == 1:
        key = sos_filename
    else:
        key = f"{'/'.join(bucket_key_split[1:])}/{sos_filename}"

    s3 = boto3.client("s3")
    try:
        response = s3.download_file(bucket, key, sos_filepath)
        logging.info("Downloaded: %s from %s/%s.", sos_filepath, bucket, key)
    except botocore.exceptions.ClientError as e:
        logging.error(f"Could not download file: %s from %s/%s.", sos_filepath, bucket, key)
        logging.info(e)
        logging.info("Cannot proceed without previous results. Exiting program.")
        sys.exit(1)

library(reticulate)

use_python("/path/to/python/executable")
source_python("/path/to/sos-read/sos_read.py")

bucket_key = "confluence-dev1-sos/unconstrained/0000"
sos_filepath = "/tmp/na_sword_v16_SOS_priors.nc"
download_sos(bucket_key, sos_filepath)

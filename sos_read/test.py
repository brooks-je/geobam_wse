from sos_read import download_sos

bucket_key = "confluence-dev1-sos/unconstrained/0000"
sos_filepath = "/tmp/na_sword_v16_SOS_priors.nc"

download_sos(bucket_key, sos_filepath)

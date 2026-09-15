using PyCall

pushfirst!(pyimport("sys")."path", "")   # Load sos_read script
sos_read = pyimport("sos_read")

bucket_key = "confluence-dev1-sos/unconstrained/0000"
sos_filepath = "/tmp/na_sword_v16_SOS_priors.nc"
sos_read.download_sos(bucket_key, sos_filepath)

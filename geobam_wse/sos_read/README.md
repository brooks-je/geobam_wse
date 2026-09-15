# sos-read

The SoS-Read module downloads a specified SoS file from an S3 bucket to local storage.

Function arguments

- bucket_key: String value that stores the bucket plus key for the SoS file to download. Example: `confluence-dev1-sos/unconstrained/0001`
- sos_filepath: String value that stores the location that the SoS will be downloaded to. Example: `/tmp/na_sword_v16_SOS_priors.nc`

## requires

`boto3` to be installed for Python to be able to download file from S3 bucket.

## execution

PYTHON
```python
from sos_read import download_sos

bucket_key = "confluence-dev1-sos/unconstrained/0000"
sos_filepath = "/tmp/na_sword_v16_SOS_priors.nc"

download_sos(bucket_key, sos_filepath)
```


---
JULIA
```julia
using PyCall

pushfirst!(pyimport("sys")."path", "")   # Load sos_read script
sos_read = pyimport("sos_read")

bucket_key = "confluence-dev1-sos/unconstrained/0000"
sos_filepath = "/tmp/na_sword_v16_SOS_priors.nc"
download_sos(bucket_key, sos_filepath)
```

Requires `PyCall` package:

```julia
ENV["PYTHON"] = "... path of the python executable ..."
Pkg.add(PackageSpec(name="PyCall", rev="master"))
Pkg.build("PyCall")
```

---
R
```R
library(reticulate)

use_python("/path/to/python/executable")
source_python("/path/to/sos-read/sos_read.py")

bucket_key = "confluence-dev1-sos/unconstrained/0000"
sos_filepath = "/tmp/na_sword_v16_SOS_priors.nc"
download_sos(bucket_key, sos_filepath)
```

Requires `reticulate` package:

```R
install.packages("reticulate")
```
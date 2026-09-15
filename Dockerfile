# STAGE 0 - base image and system packages
FROM python:3.11-slim AS stage0
RUN echo "America/New_York" > /etc/timezone \
	&& apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		build-essential \
		tzdata \
	&& rm -rf /var/lib/apt/lists/*

# STAGE 1 - Python packages
# netCDF4 ships manylinux wheels that bundle libhdf5 and libnetcdf, so no
# apt-level NetCDF development packages are needed.
FROM stage0 AS stage1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r /tmp/requirements.txt

# STAGE 2 - I/O directories and application code
FROM stage1 AS stage2
RUN mkdir -p /app/data/input /app/data/output
COPY ./run_geobam_wse.py /app/
COPY ./geobam_wse /app/geobam_wse
COPY ./sos_read /app/sos_read/

# STAGE 3 - execute algorithm
FROM stage2 AS stage3
WORKDIR /app
LABEL version="2.0" \
	description="Containerized geobam_wse algorithm (NumPyro)." \
	"confluence.contact"="ntebaldi@umass.edu" \
	"algorithm.contact"="cjgleason@umass.edu,cbrinkerhoff@umass.edu"
ENTRYPOINT [ "python3", "/app/run_geobam_wse.py" ]

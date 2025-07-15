set -x
docker build . -t intelanalytics/hunyuan3d-2.1:0701 --build-arg https_proxy=http://proxy.iil.intel.com:911 --build-arg http_proxy=http://proxy.iil.intel.com:911
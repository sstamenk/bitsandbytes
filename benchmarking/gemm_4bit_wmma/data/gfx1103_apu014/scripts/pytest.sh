#!/bin/bash
# pytest.sh <srcdir> <log>
. ~/bnbtune/env.sh
cd ~/bnbtune/$1
PYTHONPATH=$PWD python -m pytest tests/test_functional.py tests/test_linear4bit.py -k "(4bit or Linear4bit or gemv) and not fsdp and not test_4bit_quant_large" -q -p no:cacheprovider -rf > ~/bnbtune/$2 2>&1
echo "pytest rc=$?" >> ~/bnbtune/$2

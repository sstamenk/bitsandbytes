cd ~/bnbtune
. ~/bnbtune/venv/bin/activate
export PATH=$HOME/.local/bin:$HOME/bnbtune/bin:$PATH UV_CACHE_DIR=$HOME/bnbtune/uvcache
export ROCM_PATH=$(rocm-sdk path --root)

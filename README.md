notes on install:
1. Activate conda env
`conda activate <env_name>`
2. Install dependencies. Note: the python whl for the zed interface has a dependency on numpy2.0, which is not compatible with the rest of this dependency stack. To deal with this. We did the following procedure:
```
pip install -e .  # install skill dependencies
pip install /usr/local/zed/pyzed-5.0-cp310-cp310-linux_x86_64.whl  # replace with path to python wheel on your system if need be
# previous line will complain about needing numpy2.0, and probably upgrade numpy forcibly, so we now reinstall this directory
pip install -e .  # force environment back to numpy<2
pip install --no-deps /usr/local/zed/pyzed-5.0-cp310-cp310-linux_x86_64.whl  # reinstall zed bindings, but without upgrading numpy
```
3. Setup M2T2 server. Follow the instructions at: https://github.com/williamshen-nz/M2T2
3. Setup FoundatioStereo server. Follow the instructions at: https://github.com/williamshen-nz/FoundationStereo

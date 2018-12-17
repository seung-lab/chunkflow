
# test inference and donate script
```
python inference_and_donate.py --output_dir=gs://neuroglancer/golden_v0/affinitymap_jwu --exchange_dir=gs://neuroglancer/golden_v0/affinitymap_jwu/exchange --output_block_start 84 576 576 --output_block_size 84 576 576 --input_dir=gs://neuroglancer/golden_v0/image --model_path=./frameworks/rsunet.py --net_path=./frameworks/model200000.chkpt --patch_size 32 256 256 
```

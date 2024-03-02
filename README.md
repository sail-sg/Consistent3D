## Consistent3D: Towards Consistent High-Fidelity Text-to-3D Generation with Deterministic Sampling Prior (CVPR 2024)

<div  align="center">    
<img src="./load/assets.gif" alt="results" width="400">
</div>

This is an official PyTorch implementation of Consistent3D. See the paper [here](https://arxiv.org/abs/2401.09050). If you find our Consistent3D helpful or heuristic to your projects, please cite this paper and also star this repository. Thanks!

```bibtex
@misc{wu2024consistent3d,
      title={Consistent3D: Towards Consistent High-Fidelity Text-to-3D Generation with Deterministic Sampling Prior}, 
      author={Zike Wu and Pan Zhou and Xuanyu Yi and Xiaoding Yuan and Hanwang Zhang},
      year={2024},
      eprint={2401.09050},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Acknowledgement: This repo is based on the following amazing project: [threestudio](https://github.com/threestudio-project/threestudio).

## Requirements
All experiments were conducted using PyTorch 1.13.0, CUDA 11.7.1, and CuDNN 8.5.0. We strongly recommend to use the [provided Dockerfile](./docker/Dockerfile) to build an image to reproduce our experiments.
Please check out [threestudio](https://github.com/threestudio-project/threestudio) repository for alternative installation.

## Quickstart
```.bash
# --------- Coarse NeRF Optimization Stage --------- #
python launch.py --config configs/consistency-coarse.yaml --train --gpu 0 \
system.prompt_processor.prompt="a delicious hamburger"

# ------------- Mesh Refinement Stage -------------- #
# Geometry Refinement (Optional)
python launch.py --config configs/consistency-refine.yaml --train --gpu 0 \
system.prompt_processor.prompt="a delicious hamburger" \
system.geometry_convert_from=path/to/coarse/dir/ckpts/last.ckpt

# Texture Refinement
python launch.py --config configs/consistency-texture.yaml --train --gpu 0 \
system.prompt_processor.prompt="a delicious hamburger" \
system.geometry_convert_from=path/to/refine/dir/ckpts/last.ckpt


```

## (Optation) Mesh Exploration
```.bash
# this uses default mesh-exporter configurations which exports obj+mtl
python launch.py --config configs/consistency-texture.yaml --export --gpu 0 resume=path/to/refine/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter
# specify system.exporter.fmt=obj to get obj with vertex colors
# you may also add system.exporter.save_uv=false to accelerate the process, suitable for a quick peek of the result
python launch.py --config configs/consistency-texture.yaml --export --gpu 0 resume=path/to/refine/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj

```

## Tips for Improving Quality
Here are some tips that may help you improve the generation quality:
- **Increase batch size.** Large batch sizes help convergence and improve the 3D consistency of the geometry. 
- **Train longer.** This helps if you can already obtain reasonable results and would like to enhance the details. See examples in [configs](./configs) with `-long` suffix.
- **Try different seeds.** Try different seed by setting `seed=N` when you suffer from the Janus face problem. 




Please check out [threestudio](https://github.com/threestudio-project/threestudio) repository for more tips.

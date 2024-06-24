# SAM TensorRT pipeline

## Start container

```bash
nvidia-docker run --rm -it  -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.04-py3 /bin/bash
```

## Install dependencies

```bash
pip install -e .
pip install onnxruntime onnx_graphsurgeon colored polygraphy tensorrt --upgrade
```

## Download checkpoints

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Export onnx, build engines and infer

```bash
mkdir onnx engine
python scripts/trt_inference.py --checkpoint=sam_vit_h_4b8939.pth --input-image=images/apples.jpg --mode point --benchmark --visualize --output-image=output.png
```

One can provide onnx directories and engine directories with `--onnx-dir` and `--engine-dir`. `--visualize` to save output image to the path provided with `--output-image`.

Use `--mode all-masks` or `--mode point` to switch between the modes. The 1st one will sample many points and segment different thing on the image. The 2nd one segments a region given a single point. Use `--point-cord 300,300` to pass coordinate of a query point.

Specify `--benchmark` for performance profiling. One can specify `--torch` for PyTorch backend instead of TRT one.
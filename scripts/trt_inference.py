#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# li
# mitations under the License.
#
from collections import OrderedDict
import os
import time
from copy import copy

import numpy as np
import torch
from torch.nn import functional as F
import argparse
import cv2

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (engine_from_bytes, 
                                   engine_from_network, 
                                   network_from_onnx_path, save_engine)
from polygraphy.backend.trt import CreateConfig, Profile
# from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.backend.trt import SetLayerPrecisions
from polygraphy import cuda
from cuda import cudart
import onnx
import onnx_graphsurgeon as gs

import tensorrt as trt
import nvtx

from torchvision.ops.boxes import batched_nms, box_area

from segment_anything.utils.onnx import SamOnnxModel
from segment_anything.utils.transforms import ResizeLongestSide

from segment_anything.utils.amg import (build_all_layer_point_grids, generate_crop_boxes, 
                                        MaskData, calculate_stability_score,
                                        batch_iterator,
                                        batched_mask_to_box,
                                        uncrop_masks,
                                        uncrop_boxes_xyxy,
                                        box_xyxy_to_xywh,
                                        uncrop_points,
                                        uncrop_masks)

from segment_anything import sam_model_registry, SamPredictor

from utils import (resize_longest_image_size, 
                   visualize_point_and_mask, visualize_multiple_masks,
                   coords)

# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# trt_util.TRT_LOGGER = TRT_LOGGER

torch_to_numpy_dtype_dict = {
    torch.bool : bool,
    torch.uint8 : np.uint8,
    torch.int8 : np.int8,
    torch.int16 : np.int16,
    torch.int32 : np.int32,
    torch.int64 : np.int64,
    torch.float16 : np.float16,
    torch.float32 : np.float32,
    torch.float64 : np.float64,
    torch.complex64 : np.complex64,
    torch.complex128 : np.complex128
}

numpy_to_torch_dtype_dict = {
    bool : torch.bool,
    np.uint8 : torch.uint8,
    np.int8 : torch.int8,
    np.int16 : torch.int16,
    np.int32 : torch.int32,
    np.int64 : torch.int64,
    np.float16 : torch.float16,
    np.float32 : torch.float32,
    np.float64 : torch.float64,
    np.complex64 : torch.complex64,
    np.complex128 : torch.complex128
}

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix=''):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_inputs(self, keep, names=None):
        self.graph.inputs = [self.graph.inputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.inputs[i].name = name

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph))
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def get_layer_norm_names(self):
        names = []
        for node in self.graph.nodes:
            if node.op == "ReduceMean" and node.o().op == "Sub":
                # Add ReduceMean
                names.append(node.name)
                # Add Sub
                names.append(node.o().name)
                # Add Pow
                names.append(node.o().o().name)
                # Add ReduceMean
                names.append(node.o().o().o().name)
                # Add Add
                names.append(node.o().o().o().o().name)
                # Add Sqrt
                names.append(node.o().o().o().o().o().name)
                # Add Div
                names.append(node.o().o().o().o().o().o().name)
                # Add Mul
                names.append(node.o().o().o().o().o().o().o().name)
                # Add Add
                names.append(node.o().o().o().o().o().o().o().o().name)
        return names

    def get_gemm_names(self):
        names = []
        for node in self.graph.nodes:
            if node.op in ("MatMul", "Conv", "ConvTranspose"):
                names.append(node.name)
        return names

class ModelProcessing(object):
    def __init__(self, name='', onnx_dir='', engine_dir='', torch_model=None, 
                 device='cuda', embed_dim=4, image_embedding_size=(1024, 1024), 
                 seed=1997, verbose=False, memory_pool_gbs=32, opset=16, use_cuda_graph=False):
        self.verbose = verbose
        self.device = device
        self.torch_model = torch_model
        self.seed = seed
        self.name = name

        self.opset = opset

        self.mem_pool_gbs = memory_pool_gbs

        self.raw_onnx_path = os.path.join(onnx_dir, name + '.onnx')
        self.optimized_onnx_path = os.path.join(onnx_dir, name + '.opt.onnx')
        self.engine_path = os.path.join(engine_dir, name + '.engine')

        self.engine = None
        self.context = None
        self.cuda_graph_instance = None # cuda graph
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.use_cuda_graph = use_cuda_graph

        self.torch_input = None

    def export_onnx(self, export_params=True, force_export=False):
        if self.torch_model is None:
            raise ValueError("Torch model is not set. You need to set torch model using set_torch_model.")

        with torch.no_grad(), torch.autocast('cuda'):
            if os.path.exists(self.raw_onnx_path) and not force_export:
                return
            print(f"Exporting {self.name} model to {self.raw_onnx_path}")
            input = self._get_default_input()
            input_names = self._get_input_names()
            output_names = self._get_output_names()
            dynamic_axes = self._get_dynamic_axes()
            torch.onnx.export(self.torch_model,
                              input,
                              self.raw_onnx_path,
                              export_params=export_params,
                              opset_version=self.opset,
                              verbose=False,
                              do_constant_folding=True,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)

            graph = gs.import_onnx(onnx.load(self.raw_onnx_path))
            self._make_input_multidimensional(graph, "has_mask_input")
            clean_graph = self._add_cast_before_outputs(graph=graph,  output_names=["iou_predictions", "low_res_masks" ])
            # Save the modified ONNX model
            onnx.save(gs.export_onnx(clean_graph), self.raw_onnx_path)

    def _make_input_multidimensional(self, graph, input_name:str):
        # try to get the mask input 
        has_mask_input = [x for x in graph.inputs if x.name==input_name]
        if has_mask_input:
            has_mask_input= has_mask_input[0]
            # change it to 4 dims 
            has_mask_input.shape = [1, 1, 1, 1]

    def _add_cast_before_outputs(self, graph, output_names:list, dtype=onnx.TensorProto.FLOAT):

        assert isinstance(output_names, list)
        # Process each output name in the provided list
        for output_name in output_names:
            # Get the specific output to be casted
            target_output = [x for x in graph.outputs if x.name == output_name]
            if target_output:
                target_output = target_output[0]

                # Make a new tensor variable, that will be using fp32
                new_output = gs.Variable(name=f"new_{output_name}", shape=target_output.shape, dtype=dtype)

                # Make a new node, which performs the type casting
                new_casting_node = gs.Node("Cast", f"cast_{output_name}", attrs={"to": dtype}, inputs=[target_output], outputs=[new_output])
                graph.nodes.append(new_casting_node)

                # Adjust the outputs of the graph
                graph.outputs.append(new_output)
                graph.outputs.remove(target_output)

                # Remove the tensor
                target_output.name = f"{output_name}_outdated"

                # Rename the new output to the original name
                new_output.name = output_name

        # Cleanup and sort the graph
        cleaned_graph = graph.cleanup().toposort()

        return cleaned_graph

    def _optimize(self, model, minimal_optimization):
        opt = Optimizer(model, verbose=self.verbose)
        opt.info('{}: original'.format(self.name))
        opt.fold_constants()
        opt.info('{}: fold constants'.format(self.name))
        opt.infer_shapes()
        opt.info('{}: shape inference'.format(self.name))

        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info('{}: final'.format(self.name))
        return opt_onnx_graph

    def _get_layer_norm_names(self):
        model = onnx.load(self.optimized_onnx_path)
        opt = Optimizer(model, verbose=self.verbose)
        return opt.get_layer_norm_names()

    def _get_gemm_names(self):
        model = onnx.load(self.optimized_onnx_path)
        opt = Optimizer(model, verbose=self.verbose)
        return opt.get_gemm_names()

    def optimize_onnx(self, force_optimize=False, minimal_optimization=False):
        if os.path.exists(self.optimized_onnx_path):
            if not force_optimize:
                return
        raw_model = onnx.load(self.raw_onnx_path)
        print("Optimizing model {}".format(self.name))
        optimized_model = self._optimize(raw_model, minimal_optimization=minimal_optimization)
        try:
            onnx.checker.check_model(optimized_model)
        except:
            onnx.save_model(optimized_model, self.optimized_onnx_path, save_as_external_data=True, 
                            all_tensors_to_one_file=False, size_threshold=1024, convert_attribute=False)
            return
        onnx.save(optimized_model, self.optimized_onnx_path)

    def _get_default_input(self):
        return tuple(self._get_default_param_dict().values())
    
    def _get_input_names(self):
        return list(self._get_default_param_dict().keys())

    def _get_default_device_view(self):
        params = self._get_default_param_dict()
        dv_params = {}
        for k, v in params.items():
            dv_params[k] = cuda.DeviceView(ptr=v.data_ptr(), shape=v.shape, dtype=torch_to_numpy_dtype_dict[v.dtype])
        return dv_params
    
    def _get_dynamic_axes_dims(self):
        raise NotImplemented("_get_dynamic_axes_dims must be implemented in the derived class")
    
    def _get_output_names(self):
        raise NotImplemented("_get_output_names must be implemented in the derived class")

    def _get_default_param_dict(self):
        raise NotImplemented("_get_default_input must be implemented in the derived class")
    
    def _get_dynamic_axes(self):
        raise NotImplemented("_get_dynamic_axes must be implemented in the derived class")

    def get_shape_dict(self, batch, points):
        raise NotImplemented("get_shape_dict must be implemented in the derived class")
    
    def build_engine(self, timing_cache=None, tf32=True, force_build=False):
        if os.path.exists(self.engine_path) and not force_build:
            return
        
        p = Profile()
        input_profile = self._get_dynamic_axes_dims()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        print(f"Building TensorRT engine for {self.optimized_onnx_path}: {self.engine_path}")
        network =  network_from_onnx_path(self.optimized_onnx_path)
        # Force layer norms to _
        ln_names = self._get_layer_norm_names()
        ln_dict = {k: trt.DataType.FLOAT for k in ln_names}
        network = SetLayerPrecisions(network, ln_dict)
        engine = engine_from_network(network,
                                     config=CreateConfig(tf32=tf32, 
                                     profiles=[p], precision_constraints='obey',
                                     memory_pool_limits={trt.MemoryPoolType.WORKSPACE: self.mem_pool_gbs << 30}, 
                                     load_timing_cache=timing_cache), 
                                     save_timing_cache=timing_cache)

        save_engine(engine, path=self.engine_path)
        self.engine = engine

    def load_engine(self):
        if self.engine is None:
            print(f"Loading TensorRT engine: {self.engine_path}")
            self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate_engine(self, batch, points):
        self.context = self.engine.create_execution_context()
        self._allocate_buffers(shape_dict=self.get_shape_dict(batch, points))

    def infer_engine(self, stream, param_dict=None):
        for name, buf in param_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if self.use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        return self.tensors


    def infer_torch(self):
        with torch.no_grad(), torch.autocast('cuda'):
            inputs = self._get_default_input()
            output = self.torch_model(*inputs)
        return output

    def _allocate_buffers(self, shape_dict=None):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=self.device)
            self.tensors[binding] = tensor


class Encoder(ModelProcessing):
    def __init__(self, **kwargs):
        super().__init__(name="encoder", **kwargs)
        self.image_h = 1024
        self.image_w = 1024
        self.in_channels = 3

    def _get_dynamic_axes_dims(self):
        dynamic_axes = {
        }
        return dynamic_axes

    def _get_dynamic_axes(self):
        dynamic_axes = {
        }
        return dynamic_axes

    def get_shape_dict(self, batch, points):
        return {
            "x": (1, self.in_channels, self.image_h, self.image_w),
            "image_embeddings": (1, self.embed_dim, *self.image_embedding_size)
        }

    def _get_output_names(self):
        return ["image_embeddings"]

    def _get_default_param_dict(self):
        torch.manual_seed(self.seed)
        dummy_inputs = {
            "x": torch.randn(1, self.in_channels, self.image_h, self.image_w, dtype=torch.float).to(self.device),
        }
        return dummy_inputs

class Decoder(ModelProcessing):
    def __init__(self, **kwargs):
        super().__init__(name="decoder", **kwargs)
        self.min_points = 1
        self.opt_points = 1
        self.max_points = 4

        self.min_batch = 1
        self.opt_batch = 512
        self.max_batch = 512

    def _get_dynamic_axes_dims(self):
        dynamic_axes_dims = {}
        # dynamic_axes_dims = {
        #     "point_coords": [(self.min_batch, self.min_points, 2), 
        #                      (self.opt_batch, self.opt_points, 2), 
        #                      (self.max_batch, self.max_points, 2)],
        #     "point_labels": [(self.min_batch, self.min_points), 
        #                      (self.opt_batch, self.opt_points), 
        #                      (self.max_batch, self.max_points)],
        # }
        
        return dynamic_axes_dims

    def _get_dynamic_axes(self):
        dynamic_axes = {}
        # dynamic_axes = {
        #     "point_coords": {0: "B", 1: "num_points"},
        #     "point_labels": {0: "B", 1: "num_points"},
        #     "iou_predictions": {0: "B"},
        #     "low_res_masks": {0: "B"},
        # }
        return dynamic_axes

    def get_shape_dict(self, batch, points):
        embed_dim = self.embed_dim
        embed_size = self.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]

        return {
            "image_embeddings": (1, embed_dim, *embed_size),
            "point_coords": (batch, points, 2),
            "point_labels": (batch, points),
            "mask_input": (1, 1, *mask_input_size),
            "has_mask_input": (1, 1, 1, 1),
            "iou_predictions": (batch, 4), # up to 4 masks per point
            "low_res_masks": (batch, 4, *mask_input_size), # up to 4 masks per point
        }

    def _get_output_names(self):
        return ["iou_predictions", "low_res_masks"]

    def _get_default_param_dict(self):
        torch.manual_seed(self.seed)
        embed_dim = self.embed_dim
        embed_size = self.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]

        target_dtype = torch.float

        dummy_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=target_dtype).to(self.device),
            "point_coords": torch.randint(
                low=0, high=1024, size=(1, 2, 2), dtype=target_dtype
            ).to(self.device),
            "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=target_dtype).to(self.device),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=target_dtype).to(self.device),
            "has_mask_input": torch.randn(1, dtype=target_dtype).to(self.device),
        }
        return dummy_inputs

def benchmark(name, infer, command_args, infer_args):
    for _ in range(command_args.num_warmup_runs):
        infer(*infer_args)

    torch.cuda.synchronize()
    e2e_tic = time.perf_counter()

    runs = command_args.runs
    for _ in range(runs):
        infer(*infer_args)

    torch.cuda.synchronize()
    e2e_toc = time.perf_counter()

    print(f"{name} elapsed {(e2e_toc - e2e_tic)*1000./runs} ms")

class Pipeline():
    def __init__(self, args) -> None:
        self.device = 'cuda'
        self.sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(self.device)

        self.points_per_side = 32
        self.crop_n_layers = 0
        self.crop_n_points_downscale_factor = 1
        self.crop_overlap_ratio = 512 / 1500
        self.points_per_batch = 512
        self.pred_iou_thresh = 0.88
        self.stability_score_thresh = 0.95
        self.stability_score_offset = 1.0
        self.box_nms_thresh = 0.7

        self.decoder_torch = SamOnnxModel(
            model=self.sam,
            return_single_mask=False,
            use_stability_score=False,
            return_extra_metrics=False,
        ).to(self.device)

        self.models = {}

        params = {
            'onnx_dir': args.onnx_dir,
            'engine_dir': args.engine_dir,
            'device': self.device,
            'seed': args.seed,
            'verbose': args.verbose,
            'memory_pool_gbs': args.memory_pool_gbs,
            'opset': args.opset,
            'embed_dim': self.sam.prompt_encoder.embed_dim,
            'image_embedding_size': self.sam.prompt_encoder.image_embedding_size,
            'use_cuda_graph': args.use_cuda_graph
        }

        self.models['encoder'] = Encoder(torch_model=self.sam.image_encoder, **params)
        self.models['decoder'] = Decoder(torch_model=self.decoder_torch, **params)

        if args.mode == "point":
            num_of_query_points = 1
            points_per_query = 2
        elif args.mode == "all-masks":
            num_of_query_points = 512
            points_per_query = 1
        else:
            raise RuntimeError("Incorrect mode argument {}".format(args.mode))

        for m in self.models.values():
            m.export_onnx(force_export=args.force_onnx_export, export_params=not args.not_export_params)
            m.optimize_onnx(force_optimize=args.force_onnx_optimize,
                            minimal_optimization=args.onnx_minimal_optimization)

            m.build_engine(timing_cache=args.timing_cache, force_build=args.force_engine_build)
            m.load_engine()
            m.activate_engine(num_of_query_points, points_per_query)

        self.point_grids = build_all_layer_point_grids(self.points_per_side, self.crop_n_layers,
                                                        self.crop_n_points_downscale_factor)

        self.stream = cuda.Stream()
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.nvtx_profile = args.nvtx_profile
        self.torch = args.torch

    def _encode(self, image):
        if self.torch:
            predictor = SamPredictor(self.sam)

            predictor.set_image(image)
            image_embedding = predictor.get_image_embedding()
        else:
            if self.nvtx_profile:
                nvtx_image_preprocess = nvtx.start_range(message='image_preprocess', color='pink')

            if "RGB" != self.sam.image_format:
                print("Spplied RGB image, but expected {}. Refromat in-place."
                      " For better perf, consider reformatting before".format(self.sam.image_format))
                image = image[..., ::-1]

            image = torch.as_tensor(image, device=self.device)
            image = image.permute(2, 0, 1).contiguous()[None, :, :, :]
            original_dtype = image.dtype
            image = pipeline.transform.apply_image_torch(image.to(dtype=torch.float16)).to(dtype=original_dtype)

            input_image = self.sam.preprocess(image)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_image_preprocess)

            if self.nvtx_profile:
                nvtx_encoder = nvtx.start_range(message='encoder', color='red')

            features = self.models['encoder'].infer_engine(self.stream, {"x": input_image})

            if self.nvtx_profile:
                nvtx.end_range(nvtx_encoder)

            image_embedding = features['image_embeddings']

        return image_embedding

    def _mask_postprocessing(self, masks: torch.Tensor, orig_im_size) -> torch.Tensor:
        orig_im_size_torch = torch.tensor(orig_im_size, device=self.device)
        masks = F.interpolate(
            masks,
            size=(self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = resize_longest_image_size(orig_im_size_torch, self.sam.image_encoder.img_size).to(torch.int64)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

        orig_im_size_torch = orig_im_size_torch.to(torch.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks

    def _decode(self, embeddings, input_point, input_label):
        dtype = torch.float32
        if self.torch:
            dtype = torch.float32

        embeddings = embeddings.to(dtype=dtype)
        coord = input_point.to(dtype=dtype)
        label = input_label.to(dtype=dtype)

        mask_input = torch.zeros(1, 1, 4 * self.sam.prompt_encoder.image_embedding_size[0], 
                                 4 * self.sam.prompt_encoder.image_embedding_size[1], device=self.device, dtype=dtype)
        has_mask_input = torch.zeros((1, 1, 1, 1), dtype=dtype, device=self.device)

        if self.torch:
            scores, masks  = self.decoder_torch(embeddings, coord, label, mask_input, has_mask_input)
        else:
            params = {
                "image_embeddings": embeddings,
                "point_coords": coord,
                "point_labels": label,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
            }

            if self.nvtx_profile:
                nvtx_decoder = nvtx.start_range(message='decoder', color='blue')

            outputs = self.models['decoder'].infer_engine(self.stream, params)

            if self.nvtx_profile:
                nvtx.end_range(nvtx_decoder)

            masks = outputs['low_res_masks']
            scores = outputs['iou_predictions']

        if self.nvtx_profile:
            nvtx_postprocessing = nvtx.start_range(message='postprocessing', color='green')

        scores = scores.to(dtype=torch.float32)
        masks = masks.to(dtype=torch.float32)

        if self.nvtx_profile:
            nvtx.end_range(nvtx_postprocessing)

        return scores, masks

    def _run(self, image):
        raise NotImplemented("_run must be implemented in the derived class")

    def run(self, image, args):
        kwargs = {}
        if isinstance(self, SegmentPointPipeline):
            kwargs["input_point"] = torch.as_tensor(np.array([[*args.coord[0]]]), device=pipeline.device, dtype=torch.float32)
            kwargs["input_label"] = torch.as_tensor(np.array([1]), device=pipeline.device, dtype=torch.float32)
        return self._run(image, **kwargs)

class SegmentPointPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.zero_point = torch.zeros(1, 2, device=self.device, dtype=torch.float32)
        self.negative_label = torch.as_tensor(np.array([-1]), device=self.device, dtype=torch.float32)

    def _run(self, image, input_point, input_label):
        embeddings = self._encode(image)

        coord = torch.cat((input_point, self.zero_point), dim=0)[None, :, :]
        label = torch.cat((input_label, self.negative_label), dim=0)[None, :]

        coord = self.transform.apply_coords_torch(coord, image.shape[:2])

        scores, masks = self._decode(embeddings, coord, label)

        orig_im_size = [image.shape[0], image.shape[1]]

        upscaled_masks = self._mask_postprocessing(masks, orig_im_size)

        upscaled_masks = upscaled_masks > self.sam.mask_threshold

        return upscaled_masks, scores, masks

class SegmentAllMasksPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)

    def _process_batch(
        self,
        image,
        image_embeddings,
        points: np.ndarray,
        im_size,
        crop_box,
        orig_size,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        iou_preds, masks = self._decode(
            image_embeddings,
            in_points[:, None, :],
            in_labels[:, None]
        )

        mask_slice = slice(3, 4)
        masks = masks[:, mask_slice, :, :]
        iou_preds = iou_preds[:, mask_slice]

        orig_im_size = [image.shape[0], image.shape[1]]

        masks = self._mask_postprocessing(masks, orig_im_size)

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.sam.mask_threshold, self.stability_score_offset
        )

        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = uncrop_masks(data["masks"] > self.sam.mask_threshold, crop_box, orig_h, orig_w)
        data["boxes"] = batched_mask_to_box(data["masks"])

        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box,
        crop_layer_idx: int,
        orig_size,
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        image_embeddings = self._encode(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(image, image_embeddings, points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)

        data.to_numpy()
        return data

    def _run(self, image):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        # Encode masks
        data["segmentations"] = [mask for mask in data["masks"]]

        # Write mask records
        curr_anns = []
        for idx in range(len(data["segmentations"])):
            ann = {
                "segmentation": data["segmentations"][idx],
                "bbox": box_xyxy_to_xywh(data["boxes"][idx]).tolist(),
                "predicted_iou": data["iou_preds"][idx].item(),
                "point_coords": [data["points"][idx].tolist()],
                "stability_score": data["stability_score"][idx].item(),
            }
            curr_anns.append(ann)

        return curr_anns

def make_pipeline(args):
    if args.mode == "point":
        return SegmentPointPipeline(args)
    elif args.mode == "all-masks":
        return SegmentAllMasksPipeline(args)
    else:
        raise RuntimeError("Incorrect mode argument {}".format(args.mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options for MJ models export, optimization and compilation")

    # Execution options
    parser.add_argument('--input-image', help="Path to input image")
    parser.add_argument('--output-image', default='output.png', help="Path to output image")
    parser.add_argument('--onnx-dir', default='onnx', help="Path to ONNX dir")
    parser.add_argument('--engine-dir', default='engine', help="Path to TRT engine dir")
    parser.add_argument('--torch', action='store_true', help="Run PyTorch inference")
    parser.add_argument('--visualize', action='store_true', help="Visualize output")
    parser.add_argument('--mode', choices=["point", "all-masks"], default='point', help="Execution mode")
    parser.add_argument('--benchmark', action='store_true', help="Benchmark pipeline")
    parser.add_argument('--mask-slice-idx', choices=[0, 1, 2, 3], default=3, type=int, help="Mask slice index")
    parser.add_argument('--point-coord', dest="coord", type=coords, nargs=1, default=[(300,300)], help="Coordinate of a query point")
    parser.add_argument('--use-cuda-graph', action='store_true', help="Enable cuda graph")

    # ONNX export options
    parser.add_argument('--model-type', choices=["vit_b", "vit_l", "vit_h"], default='vit_h', help="Model configurations")
    parser.add_argument('--checkpoint', default='checkpoint.pth', help="Path to model checkpoint")
    parser.add_argument('--opset', type=int, default=17, help="ONNX export opset")
    parser.add_argument('--force-onnx-export', action='store_true', help="Force ONNX export of Encoder and Decoder")
    parser.add_argument('--not-export-params', action='store_true', help="Export ONNX without params. Use only to visualize ONNX")
    parser.add_argument('--force-onnx-optimize', action='store_true', help="Force ONNX optimizations for Encoder and Decoder models")
    parser.add_argument('--onnx-minimal-optimization', action='store_true', help="Restrict ONNX optimization to const folding and shape inference.")

    # TensorRT build options
    parser.add_argument('--force-engine-build', action='store_true', help="Force rebuilding the TensorRT engine")
    parser.add_argument('--timing-cache', default=None, type=str, help="Path to the precached timing measurements to accelerate build.")
    parser.add_argument('--memory-pool-gbs', default=32, type=int, help="Size of memory pool in gigabytes")
    
    # TensorRT inference options
    parser.add_argument('--num-warmup-runs', type=int, default=5, help="Number of warmup runs before benchmarking performance")
    parser.add_argument('--runs', type=int, default=5, help="Number of runs for benchmarking performance")
    parser.add_argument('--nvtx-profile', action='store_true', help="Enable NVTX markers for performance profiling")

    parser.add_argument('--seed', type=int, default=1997, help="Seed for random data generation")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose output")
    
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    pipeline = make_pipeline(args)

    if args.use_cuda_graph:
        # Run once to make a graph
        pipeline.run(image, args)

    output = pipeline.run(image, args)
    if args.visualize:
        if args.mode == "point":
            # Example input point
            upsampled_masks, scores, masks = output
            visualize_point_and_mask(args.output_image, original_image, upsampled_masks, 
                                     np.array([[*args.coord[0]]]), np.array([1]), args.mask_slice_idx)
        elif args.mode == "all-masks":
            masks = output
            visualize_multiple_masks(args.output_image, original_image, masks)
        else:
            raise RuntimeError("Incorrect mode argument {}".format(args.mode))

    if args.benchmark:
        benchmark("SAM", pipeline.run, args, (image, args))



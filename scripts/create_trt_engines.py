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

import numpy as np
import torch
from torch.nn import functional as F
import argparse

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


from segment_anything.utils.onnx import SamOnnxModel

from segment_anything.utils.amg import build_all_layer_point_grids

from segment_anything import sam_model_registry, SamPredictor

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
            os.makedirs(os.path.dirname(self.raw_onnx_path), exist_ok=True) 
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
        os.makedirs(os.path.dirname(self.engine_path), exist_ok=True) 

        
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
        # Todo: Once dynamic_axes are supported in Holoscan
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


def make_pipeline(args):
    if args.mode == "point":
        return SegmentPointPipeline(args)
    else:
        raise RuntimeError("Incorrect mode argument {}".format(args.mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options for MJ models export, optimization and compilation")

    # Execution options
    parser.add_argument('--onnx-dir', default='onnx', help="Path to ONNX dir")
    parser.add_argument('--engine-dir', default='engine', help="Path to TRT engine dir")
    parser.add_argument('--mode', choices=["point"], default='point', help="Execution mode")
    parser.add_argument('--point-coord', dest="coord", nargs=1, default=[(300,300)], help="Coordinate of a query point")
    parser.add_argument('--use-cuda-graph', action='store_true', help="Enable cuda graph")
    parser.add_argument('--seed', type=int, default=1997, help="Seed for random data generation")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose output")

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
    

    args = parser.parse_args()

    pipeline = make_pipeline(args)




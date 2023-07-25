#!/usr/bin/env python
import json
import argparse
import pprint as pp
import pickle as pkl
import math
from schedule.cost_graph import Operator,Network

# Data types
FLOAT32 = 0
FLOAT16 = 1
INT8 = 2
INT16 = 3

def make_tflite_json_tensor(t):
    return Tensor(t["name"], t["shape"])

class Tensor(object):
    def __init__(self, id: int, shape: list) -> None:
        # id or name
        self.id = id
        self.shape = shape
        ## set all to int8 for now
        self.dtype = INT8

    def extend_tensor_dims(self, n):
        """
        extend [2,2] to n rank, insert 1 from st:
        [1, 1, 2, 2]
        """

        if len(self.shape) == n:
            return self.shape

        num = n - len(self.shape)
        ret = [1 for i in range(n)]
        for i in range(num, n):
            ret[i] = self.shape[i - num]

        return ret

    def __str__(self) -> str:
        return f"<{self.id}: {self.shape},{self.dtype}>"

    def __format__(self, __format_spec: str) -> str:
        return self.__str__()


class Operator(object):
    def __init__(self):
        self.op_type = None
        self.input_tensors = None
        self.output_tensors = None
        self.params = None
        self.exec_time = None

    def info(self):
        print(">>>>>>>>>>>>>")
        print("  OpType: {}".format(self.op_type))
        print("  Inputs: {}".format(self.input_tensors))
        print("  Outputs: {}".format(self.output_tensors))
        print("  Params: {}".format(self.params))
        pass

    def __str__(self) -> str:
        def print_list(x):
            ret = ""
            for i in x:
                ret += str(i)
            return ret

        return f"""
    op_type: {self.op_type}
    inputs: {print_list(self.input_tensors)}
    outputs: {print_list(self.output_tensors)}
    params:  {self.params}
        """

    def __format__(self, __format_spec: str) -> str:
        return self.__str__()

class TFOperator(Operator):
    TF_TO_BST = {
        "Conv2D": "Conv",
        "FullyConnected": "Conv",
        "Pool2D": "MaxPool",
        "Concatenation": "Concat",
        "Reshape": "Reshape",
        "Softmax": "LogSoftmax",
    }

    def __init__(self, op_node, inputs, outputs):
        super().__init__()
        self.from_tflite_json(op_node, inputs, outputs)
        self.id = op_node["name"]

    def from_tflite_json(self, node, inputs, outputs):
        self.node = node
        self.op_type = str(node["builtin_options_type"]).replace("Options", "")

        self.input_tensors = [make_tflite_json_tensor(t) for t in inputs]
        self.output_tensors = [make_tflite_json_tensor(t) for t in outputs]

        self.params = node["builtin_options"]

    def get_bst_layer_type(self):
        if self.op_type in TFOperator.TF_TO_BST.keys():
            return TFOperator.TF_TO_BST[self.op_type]
        else:
            return self.op_type

    def extract_conv_like_params(self):
        if self.op_type == "FullyConnected":
            w_shape = [
                self.input_tensors[1].shape[0],
                1,
                1,
                self.input_tensors[1].shape[1],
            ]
            return {
                "weight_shape": w_shape,
                "dilation": 1,
                "stride": 1,
                "kernel_h": 1,
                "kernel_w": 1,
                "pad": 0,
                "group": 1,
            }
            pass
        elif self.op_type == "Conv2D":
            assert self.params["stride_w"] == self.params["stride_h"]

            if "padding" in self.params.keys():
                assert self.params["padding"] == "VALID"

            return {
                "weight_shape": self.input_tensors[1].shape,
                "dilation": 1,
                "stride": self.params["stride_w"],
                "kernel_h": self.input_tensors[1].shape[1],
                "kernel_w": self.input_tensors[1].shape[2],
                "pad": 0,
                "group": 1,
            }

    def extract_pool_like_params(self):
        if self.op_type == "Pool2D":
            assert self.params["stride_w"] == self.params["stride_h"]
            assert self.params["filter_width"] == self.params["filter_height"]
            if "padding" in self.params.keys():
                assert self.params["padding"] == "VALID"

            return {
                "stride": self.params["stride_w"],
                "kernel_h": self.params["filter_height"],
                "kernel_w": self.params["filter_width"],
                "pad": 0,
                "ceil_mode": 0,
            }

    def extract_axis_param(self):
        return {"axis": self.params["axis"]}


def make_tfilte_json_network(json_model):
    def parse_ops(net_json, net: Network):
        assert len(net_json["subgraphs"]) == 1, "Only support one subgraph now!"

        ops_json = net_json["subgraphs"][0]["operators"]

        for op in ops_json:
            net.ops.append(
                TFOperator(
                    op,
                    [net.tensors[ti] for ti in op["inputs"]],
                    [net.tensors[ti] for ti in op["outputs"]],
                )
            )

    with open(json_model, "r") as fp:
        net = Network()
        net_json = json.load(fp)

        # build tensor index map for later accessing
        net.tensors = list(net_json["subgraphs"][0]["tensors"])
        net.ops = []

        parse_ops(net_json, net)

class ONNXOperator(Operator):
    def __init__(self, op_node, inputs, outputs):
        super().__init__()
        self.from_onnx_proto(op_node, inputs, outputs)

    def from_onnx_proto(self, node: onnx.NodeProto, inputs, outputs):
        self.op_type = node.op_type
        self.id = node.name

        def __get_attr_param(n):
            """Return the attribute with name as index."""
            ret = {}
            for i in n.attribute:
                ret[i.name] = i

            return ret

        self.input_tensors = inputs
        self.output_tensors = outputs

        self.params = __get_attr_param(node)
        pass

    def get_op_type(self):
        return self.op_type

    def extract_conv_like_params(self):
        if self.op_type == "FC":
            w_shape = [
                self.input_tensors[1].shape[0],
                1,
                1,
                self.input_tensors[1].shape[1],
            ]
            return {
                "weight_shape": w_shape,
                "dilation": 1,
                "stride": 1,
                "kernel_h": 1,
                "kernel_w": 1,
                "pad": 0,
                "group": 1,
            }

        elif self.op_type == "Conv":
            # assert self.input_shapes[1][1] == self.params["kernel_shape"][0]
            # assert self.input_shapes[1][2] == self.params["kernel_shape"][1]

            dilation = 1
            if "dilations" in self.params.keys():
                dilation = self.params["dilations"].ints[0]

            return {
                "weight_shape": self.input_tensors[1].shape,
                "dilation": dilation,
                "stride": self.params["strides"].ints[0],
                "kernel_h": self.params["kernel_shape"].ints[0],
                "kernel_w": self.params["kernel_shape"].ints[1],
                "pad": 0,
                "group": 1,
            }
        else:
            return None

    def extract_pool_like_params(self):
        if self.op_type == "MaxPool" or self.op_type == "AveragePool":
            assert (
                self.params["kernel_shape"].ints[0]
                == self.params["kernel_shape"].ints[0]
            )
            assert self.params["strides"].ints[0] == self.params["strides"].ints[0]

            return {
                "stride": self.params["strides"].ints[0],
                "kernel_h": self.params["kernel_shape"].ints[0],
                "kernel_w": self.params["kernel_shape"].ints[1],
                "pad": 0,
                "ceil_mode": 0,
            }
        else:
            return None

    def extract_axis_param(self):
        return {"axis": int(self.params["axis"].i)}


def make_onnx_tensor(value_info):
    return Tensor(
        value_info.name, [s.dim_value for s in value_info.type.tensor_type.shape.dim]
    )


def make_onnx_init_tensor(value_info):
    if isinstance(value_info.dims, int):
        return Tensor(value_info.name, [value_info.dims])
    else:
        return Tensor(value_info.name, [s for s in value_info.dims])


def load_onnx_model(model_path):
    p = pathlib.Path(model_path)
    if not p.exists():
        print(f"Error: model {model_path} is not found")
        return None
    if "onnx" in p.suffix:
        return onnx.load(model_path)

    elif "json" in p.suffix:
        with open(model_path, "r") as fp:
            js = json.loads(fp.read())
            js_str = json.dumps(js)
            return Parse(js_str, onnx.ModelProto())
    else:
        return None


def make_onnx_network(onnx_model: str) -> Network:
    onnx_model = load_onnx_model(onnx_model)
    logging.info("finished loading onnx model")

    # collect tensors
    net = Network()

    # build dag
    net.nx_graph = onnx_to_dag(onnx_model)

    graph = onnx_model.graph

    net.tensors = {}
    net.ops = {}

    # add initializer as tensors
    for t in graph.initializer:
        net.tensors[t.name] = make_onnx_init_tensor(t)

    data = []
    data.extend(list(graph.input))
    data.extend(list(graph.output))
    data.extend(list(graph.value_info))
    # add value_info as tensors
    for t in data:
        if len(t.name) == 0:
            logging.error("Error: found tensor without name")
            exit(-1)

        net.tensors[t.name] = make_onnx_tensor(t)

    for i, node in enumerate(onnx_model.graph.node):
        # check if all inputs and outputs are included
        for x in list(node.input) + list(node.output):
            if x not in net.tensors.keys():
                logging.error(f"Error: Not found tensor {x}, len {len(x)}")
                exit(-1)

        logging.info(
            f"Building ONNXOperator {node.name} with inputs: {node.input} and outputs: {node.output}"
        )
        net.ops[node.name] = ONNXOperator(
            node,
            [net.tensors[x] for x in node.input],
            [net.tensors[x] for x in node.output],
        )

    return net
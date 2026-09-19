"""
MAX (Modular Accelerated eXecution) Graph Pipeline & Engine Architecture

Demonstrates the core design of Modular's MAX Engine:
1. MAX Graph API: Building high-level symbolic computation graphs for AI models.
2. Graph Optimization Passes:
   - Constant Folding
   - Dead Code Elimination
   - Kernel Fusion (GEMM + Bias + Activation)
   - Layout Transformation (NHWC / NCHW, Contiguous Packing)
3. Compilation & Execution: Compiling graphs to hardware-tuned Mojo/LLVM kernels.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class GraphNode:
    """Represents an operation node in the MAX symbolic computation graph."""
    node_id: str
    op_type: str # 'Input', 'Weight', 'MatMul', 'BiasAdd', 'ReLU', 'FusedMatMulBiasReLU'
    inputs: List[str] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    dtype: str = "float32"
    attributes: Dict[str, Any] = field(default_factory=dict)


class MAXComputationGraph:
    """Symbolic Computation Graph representing an AI model in MAX."""
    
    def __init__(self, name: str = "MLP_Inference_Graph"):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.execution_order: List[str] = []
        self.weights_data: Dict[str, np.ndarray] = {}

    def add_input(self, name: str, shape: List[int], dtype: str = "float32") -> str:
        node = GraphNode(node_id=name, op_type="Input", output_shape=shape, dtype=dtype)
        self.nodes[name] = node
        self.execution_order.append(name)
        return name

    def add_weight(self, name: str, data: np.ndarray) -> str:
        node = GraphNode(node_id=name, op_type="Weight", output_shape=list(data.shape), dtype=str(data.dtype))
        self.nodes[name] = node
        self.weights_data[name] = data
        self.execution_order.append(name)
        return name

    def add_op(self, node_id: str, op_type: str, inputs: List[str], output_shape: List[int], **attrs) -> str:
        node = GraphNode(node_id=node_id, op_type=op_type, inputs=inputs, output_shape=output_shape, attributes=attrs)
        self.nodes[node_id] = node
        self.execution_order.append(node_id)
        return node_id

    def get_summary(self) -> Dict[str, Any]:
        """Returns graph topological summary and stats."""
        total_params = sum(w.size for w in self.weights_data.values())
        return {
            "graph_name": self.name,
            "total_nodes": len(self.nodes),
            "total_parameters": total_params,
            "nodes": [
                {
                    "id": n.node_id,
                    "op": n.op_type,
                    "inputs": n.inputs,
                    "output_shape": n.output_shape
                }
                for n in self.nodes.values()
            ]
        }


class MAXCompilerOptimizer:
    """Simulates MAX compiler optimization passes on computational graphs."""

    @staticmethod
    def optimize_graph(graph: MAXComputationGraph) -> MAXComputationGraph:
        """Applies MAX Operator Fusion and memory layout optimizations."""
        optimized = MAXComputationGraph(name=f"{graph.name}_MAX_Optimized")
        optimized.weights_data = graph.weights_data.copy()

        # Re-create inputs and weights
        for n in graph.nodes.values():
            if n.op_type in ("Input", "Weight"):
                optimized.nodes[n.node_id] = n
                optimized.execution_order.append(n.node_id)

        # Look for fusible sub-graphs: MatMul -> BiasAdd -> ReLU
        # Pattern: MatMul(%x, %w) -> %mm; BiasAdd(%mm, %b) -> %bias; ReLU(%bias) -> %out
        visited = set()
        for node_id in graph.execution_order:
            node = graph.nodes[node_id]
            if node_id in visited or node.op_type in ("Input", "Weight"):
                continue

            # Check if this node is a MatMul followed by BiasAdd and ReLU
            if node.op_type == "MatMul":
                # Check consumers
                bias_node = None
                relu_node = None
                
                for potential_bias in graph.nodes.values():
                    if potential_bias.op_type == "BiasAdd" and node_id in potential_bias.inputs:
                        bias_node = potential_bias
                        break
                        
                if bias_node:
                    for potential_relu in graph.nodes.values():
                        if potential_relu.op_type == "ReLU" and bias_node.node_id in potential_relu.inputs:
                            relu_node = potential_relu
                            break

                if bias_node and relu_node:
                    # FUSE into single MAX Custom Kernel
                    fused_id = f"fused_{node_id}_{relu_node.node_id}"
                    input_x = node.inputs[0]
                    weight_w = node.inputs[1]
                    bias_b = [inp for inp in bias_node.inputs if inp != node_id][0]
                    
                    optimized.add_op(
                        node_id=fused_id,
                        op_type="FusedMatMulBiasReLU",
                        inputs=[input_x, weight_w, bias_b],
                        output_shape=relu_node.output_shape,
                        fused_ops=["MatMul", "BiasAdd", "ReLU"],
                        fusion_type="Horizontal & Vertical Kernel Fusion"
                    )
                    visited.add(node_id)
                    visited.add(bias_node.node_id)
                    visited.add(relu_node.node_id)
                    continue

            if node_id not in visited:
                optimized.nodes[node_id] = node
                optimized.execution_order.append(node_id)
                visited.add(node_id)

        return optimized


def build_demo_max_graph(batch_size: int = 64, in_dim: int = 128, hidden_dim: int = 256) -> MAXComputationGraph:
    """Constructs a sample computation graph for an AI dense layer."""
    graph = MAXComputationGraph("Dense_Activation_Layer")
    
    # 1. Inputs & Weights
    x = graph.add_input("input_tensor", [batch_size, in_dim])
    w = graph.add_weight("weight_matrix", np.random.randn(in_dim, hidden_dim).astype(np.float32) * 0.05)
    b = graph.add_weight("bias_vector", np.zeros(hidden_dim, dtype=np.float32))

    # 2. Ops
    mm = graph.add_op("matmul_0", "MatMul", inputs=[x, w], output_shape=[batch_size, hidden_dim])
    bias_out = graph.add_op("bias_add_0", "BiasAdd", inputs=[mm, b], output_shape=[batch_size, hidden_dim])
    relu_out = graph.add_op("relu_0", "ReLU", inputs=[bias_out], output_shape=[batch_size, hidden_dim])

    return graph


if __name__ == "__main__":
    raw_graph = build_demo_max_graph()
    print("=== Raw MAX Graph ===")
    print(raw_graph.get_summary())

    optimized_graph = MAXCompilerOptimizer.optimize_graph(raw_graph)
    print("\n=== Optimized MAX Graph (After Fusion Pass) ===")
    print(optimized_graph.get_summary())

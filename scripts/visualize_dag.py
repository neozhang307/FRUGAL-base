#!/usr/bin/env python3
"""
Visualize the DAG (Directed Acyclic Graph) of task dependencies and data flow.
Creates graphviz-style visualizations showing:
1. Task dependency graph with execution order overlay
2. Data flow showing which arrays connect which tasks
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import networkx as nx

def load_profile(profile_file):
    """Load profiling data JSON."""
    with open(profile_file, 'r') as f:
        return json.load(f)

def load_plan(plan_file):
    """Load optimized plan JSON (optional)."""
    if plan_file and Path(plan_file).exists():
        with open(plan_file, 'r') as f:
            return json.load(f)
    return None

def build_taskid_to_taskgroup_map(profile):
    """
    Build a mapping from internal taskId to taskGroup.id.

    The plan file uses taskId (internal kernel IDs, e.g., 1-20),
    while the profile uses taskGroup.id (0-19). These are different!

    Example from profile:
      TaskGroup 11 has taskIds: [14]  -> taskId 14 maps to taskGroupId 11
      TaskGroup 13 has taskIds: [12]  -> taskId 12 maps to taskGroupId 13
    """
    taskid_to_groupid = {}
    for tg in profile.get('taskGroups', []):
        group_id = tg['id']
        for task_id in tg.get('taskIds', []):
            taskid_to_groupid[task_id] = group_id
    return taskid_to_groupid


def extract_task_order(plan, profile):
    """
    Extract task execution order from optimized plan.

    Returns a dict mapping taskGroup.id -> execution order (0-based).

    NOTE: The plan is a DAG - the actual execution order is determined by
    topological sort of the graph, NOT by the order nodes appear in JSON.

    NOTE: The plan file stores taskId (internal kernel IDs), but we need
    to convert to taskGroup.id which is what the profile uses.
    """
    if not plan:
        return {}

    # Build the DAG from plan nodes
    G = nx.DiGraph()
    node_info = {}

    for node in plan.get('nodes', []):
        node_id = node['nodeId']
        G.add_node(node_id)
        node_info[node_id] = {
            'nodeType': node.get('nodeType', -1),
            'taskId': node.get('taskId', -1)
        }
        for edge_target in node.get('edges', []):
            G.add_edge(node_id, edge_target)

    # Topological sort gives actual execution order
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Warning: Plan graph has cycles, cannot determine execution order")
        return {}

    # Build taskId -> taskGroupId mapping
    taskid_to_groupid = build_taskid_to_taskgroup_map(profile)

    # Extract task execution order from topological sort
    task_order = {}  # taskGroupId -> execution order
    order = 0
    for node_id in topo_order:
        if node_info[node_id]['nodeType'] == 1:  # Task node
            task_id = node_info[node_id]['taskId']
            if task_id >= 0:
                # Convert taskId to taskGroupId
                group_id = taskid_to_groupid.get(task_id)
                if group_id is not None:
                    task_order[group_id] = order
                    order += 1
                else:
                    print(f"Warning: taskId {task_id} not found in profile taskGroups")

    return task_order

def create_task_dependency_graph(profile, task_order=None, output_dir=None):
    """
    Create task dependency DAG using networkx and matplotlib.
    Shows task dependencies with execution order overlay.
    """

    # Build graph
    G = nx.DiGraph()

    task_groups = profile.get('taskGroups', [])
    edges = profile.get('taskGroupEdges', {})

    # Add nodes
    for tg in task_groups:
        task_id = tg['id']
        G.add_node(task_id,
                   runtime=tg.get('runningTime', 0),
                   inputs=tg.get('inputArrays', []),
                   outputs=tg.get('outputArrays', []))

    # Add edges
    for src, dests in edges.items():
        src_id = int(src)
        for dest_id in dests:
            G.add_edge(src_id, dest_id)

    # Create figure - narrow format for column layout
    fig, ax = plt.subplots(figsize=(8, 14))

    # Use hierarchical layered layout - top to bottom
    try:
        # Get topological generations for better layout
        layers = list(nx.topological_generations(G))
        layer_map = {}
        for layer_idx, layer_nodes in enumerate(layers):
            for node in layer_nodes:
                layer_map[node] = layer_idx

        # Arrange nodes by layer with better spacing
        max_layer = max(layer_map.values())
        layer_counts = defaultdict(int)
        layer_positions = defaultdict(int)

        # Count nodes per layer
        for node in G.nodes():
            layer = layer_map.get(node, 0)
            layer_counts[layer] += 1

        # Position nodes - TOP TO BOTTOM layout
        pos = {}
        for node in sorted(G.nodes()):
            layer = layer_map.get(node, 0)
            num_in_layer = layer_counts[layer]
            position_in_layer = layer_positions[layer]
            layer_positions[layer] += 1

            # X: spread nodes within layer evenly (horizontal spread)
            if num_in_layer > 1:
                x = position_in_layer / (num_in_layer - 1)
            else:
                x = 0.5

            # Y: top to bottom based on layer (1.0 at top, 0.0 at bottom)
            y = 1.0 - (layer / max(max_layer, 1))

            pos[node] = (x, y)
    except:
        # Fall back to spring layout with more iterations
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    # Color nodes by execution order if available
    if task_order:
        max_order = max(task_order.values()) if task_order else 1
        node_colors = []
        for node in G.nodes():
            if node in task_order:
                # Color gradient from blue (early) to red (late)
                order_ratio = task_order[node] / max_order
                node_colors.append((1-order_ratio, 0.2, order_ratio))
            else:
                node_colors.append((0.7, 0.7, 0.7))  # Gray for unscheduled
    else:
        node_colors = [(0.5, 0.7, 0.9)] * len(G.nodes())

    # Node sizes proportional to runtime (smaller for narrow layout)
    runtimes = [G.nodes[n]['runtime'] for n in G.nodes()]
    max_runtime = max(runtimes) if runtimes else 1
    node_sizes = [1200 * (rt / max_runtime + 0.3) for rt in runtimes]

    # Draw edges first with prominent arrows
    nx.draw_networkx_edges(G, pos, ax=ax,
                          edge_color='#2c3e50',
                          arrows=True,
                          arrowsize=25,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.1',
                          width=2.5,
                          alpha=0.7,
                          min_source_margin=20,
                          min_target_margin=20)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax,
                          node_color=node_colors,
                          node_size=node_sizes,
                          edgecolors='black',
                          linewidths=2)

    # Draw labels
    labels = {}
    for node in G.nodes():
        runtime = G.nodes[node]['runtime']
        if task_order and node in task_order:
            labels[node] = f"T{node}\n[{task_order[node]}]\n{runtime:.2f}s"
        else:
            labels[node] = f"T{node}\n{runtime:.2f}s"

    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                           font_size=7,
                           font_weight='bold',
                           font_color='white')

    title = 'Task Dependency DAG\n(Top → Bottom: Dependency Flow)'
    if task_order:
        title = 'Task Dependency DAG with Execution Order\n(Top → Bottom: Dependency Flow)'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    # Legend
    legend_elements = []
    if task_order:
        legend_elements.extend([
            mpatches.Patch(facecolor=(1, 0.2, 0), label='Early in schedule', edgecolor='black'),
            mpatches.Patch(facecolor=(0, 0.2, 1), label='Late in schedule', edgecolor='black'),
        ])
    legend_elements.append(
        mpatches.Patch(facecolor='none', edgecolor='none',
                      label=f'Size ∝ runtime\n[#] = exec order\n↓ = dependency')
    )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / 'task_dependency_graph.pdf'
        png_path = output_dir / 'task_dependency_graph.png'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {pdf_path}")
        print(f"✅ Saved: {png_path}")

    return fig

def create_data_flow_graph(profile, task_order=None, output_dir=None):
    """
    Create data flow graph showing how arrays flow through tasks.
    Bipartite graph: tasks and arrays.
    """

    task_groups = profile.get('taskGroups', [])
    arrays = profile.get('arrays', [])

    # Build bipartite graph
    G = nx.DiGraph()

    # Add task nodes
    for tg in task_groups:
        task_id = tg['id']
        G.add_node(f"T{task_id}", node_type='task',
                   runtime=tg.get('runningTime', 0))

    # Add array nodes
    for arr in arrays:
        arr_id = arr['id']
        size_mb = arr.get('size', 0) / (1024**2)
        G.add_node(f"A{arr_id}", node_type='array', size_mb=size_mb)

    # Add edges: array -> task (input), task -> array (output)
    for tg in task_groups:
        task_id = tg['id']
        task_node = f"T{task_id}"

        # Input arrays
        for arr_id in tg.get('inputArrays', []):
            G.add_edge(f"A{arr_id}", task_node, edge_type='read')

        # Output arrays
        for arr_id in tg.get('outputArrays', []):
            G.add_edge(task_node, f"A{arr_id}", edge_type='write')

    # Create figure - narrow format for column layout
    fig, ax = plt.subplots(figsize=(10, 14))

    # Separate nodes by type for bipartite layout
    task_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'task']
    array_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'array']

    # Position nodes in two columns
    pos = {}

    # Tasks on the right - order by execution schedule if available
    num_tasks = len(task_nodes)
    if task_order:
        # Sort tasks by execution order
        task_nodes_sorted = sorted(task_nodes, key=lambda x: task_order.get(int(x[1:]), 999))
    else:
        # Sort by task ID
        task_nodes_sorted = sorted(task_nodes, key=lambda x: int(x[1:]))

    for i, node in enumerate(task_nodes_sorted):
        pos[node] = (1.5, 1 - (i / max(num_tasks-1, 1)))  # Invert Y so first task is at top

    # Arrays on the left
    num_arrays = len(array_nodes)
    for i, node in enumerate(sorted(array_nodes, key=lambda x: int(x[1:]))):
        pos[node] = (0, 1 - (i / max(num_arrays-1, 1)))  # Invert Y for consistency

    # Draw edges
    read_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'read']
    write_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'write']

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=read_edges,
                          edge_color='#3498db',  # Blue for reads
                          arrows=True, arrowsize=15,
                          width=1.5, alpha=0.6,
                          connectionstyle='arc3,rad=0.05')

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=write_edges,
                          edge_color='#e74c3c',  # Red for writes
                          arrows=True, arrowsize=15,
                          width=2, alpha=0.7,
                          connectionstyle='arc3,rad=0.05')

    # Draw task nodes (squares) - smaller for narrow layout
    task_pos = {n: pos[n] for n in task_nodes}
    runtimes = [G.nodes[n]['runtime'] for n in task_nodes]
    max_runtime = max(runtimes) if runtimes else 1
    task_sizes = [800 * (G.nodes[n]['runtime'] / max_runtime + 0.3) for n in task_nodes]

    nx.draw_networkx_nodes(G, task_pos, nodelist=task_nodes, ax=ax,
                          node_color='#2ecc71',  # Green
                          node_shape='s',  # Square
                          node_size=task_sizes,
                          edgecolors='black',
                          linewidths=2)

    # Draw array nodes (circles) - smaller for narrow layout
    array_pos = {n: pos[n] for n in array_nodes}
    array_sizes = [600 for _ in array_nodes]  # Same size

    nx.draw_networkx_nodes(G, array_pos, nodelist=array_nodes, ax=ax,
                          node_color='#9b59b6',  # Purple
                          node_shape='o',  # Circle
                          node_size=array_sizes,
                          edgecolors='black',
                          linewidths=2)

    # Labels with execution order
    task_labels = {}
    for n in task_nodes:
        task_id = int(n[1:])
        runtime = G.nodes[n]['runtime']
        if task_order and task_id in task_order:
            order = task_order[task_id]
            task_labels[n] = f"{n}\n[{order}]\n{runtime:.2f}s"
        else:
            task_labels[n] = f"{n}\n{runtime:.2f}s"

    array_labels = {n: n + f"\n{G.nodes[n]['size_mb']:.0f}MB" for n in array_nodes}

    nx.draw_networkx_labels(G, task_pos, task_labels, ax=ax,
                           font_size=7, font_weight='bold', font_color='white')
    nx.draw_networkx_labels(G, array_pos, array_labels, ax=ax,
                           font_size=7, font_weight='bold', font_color='white')

    title = 'Data Flow Graph: Arrays ↔ Tasks'
    if task_order:
        title += '\n(Ordered by Schedule)'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.text(0, -0.05, 'Data Arrays', ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(1.5, -0.05, 'Tasks\n(Top→Bottom:\nExec Order)' if task_order else 'Task Groups',
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#9b59b6', label='Arrays', edgecolor='black'),
        mpatches.Patch(facecolor='#2ecc71', label='Tasks', edgecolor='black'),
        mpatches.Patch(facecolor='none', edgecolor='#3498db', label='Read (input)'),
        mpatches.Patch(facecolor='none', edgecolor='#e74c3c', label='Write (output)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
             bbox_to_anchor=(0.75, 1.0), fontsize=8, ncol=2)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / 'data_flow_graph.pdf'
        png_path = output_dir / 'data_flow_graph.png'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {pdf_path}")
        print(f"✅ Saved: {png_path}")

    return fig

def print_graph_summary(profile):
    """Print summary statistics about the DAG."""
    task_groups = profile.get('taskGroups', [])
    edges = profile.get('taskGroupEdges', {})
    arrays = profile.get('arrays', [])

    num_tasks = len(task_groups)
    num_edges = sum(len(dests) for dests in edges.values())
    num_arrays = len(arrays)

    total_runtime = sum(tg.get('runningTime', 0) for tg in task_groups)
    total_data_size = sum(arr.get('size', 0) for arr in arrays) / (1024**3)  # GB

    print("\n" + "="*60)
    print("DAG Summary")
    print("="*60)
    print(f"Tasks: {num_tasks}")
    print(f"Dependencies (edges): {num_edges}")
    print(f"Arrays: {num_arrays}")
    print(f"Total sequential runtime: {total_runtime:.2f}s")
    print(f"Total data size: {total_data_size:.2f} GB")
    print(f"Avg runtime per task: {total_runtime/num_tasks:.3f}s")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Visualize DAG structure')
    parser.add_argument('--profile', required=True, help='Profile JSON file')
    parser.add_argument('--plan', help='Optimized plan JSON file (optional)')
    parser.add_argument('--output', default='results/ablation/visualization',
                       help='Output directory for plots')
    parser.add_argument('--graphs', nargs='+',
                       choices=['task', 'data', 'all'],
                       default=['all'],
                       help='Which graphs to generate')

    args = parser.parse_args()

    profile_path = Path(args.profile)
    plan_path = Path(args.plan) if args.plan else None
    output_dir = Path(args.output)

    print(f"Loading profile: {profile_path}")
    profile = load_profile(profile_path)

    task_order = None
    if plan_path and plan_path.exists():
        print(f"Loading plan: {plan_path}")
        plan = load_plan(plan_path)
        task_order = extract_task_order(plan, profile)
        print(f"Extracted execution order for {len(task_order)} taskGroups")

    print_graph_summary(profile)

    graphs_to_generate = args.graphs
    if 'all' in graphs_to_generate:
        graphs_to_generate = ['task', 'data']

    print("\nGenerating visualizations...")

    if 'task' in graphs_to_generate:
        print("\n📊 Creating task dependency graph...")
        create_task_dependency_graph(profile, task_order, output_dir)

    if 'data' in graphs_to_generate:
        print("\n📊 Creating data flow graph...")
        create_data_flow_graph(profile, task_order, output_dir)

    print("\n✅ Visualization complete!")

if __name__ == '__main__':
    main()

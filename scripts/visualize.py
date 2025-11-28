#!/usr/bin/env python3
"""
Unified visualization tool for FRUGAL execution plans.

Generates two visualizations:
1. Task Dependency DAG - Shows logical dependencies between tasks
2. Execution Plan DAG - Shows full execution plan with prefetch/offload nodes

Usage:
    python scripts/visualize.py --profile profile.json --plan plan.json --output output_dir/
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from pathlib import Path
import argparse
from collections import defaultdict


def load_profile(profile_file):
    """Load profiling data JSON."""
    with open(profile_file, 'r') as f:
        return json.load(f)


def load_plan(plan_file):
    """Load optimized plan JSON."""
    if plan_file and Path(plan_file).exists():
        with open(plan_file, 'r') as f:
            return json.load(f)
    return None


def build_taskid_to_taskgroup_map(profile):
    """
    Build a mapping from internal taskId to taskGroup.id.

    The plan file uses taskId (internal kernel IDs, e.g., 1-20),
    while the profile uses taskGroup.id (0-19). These are different!
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
    """
    if not plan:
        return {}

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

    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Warning: Plan graph has cycles, cannot determine execution order")
        return {}

    taskid_to_groupid = build_taskid_to_taskgroup_map(profile)

    task_order = {}
    order = 0
    for node_id in topo_order:
        if node_info[node_id]['nodeType'] == 1:  # Task node
            task_id = node_info[node_id]['taskId']
            if task_id >= 0:
                group_id = taskid_to_groupid.get(task_id)
                if group_id is not None:
                    task_order[group_id] = order
                    order += 1

    return task_order


def print_graph_summary(profile):
    """Print summary statistics about the DAG."""
    task_groups = profile.get('taskGroups', [])
    edges = profile.get('taskGroupEdges', {})
    arrays = profile.get('arrays', [])

    num_tasks = len(task_groups)
    num_edges = sum(len(dests) for dests in edges.values())
    num_arrays = len(arrays)

    total_runtime = sum(tg.get('runningTime', 0) for tg in task_groups)
    total_data_size = sum(arr.get('size', 0) for arr in arrays) / (1024**3)

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


# =============================================================================
# Task Dependency DAG Visualization
# =============================================================================

def create_task_dependency_graph(profile, task_order=None, output_dir=None):
    """
    Create task dependency DAG using networkx and matplotlib.
    Shows task dependencies with execution order overlay.
    """
    G = nx.DiGraph()

    task_groups = profile.get('taskGroups', [])
    edges = profile.get('taskGroupEdges', {})

    for tg in task_groups:
        task_id = tg['id']
        G.add_node(task_id,
                   runtime=tg.get('runningTime', 0),
                   inputs=tg.get('inputArrays', []),
                   outputs=tg.get('outputArrays', []))

    for src, dests in edges.items():
        src_id = int(src)
        for dest_id in dests:
            G.add_edge(src_id, dest_id)

    fig, ax = plt.subplots(figsize=(8, 14))

    try:
        layers = list(nx.topological_generations(G))
        layer_map = {}
        for layer_idx, layer_nodes in enumerate(layers):
            for node in layer_nodes:
                layer_map[node] = layer_idx

        max_layer = max(layer_map.values())
        layer_counts = defaultdict(int)
        layer_positions = defaultdict(int)

        for node in G.nodes():
            layer = layer_map.get(node, 0)
            layer_counts[layer] += 1

        pos = {}
        for node in sorted(G.nodes()):
            layer = layer_map.get(node, 0)
            num_in_layer = layer_counts[layer]
            position_in_layer = layer_positions[layer]
            layer_positions[layer] += 1

            if num_in_layer > 1:
                x = position_in_layer / (num_in_layer - 1)
            else:
                x = 0.5

            y = 1.0 - (layer / max(max_layer, 1))
            pos[node] = (x, y)
    except:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    if task_order:
        max_order = max(task_order.values()) if task_order else 1
        node_colors = []
        for node in G.nodes():
            if node in task_order:
                order_ratio = task_order[node] / max_order
                node_colors.append((1-order_ratio, 0.2, order_ratio))
            else:
                node_colors.append((0.7, 0.7, 0.7))
    else:
        node_colors = [(0.5, 0.7, 0.9)] * len(G.nodes())

    runtimes = [G.nodes[n]['runtime'] for n in G.nodes()]
    max_runtime = max(runtimes) if runtimes else 1
    node_sizes = [1200 * (rt / max_runtime + 0.3) for rt in runtimes]

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

    nx.draw_networkx_nodes(G, pos, ax=ax,
                          node_color=node_colors,
                          node_size=node_sizes,
                          edgecolors='black',
                          linewidths=2)

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

    title = 'Task Dependency DAG\n(Top -> Bottom: Dependency Flow)'
    if task_order:
        title = 'Task Dependency DAG with Execution Order\n(Top -> Bottom: Dependency Flow)'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    legend_elements = []
    if task_order:
        legend_elements.extend([
            mpatches.Patch(facecolor=(1, 0.2, 0), label='Early in schedule', edgecolor='black'),
            mpatches.Patch(facecolor=(0, 0.2, 1), label='Late in schedule', edgecolor='black'),
        ])
    legend_elements.append(
        mpatches.Patch(facecolor='none', edgecolor='none',
                      label=f'Size = runtime\n[#] = exec order\nArrow = dependency')
    )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / 'task_dependency_graph.pdf'
        png_path = output_dir / 'task_dependency_graph.png'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {pdf_path}")
        print(f"  Saved: {png_path}")

    plt.close()
    return fig


# =============================================================================
# Execution Plan DAG Visualization
# =============================================================================

def find_overlapping_migrations(G, node_info, profile):
    """Detect which migration nodes overlap with task execution."""
    taskid_to_groupid = {}
    for tg in profile.get('taskGroups', []):
        for task_id in tg.get('taskIds', []):
            taskid_to_groupid[task_id] = tg['id']

    topo_order = list(nx.topological_sort(G))
    task_exec_order = {}
    order = 0
    for node_id in topo_order:
        if node_info.get(node_id, {}).get('type') == 'task':
            task_exec_order[node_id] = order
            order += 1

    def find_task_predecessors(node_id, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited:
            return []
        visited.add(node_id)

        tasks = []
        for pred in G.predecessors(node_id):
            if node_info.get(pred, {}).get('type') == 'task':
                tasks.append(pred)
            else:
                tasks.extend(find_task_predecessors(pred, visited))
        return tasks

    def find_task_successors(node_id, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited:
            return []
        visited.add(node_id)

        tasks = []
        for succ in G.successors(node_id):
            if node_info.get(succ, {}).get('type') == 'task':
                tasks.append(succ)
            else:
                tasks.extend(find_task_successors(succ, visited))
        return tasks

    overlapping_nodes = set()

    for node_id, info in node_info.items():
        if info.get('type') not in ['prefetch', 'offload']:
            continue

        task_preds = find_task_predecessors(node_id)
        task_succs = find_task_successors(node_id)

        pred_orders = [task_exec_order[t] for t in task_preds if t in task_exec_order]
        succ_orders = [task_exec_order[t] for t in task_succs if t in task_exec_order]

        start_after = max(pred_orders) if pred_orders else -1
        finish_before = min(succ_orders) if succ_orders else len(task_exec_order)

        tasks_spanned = finish_before - start_after - 1

        if tasks_spanned > 0:
            overlapping_nodes.add(node_id)

    return overlapping_nodes


def visualize_plan_dag(profile, plan, output_dir, verbose=False):
    """
    Visualize the execution plan as a DAG.
    Shows tasks, prefetch/offload nodes, and control nodes with their connections.
    """
    full_G = nx.DiGraph()
    nodes = plan.get('nodes', [])

    array_sizes_gb = {}
    for array in profile.get('arrays', []):
        array_id = array['id']
        size_bytes = array['size']
        size_gb = size_bytes / (1024**3)
        array_sizes_gb[array_id] = size_gb

    task_times = {}
    cumulative_time = 0
    for i, task_group in enumerate(profile['taskGroups']):
        duration = task_group.get('runningTime', 0.1)
        task_times[i] = {
            'start': cumulative_time,
            'end': cumulative_time + duration,
            'duration': duration
        }
        cumulative_time += duration

    node_info = {}
    for node in nodes:
        node_id = node['nodeId']
        node_type = node['nodeType']

        if node_type == 0:  # Control node
            node_info[node_id] = {'type': 'control', 'label': f'C{node_id}'}
            full_G.add_node(node_id)

        elif node_type == 1:  # Task node
            task_id = node.get('taskId', -1)
            if task_id >= 0 and task_id in task_times:
                node_info[node_id] = {
                    'type': 'task',
                    'label': f"T{task_id}",
                    'task_id': task_id,
                    'time': task_times[task_id]['start'],
                    'duration': task_times[task_id]['duration']
                }
            else:
                node_info[node_id] = {'type': 'task', 'label': f"T{task_id}", 'task_id': task_id}
            full_G.add_node(node_id)

        elif node_type == 2:  # Migration node
            array_id = node.get('arrayId', -1)
            direction = node.get('direction', -1)
            size_gb = array_sizes_gb.get(array_id, 0)
            label = f"A{array_id}\n{size_gb:.1f}GB"
            mig_type = 'prefetch' if direction == 0 else 'offload'
            node_info[node_id] = {'type': mig_type, 'label': label, 'array_id': array_id, 'size_gb': size_gb}
            full_G.add_node(node_id)

        for target in node.get('edges', []):
            full_G.add_edge(node_id, target)

    G = nx.DiGraph()

    for node_id, info in node_info.items():
        if info['type'] in ['task', 'prefetch', 'offload', 'control']:
            G.add_node(node_id, **info)

    for node_id in G.nodes():
        for target in full_G.successors(node_id):
            if target in G.nodes():
                G.add_edge(node_id, target)

    task_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'task']
    prefetch_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'prefetch']
    offload_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'offload']
    control_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'control']

    overlapping_nodes = find_overlapping_migrations(G, node_info, profile)

    prefetch_overlap = [n for n in prefetch_nodes if n in overlapping_nodes]
    prefetch_non_overlap = [n for n in prefetch_nodes if n not in overlapping_nodes]
    offload_overlap = [n for n in offload_nodes if n in overlapping_nodes]
    offload_non_overlap = [n for n in offload_nodes if n not in overlapping_nodes]

    if verbose:
        print(f"  Node counts: {len(task_nodes)} tasks, {len(control_nodes)} control, "
              f"{len(prefetch_nodes)} prefetch, {len(offload_nodes)} offload")
        print(f"  Overlapping: {len(prefetch_overlap)} prefetch, {len(offload_overlap)} offload")

    # Calculate layout
    pos = {}
    node_levels = {}
    for node_id in nx.topological_sort(G):
        predecessors = list(G.predecessors(node_id))
        if not predecessors:
            node_levels[node_id] = 0
        else:
            node_levels[node_id] = max(node_levels[p] for p in predecessors) + 1

    levels = {}
    for node_id, level in node_levels.items():
        if level not in levels:
            levels[level] = []
        levels[level].append(node_id)

    num_levels = len(levels)
    levels_per_row = (num_levels + 2) // 3

    x_spacing = 4.0
    row_height = 60.0

    for level in sorted(levels.keys()):
        level_nodes = levels[level]

        row = level // levels_per_row
        col = level % levels_per_row

        x_position = col * x_spacing
        y_base = -row * row_height

        level_tasks = [n for n in level_nodes if node_info[n]['type'] == 'task']
        level_control = [n for n in level_nodes if node_info[n]['type'] == 'control']
        level_prefetch = [n for n in level_nodes if node_info[n]['type'] == 'prefetch']
        level_offload = [n for n in level_nodes if node_info[n]['type'] == 'offload']

        for i, node_id in enumerate(sorted(level_tasks, key=lambda n: node_info[n].get('task_id', 0))):
            y_offset = (i - len(level_tasks)/2 + 0.5) * 0.5
            pos[node_id] = (x_position, y_base + 0.8 + y_offset)

        for i, node_id in enumerate(level_control):
            y_offset = (i - len(level_control)/2 + 0.5) * 0.4
            pos[node_id] = (x_position, y_base - 0.8 + y_offset)

        for i, node_id in enumerate(level_prefetch):
            vertical_offset = (i - len(level_prefetch)/2 + 0.5) * 10.0
            pos[node_id] = (x_position, y_base + 8.0 + vertical_offset)

        for i, node_id in enumerate(level_offload):
            vertical_offset = (i - len(level_offload)/2 + 0.5) * 10.0
            pos[node_id] = (x_position, y_base - 8.0 - vertical_offset)

    # Create figure (A4 landscape)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    # Draw edges
    overlap_edges = [(u, v) for u, v in G.edges() if u in overlapping_nodes or v in overlapping_nodes]
    non_overlap_edges = [(u, v) for u, v in G.edges() if u not in overlapping_nodes and v not in overlapping_nodes]

    if non_overlap_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=non_overlap_edges,
                              edge_color='#7f8c8d', arrows=True, arrowsize=8,
                              arrowstyle='-|>', width=0.8, alpha=0.5,
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=10, min_target_margin=10)

    if overlap_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=overlap_edges,
                              edge_color='black', arrows=True, arrowsize=10,
                              arrowstyle='-|>', width=1.5, alpha=0.9,
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=10, min_target_margin=10)

    # Draw nodes
    task_pos_dict = {n: pos[n] for n in task_nodes if n in pos}
    if task_pos_dict:
        nx.draw_networkx_nodes(G, task_pos_dict, nodelist=task_nodes,
                              node_color='#3498db', node_shape='s',
                              node_size=800, edgecolors='black', linewidths=1.5, ax=ax)

    prefetch_non_overlap_pos = {n: pos[n] for n in prefetch_non_overlap if n in pos}
    if prefetch_non_overlap_pos:
        nx.draw_networkx_nodes(G, prefetch_non_overlap_pos, nodelist=prefetch_non_overlap,
                              node_color='#7dcea0', node_shape='v',
                              node_size=800, edgecolors='black', linewidths=1.5, ax=ax)

    prefetch_overlap_pos = {n: pos[n] for n in prefetch_overlap if n in pos}
    if prefetch_overlap_pos:
        nx.draw_networkx_nodes(G, prefetch_overlap_pos, nodelist=prefetch_overlap,
                              node_color='#1a7a3e', node_shape='v',
                              node_size=800, edgecolors='black', linewidths=1.5, ax=ax)

    offload_non_overlap_pos = {n: pos[n] for n in offload_non_overlap if n in pos}
    if offload_non_overlap_pos:
        nx.draw_networkx_nodes(G, offload_non_overlap_pos, nodelist=offload_non_overlap,
                              node_color='#f1948a', node_shape='^',
                              node_size=800, edgecolors='black', linewidths=1.5, ax=ax)

    offload_overlap_pos = {n: pos[n] for n in offload_overlap if n in pos}
    if offload_overlap_pos:
        nx.draw_networkx_nodes(G, offload_overlap_pos, nodelist=offload_overlap,
                              node_color='#922b21', node_shape='^',
                              node_size=800, edgecolors='black', linewidths=1.5, ax=ax)

    control_pos_dict = {n: pos[n] for n in control_nodes if n in pos}
    if control_pos_dict:
        nx.draw_networkx_nodes(G, control_pos_dict, nodelist=control_nodes,
                              node_color='#95a5a6', node_shape='d',
                              node_size=200, edgecolors='black', linewidths=0.8, ax=ax)

    # Draw labels
    labels = {}
    for n in G.nodes():
        if n in node_info:
            if node_info[n]['type'] == 'control':
                continue
            elif node_info[n]['type'] == 'task' and 'duration' in node_info[n]:
                labels[n] = f"{node_info[n]['label']}\n{node_info[n]['duration']:.2f}s"
            else:
                labels[n] = node_info[n]['label']

    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                           font_size=5, font_weight='bold', font_color='white')

    ax.set_title('Execution Plan DAG: Nodes and Edges', fontsize=10, fontweight='bold', pad=10)
    ax.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label=f'Tasks ({len(task_nodes)})', edgecolor='black'),
        mpatches.Patch(facecolor='#1a7a3e', label=f'Prefetch overlap ({len(prefetch_overlap)})', edgecolor='black'),
        mpatches.Patch(facecolor='#7dcea0', label=f'Prefetch non-overlap ({len(prefetch_non_overlap)})', edgecolor='black'),
        mpatches.Patch(facecolor='#922b21', label=f'Offload overlap ({len(offload_overlap)})', edgecolor='black'),
        mpatches.Patch(facecolor='#f1948a', label=f'Offload non-overlap ({len(offload_non_overlap)})', edgecolor='black'),
        mpatches.Patch(facecolor='#95a5a6', label=f'Control ({len(control_nodes)})', edgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.95,
              bbox_to_anchor=(1.0, 1.15))

    summary = f"Total nodes: {len(G.nodes())}\nTotal edges: {len(G.edges())}"
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
           fontsize=7, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / 'plan_dag_graph.pdf'
        png_path = output_dir / 'plan_dag_graph.png'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {pdf_path}")
        print(f"  Saved: {png_path}")

    plt.close()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize FRUGAL execution plans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both visualizations
  python scripts/visualize.py --profile profile.json --plan plan.json

  # Generate only task dependency graph (no plan needed)
  python scripts/visualize.py --profile profile.json --only-dependency

  # Generate only execution plan DAG
  python scripts/visualize.py --profile profile.json --plan plan.json --only-plan

  # Specify output directory
  python scripts/visualize.py --profile profile.json --plan plan.json --output results/viz/
        """
    )
    parser.add_argument('--profile', required=True, help='Profile JSON file')
    parser.add_argument('--plan', help='Optimized plan JSON file')
    parser.add_argument('--output', default='results/ablation/visualization',
                       help='Output directory for plots (default: results/ablation/visualization)')
    parser.add_argument('--only-dependency', action='store_true',
                       help='Generate only task dependency graph')
    parser.add_argument('--only-plan', action='store_true',
                       help='Generate only execution plan DAG')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information')

    args = parser.parse_args()

    profile_path = Path(args.profile)
    plan_path = Path(args.plan) if args.plan else None
    output_dir = Path(args.output)

    print("=" * 60)
    print("FRUGAL Execution Plan Visualizer")
    print("=" * 60)

    print(f"\nLoading profile: {profile_path}")
    profile = load_profile(profile_path)

    plan = None
    task_order = None
    if plan_path:
        print(f"Loading plan: {plan_path}")
        plan = load_plan(plan_path)
        if plan:
            task_order = extract_task_order(plan, profile)
            if args.verbose:
                print(f"Extracted execution order for {len(task_order)} taskGroups")

    print_graph_summary(profile)

    # Generate visualizations
    if not args.only_plan:
        print("\n[1/2] Generating task dependency graph...")
        create_task_dependency_graph(profile, task_order, output_dir)

    if not args.only_dependency:
        if plan:
            print("\n[2/2] Generating execution plan DAG...")
            visualize_plan_dag(profile, plan, output_dir, verbose=args.verbose)
        else:
            print("\n[2/2] Skipping execution plan DAG (no plan file provided)")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Visualize the execution plan as a DAG showing nodes and edges.
This shows the actual graph structure from the optimized plan.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from pathlib import Path
import argparse

def load_profile(profile_file):
    """Load profiling data JSON."""
    with open(profile_file, 'r') as f:
        return json.load(f)

def load_plan(plan_file):
    """Load optimized plan JSON."""
    with open(plan_file, 'r') as f:
        return json.load(f)

def find_overlapping_migrations(G, node_info, profile):
    """
    Detect which migration nodes (prefetch/offload) overlap with task execution.

    Overlap means: the migration starts after task X and finishes before task Y,
    where there are other tasks executing between X and Y.

    Returns a set of node_ids that are overlapping.
    """
    # Build taskId -> taskGroupId mapping
    taskid_to_groupid = {}
    for tg in profile.get('taskGroups', []):
        for task_id in tg.get('taskIds', []):
            taskid_to_groupid[task_id] = tg['id']

    # Get task execution order from topological sort
    topo_order = list(nx.topological_sort(G))
    task_exec_order = {}
    order = 0
    for node_id in topo_order:
        if node_info.get(node_id, {}).get('type') == 'task':
            task_exec_order[node_id] = order
            order += 1

    def find_task_predecessors(node_id, visited=None):
        """Find all task nodes that are predecessors (through control nodes)."""
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
        """Find all task nodes that are successors (through control nodes)."""
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

        # Get execution order of predecessor and successor tasks
        pred_orders = [task_exec_order[t] for t in task_preds if t in task_exec_order]
        succ_orders = [task_exec_order[t] for t in task_succs if t in task_exec_order]

        # Migration starts after max(pred_orders), finishes before min(succ_orders)
        start_after = max(pred_orders) if pred_orders else -1
        finish_before = min(succ_orders) if succ_orders else len(task_exec_order)

        # Overlap if there are tasks between start and finish
        tasks_spanned = finish_before - start_after - 1

        if tasks_spanned > 0:
            overlapping_nodes.add(node_id)

    return overlapping_nodes


def visualize_plan_dag(profile, plan, output_dir):
    """
    Visualize the execution plan as a DAG.
    Shows only tasks and migrations (prefetch/offload) with their connections.
    Control nodes are removed, edges are connected through them.

    Overlapping migrations (those that span multiple tasks) are shown in different colors.
    """

    # Build full graph from plan
    full_G = nx.DiGraph()

    nodes = plan.get('nodes', [])

    # Get array sizes from profile (in GB)
    array_sizes_gb = {}
    for array in profile.get('arrays', []):
        array_id = array['id']
        size_bytes = array['size']
        size_gb = size_bytes / (1024**3)  # Convert bytes to GB
        array_sizes_gb[array_id] = size_gb

    # Get task times for positioning
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

    # First pass: identify node types and build full graph
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
            if direction == 0:
                label = f"A{array_id}\n{size_gb:.1f}GB"
                mig_type = 'prefetch'
            else:
                label = f"A{array_id}\n{size_gb:.1f}GB"
                mig_type = 'offload'
            node_info[node_id] = {'type': mig_type, 'label': label, 'array_id': array_id, 'size_gb': size_gb}
            full_G.add_node(node_id)

        # Add edges to full graph
        for target in node.get('edges', []):
            full_G.add_edge(node_id, target)

    # Build graph: tasks, migrations, and control nodes
    G = nx.DiGraph()

    # Add task, migration, and control nodes
    for node_id, info in node_info.items():
        if info['type'] in ['task', 'prefetch', 'offload', 'control']:
            G.add_node(node_id, **info)

    # Add edges directly from the plan (no traversal needed)
    edges_added = 0

    for node_id in G.nodes():
        # Add edges from this node to its direct successors (if they're in G)
        for target in full_G.successors(node_id):
            if target in G.nodes():
                G.add_edge(node_id, target)
                edges_added += 1

    print(f"\nAdded {edges_added} edges between {len(G.nodes())} nodes (including {len([n for n in G.nodes() if node_info[n]['type'] == 'control'])} control nodes)")

    # Analyze edge structure for all offload nodes
    offload_nodes_list = [n for n in G.nodes() if node_info[n]['type'] == 'offload']
    print(f"\n=== Analyzing ALL offload nodes edge structure ===")
    print(f"Total offload nodes: {len(offload_nodes_list)}")

    incoming_counts = {}
    for off_node in offload_nodes_list:
        label = node_info[off_node]['label']
        in_edges = list(G.predecessors(off_node))
        out_edges = list(G.successors(off_node))
        in_count = len(in_edges)

        if in_count not in incoming_counts:
            incoming_counts[in_count] = []
        incoming_counts[in_count].append(label)

        print(f"\n{label} offload (node {off_node}):")
        print(f"  Incoming: {in_count} edges")
        if in_edges:
            for pred in in_edges:
                pred_type = node_info[pred]['type']
                pred_label = node_info[pred]['label']
                print(f"    <- {pred_label} ({pred_type})")
        print(f"  Outgoing: {len(out_edges)} edges")
        if out_edges:
            for succ in out_edges:
                succ_type = node_info[succ]['type']
                succ_label = node_info[succ]['label']
                print(f"    -> {succ_label} ({succ_type})")

    print(f"\n=== Summary of incoming edge counts for offload nodes ===")
    for count in sorted(incoming_counts.keys()):
        arrays = incoming_counts[count]
        print(f"{count} incoming: {len(arrays)} offload nodes - {arrays}")

    # Also analyze prefetch nodes
    prefetch_nodes_list = [n for n in G.nodes() if node_info[n]['type'] == 'prefetch']
    print(f"\n=== Analyzing ALL prefetch nodes edge structure ===")
    print(f"Total prefetch nodes: {len(prefetch_nodes_list)}")

    prefetch_incoming_counts = {}
    for pref_node in prefetch_nodes_list:
        label = node_info[pref_node]['label']
        in_edges = list(G.predecessors(pref_node))
        out_edges = list(G.successors(pref_node))
        in_count = len(in_edges)

        if in_count not in prefetch_incoming_counts:
            prefetch_incoming_counts[in_count] = []
        prefetch_incoming_counts[in_count].append(label)

        print(f"\n{label} prefetch (node {pref_node}):")
        print(f"  Incoming: {in_count} edges")
        if in_edges:
            for pred in in_edges:
                pred_type = node_info[pred]['type']
                pred_label = node_info[pred]['label']
                print(f"    <- {pred_label} ({pred_type})")
        print(f"  Outgoing: {len(out_edges)} edges")
        if out_edges:
            for succ in out_edges[:3]:  # Show first 3
                succ_type = node_info[succ]['type']
                succ_label = node_info[succ]['label']
                print(f"    -> {succ_label} ({succ_type})")
            if len(out_edges) > 3:
                print(f"    ... and {len(out_edges)-3} more")

    print(f"\n=== Summary of incoming edge counts for prefetch nodes ===")
    for count in sorted(prefetch_incoming_counts.keys()):
        arrays = prefetch_incoming_counts[count]
        print(f"{count} incoming: {len(arrays)} prefetch nodes - {arrays}")

    # Create layout based on task execution order and topological sort
    pos = {}

    # Group nodes by type
    task_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'task']
    prefetch_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'prefetch']
    offload_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'offload']
    control_nodes = [n for n in G.nodes() if node_info.get(n, {}).get('type') == 'control']

    # Detect overlapping migrations
    overlapping_nodes = find_overlapping_migrations(G, node_info, profile)

    # Split prefetch/offload into overlapping and non-overlapping
    prefetch_overlap = [n for n in prefetch_nodes if n in overlapping_nodes]
    prefetch_non_overlap = [n for n in prefetch_nodes if n not in overlapping_nodes]
    offload_overlap = [n for n in offload_nodes if n in overlapping_nodes]
    offload_non_overlap = [n for n in offload_nodes if n not in overlapping_nodes]

    print(f"\nNode counts: {len(task_nodes)} tasks, {len(control_nodes)} control, {len(prefetch_nodes)} prefetch, {len(offload_nodes)} offload")
    print(f"Overlapping migrations: {len(prefetch_overlap)} prefetch, {len(offload_overlap)} offload")
    print(f"Non-overlapping migrations: {len(prefetch_non_overlap)} prefetch, {len(offload_non_overlap)} offload")

    # Use the full graph (with control nodes) for topological ordering
    # This gives us the proper execution order
    y_base = 0

    # Calculate execution level (longest path from any source) for ALL nodes
    node_levels = {}
    for node_id in nx.topological_sort(G):
        predecessors = list(G.predecessors(node_id))
        if not predecessors:
            node_levels[node_id] = 0
        else:
            node_levels[node_id] = max(node_levels[p] for p in predecessors) + 1

    # Group ALL nodes by level
    levels = {}
    for node_id, level in node_levels.items():
        if level not in levels:
            levels[level] = []
        levels[level].append(node_id)

    # Split levels into 3 rows for A4 page layout
    num_levels = len(levels)
    levels_per_row = (num_levels + 2) // 3  # Divide into 3 rows, round up

    print(f"\n=== Layout: {num_levels} levels split into 3 rows ({levels_per_row} levels per row) ===")

    # Position all nodes: split into 3 rows
    x_spacing = 4.0  # Spacing for A4 width
    row_height = 60.0  # Vertical spacing between rows (increased for 10 unit node spread)

    for level in sorted(levels.keys()):
        level_nodes = levels[level]

        # Determine which row this level belongs to
        row = level // levels_per_row
        col = level % levels_per_row

        # Calculate position
        x_position = col * x_spacing
        y_base = -row * row_height  # Each row goes down

        # Separate by type for vertical positioning
        level_tasks = [n for n in level_nodes if node_info[n]['type'] == 'task']
        level_control = [n for n in level_nodes if node_info[n]['type'] == 'control']
        level_prefetch = [n for n in level_nodes if node_info[n]['type'] == 'prefetch']
        level_offload = [n for n in level_nodes if node_info[n]['type'] == 'offload']

        # Position tasks at y_base + 0.8 (slightly above center of row)
        for i, node_id in enumerate(sorted(level_tasks, key=lambda n: node_info[n].get('task_id', 0))):
            y_offset = (i - len(level_tasks)/2 + 0.5) * 0.5
            pos[node_id] = (x_position, y_base + 0.8 + y_offset)

        # Position control nodes at y_base - 0.8 (slightly below center of row)
        for i, node_id in enumerate(level_control):
            y_offset = (i - len(level_control)/2 + 0.5) * 0.4
            pos[node_id] = (x_position, y_base - 0.8 + y_offset)

        # Position prefetch above (y > 0) - spread vertically if multiple
        for i, node_id in enumerate(level_prefetch):
            # Spread prefetch nodes vertically with 10 unit spacing
            vertical_offset = (i - len(level_prefetch)/2 + 0.5) * 10.0
            pos[node_id] = (x_position, y_base + 8.0 + vertical_offset)

        # Position offload below (y < 0) - spread vertically if multiple
        for i, node_id in enumerate(level_offload):
            # Spread offload nodes vertically with 10 unit spacing
            vertical_offset = (i - len(level_offload)/2 + 0.5) * 10.0
            pos[node_id] = (x_position, y_base - 8.0 - vertical_offset)

    print(f"\n=== Execution levels (left to right, {len(levels)} levels total) ===")

    # Create visualization - A4 page size (landscape)
    # A4 landscape: 11.69 x 8.27 inches
    fig_width = 11.69
    fig_height = 8.27
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    print(f"Figure size: {fig_width} x {fig_height} inches (A4 landscape)")

    # Separate edges into overlapping and non-overlapping
    # Overlapping edges: any edge connected to an overlapping migration node
    overlap_edges = []
    non_overlap_edges = []
    for u, v in G.edges():
        if u in overlapping_nodes or v in overlapping_nodes:
            overlap_edges.append((u, v))
        else:
            non_overlap_edges.append((u, v))

    print(f"Edges: {len(overlap_edges)} overlapping, {len(non_overlap_edges)} non-overlapping")

    # Draw non-overlapping edges in gray
    if non_overlap_edges:
        nx.draw_networkx_edges(G, pos, ax=ax,
                              edgelist=non_overlap_edges,
                              edge_color='#7f8c8d',
                              arrows=True,
                              arrowsize=8,
                              arrowstyle='-|>',
                              width=0.8,
                              alpha=0.5,
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=10,
                              min_target_margin=10)

    # Draw overlapping edges in dark black
    if overlap_edges:
        nx.draw_networkx_edges(G, pos, ax=ax,
                              edgelist=overlap_edges,
                              edge_color='black',
                              arrows=True,
                              arrowsize=10,
                              arrowstyle='-|>',
                              width=1.5,
                              alpha=0.9,
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=10,
                              min_target_margin=10)

    # All tasks same size, draw with execution time labels - smaller
    task_pos_dict = {n: pos[n] for n in task_nodes if n in pos}
    if task_pos_dict:
        # Draw task nodes (blue rectangles) - smaller size
        nx.draw_networkx_nodes(G, task_pos_dict, nodelist=task_nodes,
                              node_color='#3498db', node_shape='s',
                              node_size=800, edgecolors='black', linewidths=1.5, ax=ax)

    # Draw prefetch nodes - different colors for overlapping vs non-overlapping
    # Overlapping prefetch: darker green (#1a7a3e), Non-overlapping: lighter green (#7dcea0)
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

    # Draw offload nodes - different colors for overlapping vs non-overlapping
    # Overlapping offload: darker red (#922b21), Non-overlapping: lighter red (#f1948a)
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

    # Draw control nodes (gray diamonds) - much smaller (2x smaller than others)
    control_pos_dict = {n: pos[n] for n in control_nodes if n in pos}
    if control_pos_dict:
        nx.draw_networkx_nodes(G, control_pos_dict, nodelist=control_nodes,
                              node_color='#95a5a6', node_shape='d',
                              node_size=200, edgecolors='black', linewidths=0.8, ax=ax)

    # Draw labels - tasks show execution time, migrations show array ID, control nodes no labels
    labels = {}
    for n in G.nodes():
        if n in node_info:
            if node_info[n]['type'] == 'control':
                # No labels for control nodes (too cluttered)
                continue
            elif node_info[n]['type'] == 'task' and 'duration' in node_info[n]:
                # Task label with execution time
                labels[n] = f"{node_info[n]['label']}\n{node_info[n]['duration']:.2f}s"
            else:
                # Migration nodes just show array ID
                labels[n] = node_info[n]['label']

    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                           font_size=5, font_weight='bold', font_color='white')

    # Styling
    ax.set_title('Execution Plan DAG: Nodes and Edges', fontsize=10, fontweight='bold', pad=10)
    ax.axis('off')

    # Legend - show overlapping vs non-overlapping migrations
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label=f'Tasks ({len(task_nodes)})',
                      edgecolor='black'),
        mpatches.Patch(facecolor='#1a7a3e', label=f'Prefetch overlap ({len(prefetch_overlap)})',
                      edgecolor='black'),
        mpatches.Patch(facecolor='#7dcea0', label=f'Prefetch non-overlap ({len(prefetch_non_overlap)})',
                      edgecolor='black'),
        mpatches.Patch(facecolor='#922b21', label=f'Offload overlap ({len(offload_overlap)})',
                      edgecolor='black'),
        mpatches.Patch(facecolor='#f1948a', label=f'Offload non-overlap ({len(offload_non_overlap)})',
                      edgecolor='black'),
        mpatches.Patch(facecolor='#95a5a6', label=f'Control ({len(control_nodes)})',
                      edgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.95)

    # Summary
    summary = f"Total nodes: {len(G.nodes())}\nTotal edges: {len(G.edges())}"
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
           fontsize=7, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / 'plan_dag_graph.pdf'
    png_path = output_dir / 'plan_dag_graph.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {pdf_path}")
    print(f"✅ Saved: {png_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize plan DAG')
    parser.add_argument('--profile', required=True, help='Profile JSON file')
    parser.add_argument('--plan', required=True, help='Optimized plan JSON file')
    parser.add_argument('--output', default='results/ablation/visualization',
                       help='Output directory for plots')

    args = parser.parse_args()

    profile_path = Path(args.profile)
    plan_path = Path(args.plan)
    output_dir = Path(args.output)

    print(f"Loading profile: {profile_path}")
    profile = load_profile(profile_path)

    print(f"Loading plan: {plan_path}")
    plan = load_plan(plan_path)

    print("Generating DAG visualization...")
    visualize_plan_dag(profile, plan, output_dir)

    print("\n✅ Visualization complete!")

if __name__ == '__main__':
    main()

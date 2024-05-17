import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
import mplcursors
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

def plot_heatmap(data, adapter_base_names, title, output_path, html_output_path, mask_counts=None):
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure width for better readability
    cax = ax.matshow(data, aspect='auto', cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(range(len(data[0])))
    ax.set_yticks(range(len(adapter_base_names)))
    ax.set_yticklabels(adapter_base_names, fontsize=8)  # Reduced font size for y-tick labels
    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_title(title, fontsize=12)

    plt.xticks(rotation=90)

    # Add the mask counts to the right side of the heatmap
    if mask_counts is not None:
        for i, count in enumerate(mask_counts):
            ax.text(len(data[0]), i, f'{int(count)}', va='center', ha='left', fontsize=8, color='black')

    # Add interactive hover feature for the heatmap
    cursor = mplcursors.cursor(cax, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        i, j = sel.index
        sel.annotation.set_text(f'Module: {adapter_base_names[i]}\nTime Step: {j}\nValue: {data[i, j]:.4f}')

    # Save the plot to a file
    plt.savefig(output_path, bbox_inches='tight')

    # Show the plot
    plt.show()
    plt.close(fig)  # Close the figure to free memory

    # Save an interactive version using Plotly
    plotly_data = {
        "z": data,
        "x": list(range(data.shape[1])),
        "y": adapter_base_names,
    }

    fig = go.Figure(data=go.Heatmap(
        z=plotly_data["z"],
        x=plotly_data["x"],
        y=plotly_data["y"],
        colorscale='Viridis'
    ))

    # Add the mask counts to the right side of the Plotly heatmap
    if mask_counts is not None:
        annotations = []
        for i, count in enumerate(mask_counts):
            annotations.append(
                dict(
                    x=len(data[0]),
                    y=i,
                    text=f'{int(count)}',
                    showarrow=False,
                    font=dict(color='black', size=10),
                    xanchor='left',
                    xshift=10
                )
            )
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=title,
        xaxis_title="Time Steps",
        yaxis_title="Modules",
        yaxis=dict(tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
        height=800,
        width=1200  # Increased width to accommodate counts
    )

    # Save interactive HTML
    pio.write_html(fig, file=html_output_path, auto_open=False)

def main(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract metadata for title
    schedule = data.get('schedule', 'N/A')
    allocator = data.get('allocator', 'N/A')
    aggregate = data.get('aggregate', 'N/A')

    # Prepare data for heatmaps
    cum_acts = np.array(data['cum_acts']).T  # Transpose to match the expected shape
    masks = np.array(data['masks']).T  # Transpose to match the expected shape
    adapter_base_names = data['adapter_base_names']

    # Calculate mask counts
    mask_counts = np.sum(masks, axis=1)

    # Create titles
    cum_acts_title = f'Cumulative Activations Heatmap\nSchedule: {schedule}, Allocator: {allocator}, Aggregate: {aggregate}'
    masks_title = f'Masks Heatmap\nSchedule: {schedule}, Allocator: {allocator}, Aggregate: {aggregate}'

    # Define output file paths
    json_dir = os.path.dirname(json_path)
    cum_acts_output_path = os.path.join(json_dir, 'cum_acts_heatmap.png')
    cum_acts_html_output_path = os.path.join(json_dir, 'cum_acts_heatmap.html')
    masks_output_path = os.path.join(json_dir, 'masks_heatmap.png')
    masks_html_output_path = os.path.join(json_dir, 'masks_heatmap.html')

    # Plot and save heatmaps
    plot_heatmap(cum_acts, adapter_base_names, cum_acts_title, cum_acts_output_path, cum_acts_html_output_path)
    plot_heatmap(masks, adapter_base_names, masks_title, masks_output_path, masks_html_output_path, mask_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmaps for cum_acts and masks from a JSON file.")
    parser.add_argument('--json_path', type=str, help="Path to the JSON file containing the data.")
    args = parser.parse_args()

    main(args.json_path)
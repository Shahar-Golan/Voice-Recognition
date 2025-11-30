"""
Create a focused summary plot with key speaker comparison metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

def create_summary_plot():
    """Create a focused summary comparison plot."""
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "outputs", "plots")
    
    # Data from the analysis
    trump_data = {
        'Total Windows': 503,
        'Total Duration (min)': 119.5,
        'Avg Duration (s)': 14.3,
        'Avg Words/Window': 50.9,
        'Speaking Rate (w/s)': 3.59
    }
    
    rogan_data = {
        'Total Windows': 184,
        'Total Duration (min)': 41.6,
        'Avg Duration (s)': 13.6,
        'Avg Words/Window': 46.2,
        'Speaking Rate (w/s)': 3.46
    }
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Speaker Comparison: Key Metrics\nDonald Trump vs Joe Rogan', fontsize=16, fontweight='bold')
    
    # Colors
    trump_color = '#FF6B6B'
    rogan_color = '#4ECDC4'
    
    # 1. Total Windows - separate from Duration to avoid scale issues
    windows_data = [trump_data['Total Windows'], rogan_data['Total Windows']]
    speakers = ['Donald Trump', 'Joe Rogan']
    colors = [trump_color, rogan_color]
    
    bars_windows = ax1.bar(speakers, windows_data, color=colors, alpha=0.8)
    ax1.set_title('Total Windows Comparison', fontweight='bold')
    ax1.set_ylabel('Number of Segments')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars_windows, windows_data):
        height = bar.get_height()
        ax1.annotate(f'{value}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Speaking Time Distribution (Pie Chart)
    total_time = trump_data['Total Duration (min)'] + rogan_data['Total Duration (min)']
    trump_pct = (trump_data['Total Duration (min)'] / total_time) * 100
    rogan_pct = (rogan_data['Total Duration (min)'] / total_time) * 100
    
    wedges, texts, autotexts = ax2.pie([trump_pct, rogan_pct], 
                                      labels=['Donald Trump', 'Joe Rogan'],
                                      colors=[trump_color, rogan_color],
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      explode=(0.05, 0.05))
    ax2.set_title('Speaking Time Distribution', fontweight='bold')
    
    # Add duration info as subtitle
    ax2.text(0, -1.4, f'Trump: {trump_data["Total Duration (min)"]:.1f} min, Rogan: {rogan_data["Total Duration (min)"]:.1f} min', 
             ha='center', va='center', fontsize=11, style='italic')
    
    # 3. Quality Metrics
    categories = ['Avg Duration (s)', 'Avg Words/Window', 'Speaking Rate (w/s)']
    trump_values = [trump_data['Avg Duration (s)'], trump_data['Avg Words/Window'], trump_data['Speaking Rate (w/s)']]
    rogan_values = [rogan_data['Avg Duration (s)'], rogan_data['Avg Words/Window'], rogan_data['Speaking Rate (w/s)']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, trump_values, width, label='Donald Trump', color=trump_color, alpha=0.8)
    bars2 = ax3.bar(x + width/2, rogan_values, width, label='Joe Rogan', color=rogan_color, alpha=0.8)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Values')
    ax3.set_title('Quality Metrics Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    all_values = trump_values + rogan_values
    y_min = min(all_values) * 0.9
    y_max = max(all_values) * 1.15
    ax3.set_ylim(y_min, y_max)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Summary Stats Table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Donald Trump', 'Joe Rogan'],
        ['Total Segments', '503', '184'],
        ['Total Time (min)', '119.5', '41.6'],
        ['Avg Duration (s)', '14.3', '13.6'],
        ['Max Duration (s)', '75.7', '85.3'],
        ['Avg Words/Segment', '50.9', '46.2'],
        ['Speaking Rate (w/s)', '3.59', '3.46'],
        ['Speaking Time %', '74.2%', '25.8%']
    ]
    
    table = ax4.table(cellText=table_data[1:], 
                     colLabels=table_data[0],
                     cellLoc='center',
                     loc='center',
                     colColours=[None, trump_color, rogan_color])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    # Make header row bold
    for i in range(len(table_data[0])):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_facecolor('#2C3E50')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'speaker_summary_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Summary comparison plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    create_summary_plot()

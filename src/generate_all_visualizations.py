#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Visualization Generator for PV Forecasting Data

This script runs all three analysis scripts to generate a complete set of
visualizations for scientific paper publication. It includes:
1. Basic station data analysis
2. Deep learning feature analysis
3. Temporal resolution comparison

Usage:
    python generate_all_visualizations.py
"""

import os
import time
import subprocess
import sys
from datetime import datetime

def run_script(script_path, description):
    """Run a Python script and print its output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run the script and capture its output
        process = subprocess.Popen([sys.executable, script_path], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True)
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\nScript completed successfully in {time.time() - start_time:.2f} seconds.")
        else:
            print(f"\nScript failed with return code {process.returncode}.")
            
    except Exception as e:
        print(f"Error running script: {e}")

def create_visualization_index():
    """Create an HTML index of all visualizations for easy viewing."""
    visualization_dir = 'visualizations'
    
    if not os.path.exists(visualization_dir):
        print("Visualization directory not found. No index created.")
        return
    
    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(visualization_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.join(root, file).replace('\\', '/')
                category = os.path.basename(os.path.dirname(rel_path))
                image_files.append((rel_path, category, file))
    
    # Sort by category and filename
    image_files.sort(key=lambda x: (x[1], x[2]))
    
    # Group by category
    categories = {}
    for path, category, filename in image_files:
        if category not in categories:
            categories[category] = []
        categories[category].append((path, filename))
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PV Forecasting Visualizations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
            border-bottom: 1px solid #3498db;
            padding-bottom: 5px;
        }}
        .visualization-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .visualization-item {{
            margin: 15px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            width: 45%;
            transition: transform 0.3s ease;
        }}
        .visualization-item:hover {{
            transform: scale(1.02);
        }}
        .visualization-item img {{
            width: 100%;
            border: 1px solid #ddd;
        }}
        .visualization-item p {{
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-size: 0.9em;
        }}
        #toc {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        #toc h2 {{
            margin-top: 0;
        }}
        #toc ul {{
            list-style-type: none;
            padding-left: 10px;
        }}
        #toc li {{
            margin: 5px 0;
        }}
        #toc a {{
            text-decoration: none;
            color: #2980b9;
        }}
        #toc a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>PV Forecasting Visualizations for Scientific Paper</h1>
    
    <div id="toc">
        <h2>Table of Contents</h2>
        <ul>
"""

    # Add TOC entries
    for category in categories.keys():
        html_content += f'            <li><a href="#{category}">{category.replace("_", " ").title()}</a></li>\n'
    
    html_content += """        </ul>
    </div>
"""
    
    # Add visualizations by category
    for category, images in categories.items():
        html_content += f"""    <h2 id="{category}">{category.replace("_", " ").title()} ({len(images)} visualizations)</h2>
    <div class="visualization-container">
"""
        
        for path, filename in images:
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            html_content += f"""        <div class="visualization-item">
            <img src="{path}" alt="{name}">
            <p>{name}</p>
        </div>
"""
        
        html_content += "    </div>\n"
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content += f"""    <p class="timestamp">Generated on {timestamp}</p>
</body>
</html>
"""
    
    # Write HTML file
    with open('visualizations/index.html', 'w') as f:
        f.write(html_content)
    
    print(f"Visualization index created at visualizations/index.html")

def main():
    """Run all visualization scripts and create an index."""
    print("Starting comprehensive visualization generation...")
    
    # Create main visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Run each analysis script
    scripts = [
        ('src/analyze_station_data.py', 'Basic Station Data Analysis'),
        ('src/analyze_deep_learning_features.py', 'Deep Learning Feature Analysis'),
        ('src/analyze_temporal_resolution.py', 'Temporal Resolution Comparison')
    ]
    
    for script_path, description in scripts:
        run_script(script_path, description)
    
    # Create an HTML index of all visualizations
    create_visualization_index()
    
    print("\nAll visualizations have been generated successfully!")
    print("Open 'visualizations/index.html' in a web browser to view all visualizations.")

if __name__ == "__main__":
    main()
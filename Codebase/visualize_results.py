import json
from pathlib import Path
from datetime import datetime

def create_results_html():
    results_dir = Path("data/save")
    if not results_dir.exists():
        print("No results found")
        return
    
    runs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not runs:
        print("No training runs found")
        return
    
    latest = runs[-1]
    
    # Collect all results
    results = {}
    for file in latest.glob("*.json"):
        with open(file) as f:
            results[file.stem] = json.load(f)
    
    # Create HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>D2E2S Training Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-label {{ 
            font-size: 0.9em; 
            color: #666; 
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .metric-value {{ 
            font-size: 2.5em; 
            font-weight: bold;
            color: #667eea;
        }}
        .json-section {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .json-section h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        pre {{
            background: #2d3748;
            color: #68d391;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9em;
            line-height: 1.6;
        }}
        .download-btn {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: bold;
            margin: 10px;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s;
        }}
        .download-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 D2E2S Training Results</h1>
            <p>Run: {latest.name}</p>
        </div>
        <div class="content">
"""
    
    # Extract key metrics if available
    if results:
        html += '<div class="metric-grid">'
        
        # Try to extract common metrics
        for key, data in results.items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    if isinstance(value, (int, float)):
                        html += f'''
            <div class="metric-card">
                <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                <div class="metric-value">{value:.4f if isinstance(value, float) else value}</div>
            </div>
'''
        html += '</div>'
    
    # Add JSON sections
    for filename, data in results.items():
        html += f'''
            <div class="json-section">
                <h3>📄 {filename}.json</h3>
                <pre>{json.dumps(data, indent=2)}</pre>
            </div>
'''
    
    # Download buttons
    html += '''
            <div style="text-align: center; margin-top: 40px;">
                <h3 style="color: #667eea; margin-bottom: 20px;">📥 Download Results</h3>
'''
    
    for file in latest.glob("*.json"):
        html += f'<a href="{file.name}" class="download-btn" download>⬇️ {file.name}</a>\n'
    
    html += f'''
            </div>
            <div class="timestamp">
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>
    </div>
</body>
</html>
'''
    
    # Save HTML
    output_file = latest / "results.html"
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✅ Results visualization created: {output_file}")
    print(f"📊 Open in browser or download from Kaggle output")
    
    return output_file

if __name__ == "__main__":
    create_results_html()

# 🚀 Advanced Graph Visualizer Pro

A comprehensive web-based graph visualization tool built with Streamlit and Plotly, featuring mathematical equation plotting, data visualization, statistical analysis, and interactive features.

## ✨ Features

### 📈 Mathematical Graphs
- **Multiple Plot Types**: 2D Line, 2D Scatter, 3D Surface, 3D Scatter, Contour, Heatmap
- **Equation Support**: Enter mathematical equations using standard syntax
- **Quick Examples**: Pre-built examples for common functions (sine waves, parabolas, exponentials, etc.)
- **Interactive Constants**: Real-time adjustment of equation parameters
- **Range Control**: Customizable X and Y axis ranges
- **Resolution Control**: Adjustable plot resolution for quality vs performance

### 📊 Data Visualization
- **Multiple Input Methods**: Upload CSV/Excel files, generate sample data, or paste data directly
- **Sample Data Generation**: Random, sine wave, normal distribution, time series, scatter data
- **Rich Chart Types**: Scatter plots, line plots, bar charts, histograms, box plots, violin plots, heatmaps, 3D scatter, bubble charts, area charts
- **Column Selection**: Dynamic column selection for X, Y, Z axes and color coding

### 🎨 Styling & Themes
- **Multiple Themes**: plotly, plotly_white, plotly_dark, simple_white, presentation, and more
- **Color Schemes**: viridis, plasma, inferno, magma, cividis, RdBu, Spectral, Set1, Set2, Set3
- **Customization Options**: Show/hide legend, grid, animation settings
- **Animation Support**: Configurable animation duration and easing

### 📁 Import/Export
- **Export Formats**: PNG, SVG, PDF, HTML, JSON
- **Import Capability**: Upload previously exported plot JSON files
- **Timestamped Files**: Automatic file naming with timestamps

### 📋 Statistics & Analysis
- **Statistical Summary**: Basic statistics, data info, missing values analysis
- **Correlation Analysis**: Interactive correlation matrix heatmaps
- **Distribution Analysis**: Histograms and box plots for data distribution
- **Outlier Detection**: Automatic outlier detection using IQR method
- **Data Metrics**: Key metrics displayed in cards

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run graph_vizualizer.py
   ```

3. **Open in Browser**: The app will open at `http://localhost:8501`

## 📖 Usage Guide

### Mathematical Equations
1. Go to the "📈 Mathematical Graphs" tab
2. Enter equations in the text area (e.g., `y = sin(x)`, `z = x^2 + y^2`)
3. Use the sidebar sliders to adjust constants
4. Choose plot type and customize ranges
5. View your interactive visualization

### Data Visualization
1. Go to the "📊 Data Visualization" tab
2. Choose your data input method:
   - **Upload**: Upload CSV or Excel files
   - **Generate**: Create sample data with different distributions
   - **Paste**: Paste CSV data directly
3. Select visualization type and configure axes
4. Explore your data with interactive charts

### Styling & Export
1. Go to the "🎨 Styling & Themes" tab
2. Choose theme and color scheme
3. Configure animation and display options
4. Use the "📁 Import/Export" tab to save your work

### Statistical Analysis
1. Load data in the "📊 Data Visualization" tab
2. Go to the "📋 Statistics & Analysis" tab
3. Explore statistical summaries, correlations, and distributions
4. Detect outliers in your data

## 🎯 Examples

### Mathematical Equations
```
# Simple sine wave
y = a*sin(b*x + c)

# Parabola
y = a*x^2 + b*x + c

# 3D surface
z = sin(x) * cos(y)

# Parametric equations
x = a*cos(t)
y = b*sin(t)
```

### Sample Data Types
- **Random**: Random normal distribution data
- **Sine Wave**: Sine wave with noise
- **Normal Distribution**: Multi-dimensional normal data
- **Time Series**: Cumulative random walk data
- **Scatter Data**: Multi-dimensional scatter with size and color

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web framework for the interface
- **Plotly**: Interactive plotting library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SymPy**: Symbolic mathematics
- **OpenPyXL**: Excel file support

### Architecture
- **Tabbed Interface**: Organized into 5 main sections
- **Session State**: Maintains plot history and current state
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: Graceful error handling with user feedback

## 🎨 Customization

### Themes
The app supports multiple Plotly themes:
- `plotly`: Default theme
- `plotly_white`: Clean white background
- `plotly_dark`: Dark theme
- `simple_white`: Minimal white theme
- `presentation`: Presentation-style theme

### Color Schemes
Available color schemes for plots:
- **Sequential**: viridis, plasma, inferno, magma, cividis
- **Diverging**: RdBu, Spectral
- **Qualitative**: Set1, Set2, Set3

## 📊 Performance Tips

1. **Resolution**: Lower resolution for faster rendering of complex plots
2. **Data Size**: For large datasets, consider sampling or aggregation
3. **3D Plots**: 3D visualizations require more computational resources
4. **Animation**: Disable animation for better performance on slower devices

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Plot Not Rendering**: Check equation syntax and variable names
3. **Slow Performance**: Reduce resolution or data size
4. **Export Issues**: Ensure plot is created before attempting export

### Error Messages
- **Equation Parsing Error**: Check mathematical syntax
- **Data Loading Error**: Verify file format and structure
- **Plot Error**: Ensure all required variables are defined

## 🤝 Contributing

Feel free to contribute to this project by:
1. Reporting bugs
2. Suggesting new features
3. Improving documentation
4. Adding new visualization types

## 📄 License

This project is open source and available under the MIT License.

---

**🚀 Advanced Graph Visualizer Pro** - Transform your data into stunning visualizations! 
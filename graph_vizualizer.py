import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sympy import symbols, sympify, lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import io
import base64
from datetime import datetime
import json

# Page Configuration
st.set_page_config(
    page_title="Advanced Graph Visualizer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced Graph Visualizer Pro</h1>
    <p>Create stunning visualizations with mathematical equations, data analysis, and interactive features</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'plot_history' not in st.session_state:
    st.session_state.plot_history = []
if 'current_plot' not in st.session_state:
    st.session_state.current_plot = None

# Sidebar Configuration
st.sidebar.markdown("## 🎛️ Control Panel")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Mathematical Graphs", 
    "📊 Data Visualization", 
    "🎨 Styling & Themes", 
    "📁 Import/Export", 
    "📋 Statistics & Analysis"
])

with tab1:
    st.markdown("### Mathematical Equation Visualizer")
    
    # Equation input with examples
    col1, col2 = st.columns([2, 1])
    
    with col1:
        equation_input = st.text_area(
            "✏️ Enter equation(s)", 
            "y = a*sin(x) + b*x\nz = a*x**2 + b*y**2", 
            height=150,
            help="Enter mathematical equations. Use 'z =' for 3D plots. Examples:\n- y = sin(x)\n- y = a*x^2 + b*x + c\n- z = sin(x) * cos(y)"
        )
    
    with col2:
        st.markdown("#### 📝 Quick Examples")
        examples = {
            "Sine Wave": "y = a*sin(b*x + c)",
            "Parabola": "y = a*x^2 + b*x + c",
            "Exponential": "y = a*exp(b*x)",
            "3D Surface": "z = sin(x) * cos(y)",
            "Spiral": "x = t*cos(t)\ny = t*sin(t)",
            "Parametric": "x = a*cos(t)\ny = b*sin(t)"
        }
        
        selected_example = st.selectbox("Load Example:", list(examples.keys()))
        if st.button("Load Example"):
            equation_input = examples[selected_example]
            st.rerun()
    
    # Parse equations
    equations = [eq.strip() for eq in equation_input.strip().splitlines() if eq.strip()]
    all_symbols = set()
    parsed_eqs = []
    
    transformations = (standard_transformations + (implicit_multiplication_application,))
    x, y, z, t = symbols("x y z t")
    
    for eq in equations:
        if "=" in eq:
            lhs, rhs = eq.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
        else:
            lhs = "y"
            rhs = eq
        
        try:
            expr = parse_expr(rhs, transformations=transformations)
            vars_in_expr = expr.free_symbols
            all_symbols.update(vars_in_expr - {x, y, z, t})
            parsed_eqs.append((lhs, expr))
        except Exception as e:
            st.error(f"Error parsing equation: {eq} - {e}")
    
    # Advanced plotting options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        plot_type = st.selectbox(
            "📊 Plot Type",
            ["2D Line", "2D Scatter", "3D Surface", "3D Scatter", "Contour", "Heatmap"],
            help="Choose the type of visualization"
        )
    
    with col2:
        x_range = st.slider("X Range", -20.0, 20.0, (-10.0, 10.0), step=0.5)
        y_range = st.slider("Y Range", -20.0, 20.0, (-10.0, 10.0), step=0.5)
    
    with col3:
        resolution = st.slider("Resolution", 50, 1000, 200, help="Number of points for plotting")
    
    # Constants sliders
    st.sidebar.markdown("### 🎚️ Constants")
    const_values = {}
    for sym in sorted(all_symbols, key=lambda s: s.name):
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            const_values[sym.name] = st.slider(
                f"{sym.name}",
                min_value=-20.0,
                max_value=20.0,
                value=1.0,
                step=0.1,
                format="%.1f"
            )
        with col2:
            if st.button(f"Reset {sym.name}", key=f"reset_{sym.name}"):
                const_values[sym.name] = 1.0
    
    # Generate plot
    if parsed_eqs:
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        
        fig = go.Figure()
        
        for i, (lhs, expr) in enumerate(parsed_eqs):
            f_vars = sorted(expr.free_symbols, key=lambda s: s.name)
            f = lambdify(f_vars, expr, modules=["numpy"])
            
            try:
                if lhs == "z" or plot_type in ["3D Surface", "3D Scatter", "Contour", "Heatmap"]:
                    # 3D plotting
                    X, Y = np.meshgrid(x_vals, y_vals)
                    arg_values = {str(s): const_values.get(str(s), X if s == x else Y) for s in f_vars}
                    Z = f(**arg_values)
                    
                    if plot_type == "3D Surface":
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z, 
                            name=f"{lhs} = {expr}",
                            colorscale='viridis',
                            showscale=True
                        ))
                    elif plot_type == "3D Scatter":
                        fig.add_trace(go.Scatter3d(
                            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                            mode='markers',
                            name=f"{lhs} = {expr}",
                            marker=dict(size=2)
                        ))
                    elif plot_type == "Contour":
                        fig.add_trace(go.Contour(
                            x=x_vals, y=y_vals, z=Z,
                            name=f"{lhs} = {expr}",
                            colorscale='viridis'
                        ))
                    elif plot_type == "Heatmap":
                        fig.add_trace(go.Heatmap(
                            x=x_vals, y=y_vals, z=Z,
                            name=f"{lhs} = {expr}",
                            colorscale='viridis'
                        ))
                else:
                    # 2D plotting
                    arg_values = {str(s): const_values.get(str(s), x_vals) for s in f_vars}
                    Y = f(**arg_values)
                    
                    if plot_type == "2D Line":
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=Y, 
                            mode='lines', 
                            name=f"{lhs} = {expr}",
                            line=dict(width=3)
                        ))
                    elif plot_type == "2D Scatter":
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=Y, 
                            mode='markers', 
                            name=f"{lhs} = {expr}",
                            marker=dict(size=5)
                        ))
                        
            except Exception as e:
                st.error(f"Plot error for {lhs} = {expr}: {e}")
        
        # Update layout
        if plot_type in ["3D Surface", "3D Scatter"]:
            fig.update_layout(
                title=f"{plot_type} Visualization",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y', 
                    zaxis_title='Z',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                height=600,
                margin=dict(l=0, r=0, b=0, t=30)
            )
        else:
            fig.update_layout(
                title=f"{plot_type} Visualization",
                xaxis_title="X",
                yaxis_title="Y",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40)
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save to session state
        st.session_state.current_plot = fig

with tab2:
    st.markdown("### Data Visualization")
    
    # Data input methods
    data_input_method = st.radio(
        "📥 Data Input Method",
        ["Upload CSV/Excel", "Generate Sample Data", "Paste Data"]
    )
    
    df = None
    
    if data_input_method == "Upload CSV/Excel":
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload your data file"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    elif data_input_method == "Generate Sample Data":
        sample_type = st.selectbox(
            "Sample Data Type",
            ["Random", "Sine Wave", "Normal Distribution", "Time Series", "Scatter Data"]
        )
        
        n_points = st.slider("Number of points", 10, 1000, 100)
        
        if sample_type == "Random":
            df = pd.DataFrame({
                'x': np.random.randn(n_points),
                'y': np.random.randn(n_points),
                'category': np.random.choice(['A', 'B', 'C'], n_points)
            })
        elif sample_type == "Sine Wave":
            x = np.linspace(0, 4*np.pi, n_points)
            df = pd.DataFrame({
                'x': x,
                'y': np.sin(x) + 0.1*np.random.randn(n_points),
                'amplitude': np.random.uniform(0.5, 2, n_points)
            })
        elif sample_type == "Normal Distribution":
            df = pd.DataFrame({
                'x': np.random.normal(0, 1, n_points),
                'y': np.random.normal(0, 1, n_points),
                'z': np.random.normal(0, 1, n_points)
            })
        elif sample_type == "Time Series":
            dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'value': np.cumsum(np.random.randn(n_points)),
                'volume': np.random.randint(100, 1000, n_points)
            })
        elif sample_type == "Scatter Data":
            df = pd.DataFrame({
                'x': np.random.randn(n_points),
                'y': np.random.randn(n_points),
                'size': np.random.randint(10, 100, n_points),
                'color': np.random.choice(['red', 'blue', 'green'], n_points)
            })
    
    elif data_input_method == "Paste Data":
        data_text = st.text_area(
            "Paste your data (CSV format)",
            "x,y,z\n1,2,3\n4,5,6\n7,8,9",
            height=200
        )
        if data_text:
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(data_text))
                st.success("✅ Data parsed successfully!")
            except Exception as e:
                st.error(f"Error parsing data: {e}")
    
    # Display data and create visualizations
    if df is not None:
        st.markdown("#### 📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Visualization options
        viz_type = st.selectbox(
            "📊 Visualization Type",
            ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", 
             "Heatmap", "3D Scatter", "Bubble Chart", "Area Chart"]
        )
        
        # Column selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        if viz_type in ["Scatter Plot", "Line Plot", "3D Scatter"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X Axis", all_cols, index=0 if len(all_cols) > 0 else None)
            with col2:
                y_col = st.selectbox("Y Axis", all_cols, index=1 if len(all_cols) > 1 else 0)
            with col3:
                color_col = st.selectbox("Color By", [None] + all_cols)
            
            if viz_type == "3D Scatter" and len(numeric_cols) >= 3:
                z_col = st.selectbox("Z Axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
            else:
                z_col = None
        
        elif viz_type in ["Bar Chart", "Histogram"]:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X Axis", all_cols)
            with col2:
                y_col = st.selectbox("Y Axis", numeric_cols) if viz_type == "Bar Chart" else None
        
        elif viz_type == "Box Plot":
            x_col = st.selectbox("Category Column", all_cols)
            y_col = st.selectbox("Value Column", numeric_cols)
        
        elif viz_type == "Heatmap":
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for heatmap")
        
        # Create the visualization
        if viz_type != "Heatmap":
            try:
                if viz_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title="Scatter Plot")
                elif viz_type == "Line Plot":
                    fig = px.line(df, x=x_col, y=y_col, color=color_col, title="Line Plot")
                elif viz_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, title="Bar Chart")
                elif viz_type == "Histogram":
                    fig = px.histogram(df, x=x_col, title="Histogram")
                elif viz_type == "Box Plot":
                    fig = px.box(df, x=x_col, y=y_col, title="Box Plot")
                elif viz_type == "Violin Plot":
                    fig = px.violin(df, x=x_col, y=y_col, title="Violin Plot")
                elif viz_type == "3D Scatter":
                    if z_col:
                        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, title="3D Scatter Plot")
                    else:
                        st.warning("Need 3 numeric columns for 3D scatter plot")
                        fig = None
                elif viz_type == "Bubble Chart":
                    size_col = st.selectbox("Size Column", numeric_cols)
                    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col, title="Bubble Chart")
                elif viz_type == "Area Chart":
                    fig = px.area(df, x=x_col, y=y_col, color=color_col, title="Area Chart")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.current_plot = fig
                    
            except Exception as e:
                st.error(f"Error creating visualization: {e}")

with tab3:
    st.markdown("### 🎨 Styling & Themes")
    
    if st.session_state.current_plot is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Theme Settings")
            theme = st.selectbox(
                "Theme",
                ["plotly", "plotly_white", "plotly_dark", "simple_white", "presentation", "xgridoff", "ygridoff", "gridon"]
            )
            
            color_scheme = st.selectbox(
                "Color Scheme",
                ["viridis", "plasma", "inferno", "magma", "cividis", "RdBu", "Spectral", "Set1", "Set2", "Set3"]
            )
            
            show_legend = st.checkbox("Show Legend", value=True)
            show_grid = st.checkbox("Show Grid", value=True)
            
            # Animation settings
            st.markdown("#### Animation")
            animate = st.checkbox("Enable Animation", value=False)
            if animate:
                animation_duration = st.slider("Animation Duration (ms)", 500, 3000, 1000)
                animation_easing = st.selectbox("Easing", ["linear", "quad", "cubic", "sin", "exp"])
        
        with col2:
            st.markdown("#### Preview")
            fig = st.session_state.current_plot
            
            # Apply theme
            fig.update_layout(
                template=theme,
                showlegend=show_legend,
                xaxis=dict(showgrid=show_grid),
                yaxis=dict(showgrid=show_grid)
            )
            
            if animate:
                fig.update_layout(
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        buttons=[dict(
                            label="Play",
                            method="animate",
                            args=[None, dict(
                                frame=dict(duration=animation_duration, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=animation_duration, easing=animation_easing)
                            )]
                        )]
                    )]
                )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Create a plot first to access styling options")

with tab4:
    st.markdown("### 📁 Import/Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Export Options")
        
        if st.session_state.current_plot is not None:
            export_format = st.selectbox(
                "Export Format",
                ["PNG", "SVG", "PDF", "HTML", "JSON"]
            )
            
            if st.button("Export Plot"):
                fig = st.session_state.current_plot
                
                if export_format == "HTML":
                    html_string = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="Download HTML",
                        data=html_string,
                        file_name=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                elif export_format == "JSON":
                    json_string = fig.to_json()
                    st.download_button(
                        label="Download JSON",
                        data=json_string,
                        file_name=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    # For image formats
                    img_bytes = fig.to_image(format=export_format.lower())
                    st.download_button(
                        label=f"Download {export_format}",
                        data=img_bytes,
                        file_name=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                        mime=f"image/{export_format.lower()}"
                    )
        else:
            st.info("Create a plot first to export")
    
    with col2:
        st.markdown("#### 📥 Import Options")
        
        uploaded_plot = st.file_uploader(
            "Upload Plot JSON",
            type=['json'],
            help="Upload a previously exported plot"
        )
        
        if uploaded_plot is not None:
            try:
                plot_data = json.load(uploaded_plot)
                fig = go.Figure(plot_data)
                st.session_state.current_plot = fig
                st.success("✅ Plot imported successfully!")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error importing plot: {e}")

with tab5:
    st.markdown("### 📋 Statistics & Analysis")
    
    # Data analysis section
    if 'df' in locals() and df is not None:
        st.markdown("#### 📊 Statistical Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Basic Statistics**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**Data Info**")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col3:
            st.markdown("**Missing Values**")
            missing_data = df.isnull().sum()
            st.dataframe(missing_data.to_frame(name='Missing Count'), use_container_width=True)
        
        # Correlation analysis
        if len(numeric_cols) >= 2:
            st.markdown("#### 🔗 Correlation Analysis")
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribution analysis
        st.markdown("#### 📈 Distribution Analysis")
        selected_col = st.selectbox("Select column for distribution analysis", numeric_cols)
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    df, 
                    x=selected_col, 
                    nbins=30,
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    df, 
                    y=selected_col,
                    title=f"Box Plot of {selected_col}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Outlier detection
        st.markdown("#### 🎯 Outlier Detection")
        if len(numeric_cols) >= 2:
            outlier_col = st.selectbox("Select column for outlier analysis", numeric_cols)
            
            if outlier_col:
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Points", len(df))
                with col2:
                    st.metric("Outliers", len(outliers))
                with col3:
                    st.metric("Outlier %", f"{len(outliers)/len(df)*100:.2f}%")
                
                if len(outliers) > 0:
                    st.dataframe(outliers, use_container_width=True)
    
    else:
        st.info("Load some data to see statistical analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Advanced Graph Visualizer Pro - Powered by Streamlit & Plotly</p>
    <p>Features: Mathematical Equations • Data Visualization • Statistical Analysis • Export/Import • Custom Styling</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import subprocess
import json
import os
import time
from datetime import datetime
import pandas as pd
from pathlib import Path

# Try to import optional dependencies
try:
    import yaml
except ImportError:
    st.error("PyYAML not installed. Run: pip install pyyaml")
    st.stop()

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="GuideLLM Benchmark Workbench",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 GuideLLM Benchmark Workbench")
st.markdown("*A user-friendly interface for running GuideLLM benchmarks*")

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'current_output' not in st.session_state:
    st.session_state.current_output = ""
if 'benchmark_running' not in st.session_state:
    st.session_state.benchmark_running = False

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Basic Parameters
    st.subheader("Endpoint Configuration")
    
    target = st.text_input(
        "Target Endpoint", 
        value="http://localhost:8000/v1",
        placeholder="http://your-endpoint:8000/v1",
        help="Specifies the target path for the backend server to run benchmarks against. Format: http://your-server:port/v1. Required parameter to define the server endpoint."
    )
    
    model_name = st.text_input(
        "Model Name",
        value="llama-3-2-3b",
        placeholder="Enter model identifier",
        help="Allows selecting a specific model from the server. If not provided, defaults to the first model available. Useful when multiple models are hosted on the same endpoint."
    )
    
    # Authentication
    st.subheader("Authentication")
    
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Your API key (optional)",
        help="Authentication key for accessing the target endpoint. Optional for secured endpoints. Set as GUIDELLM__OPENAI__API_KEY environment variable. Required for OpenAI API, Azure OpenAI, and other secured services."
    )
    
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password", 
        placeholder="For gated models (optional)",
        help="Authentication token for accessing gated Hugging Face models. Required only when using gated/private models from Hugging Face. Visit HuggingFace Settings to create a token."
    )
    
    # Advanced Parameters
    st.subheader("Benchmark Parameters")
    
    rate_type = st.selectbox(
        "Rate Type",
        ["synchronous", "throughput", "concurrent", "constant", "poisson", "sweep"],
        index=0,
        help="Defines the type of benchmark to run. synchronous: Single stream one at a time. throughput: All requests in parallel. concurrent: Fixed parallel streams. constant: Requests at constant rate. poisson: Poisson distribution. sweep: Auto-determines min/max rates."
    )
    
    # Rate parameter (required for some rate types)
    if rate_type in ["constant", "poisson"]:
        rate = st.number_input(
            "Rate (requests/second)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )
    elif rate_type == "sweep":
        rate_start = st.number_input(
            "Rate Start (req/s)",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1
        )
        rate_end = st.number_input(
            "Rate End (req/s)", 
            min_value=0.1,
            max_value=1000.0,
            value=50.0,
            step=0.1
        )
        rate_step = st.number_input(
            "Rate Step (req/s)",
            min_value=0.1,
            max_value=10.0,
            value=5.0,
            step=0.1
        )
        rate = f"{rate_start}:{rate_end}:{rate_step}"
    else:
        rate = None
    
    max_seconds = st.number_input(
        "Max Duration (seconds)",
        min_value=10,
        max_value=3600,
        value=60,
        step=10,
        help="Sets the maximum duration for each benchmark run. Benchmark stops when either duration OR request limit is reached. Typical values: 30-300s for quick tests, 600+ for production validation."
    )
    
    max_requests = st.number_input(
        "Max Requests",
        min_value=1,
        max_value=10000,
        value=100,
        step=10,
        help="Sets the maximum number of requests for each benchmark run. If not provided, runs until max-seconds is reached or dataset exhausted. Useful for consistent test sizes. Typical: 100-1000 for quick tests, 5000+ for thorough benchmarks."
    )
    
    max_concurrency = st.number_input(
        "Max Concurrency",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Global concurrency setting that limits parallel request execution. Used in throughput benchmarks to measure maximum server capacity. Higher values increase load but may hit server limits. Typical: 1-10 for single-user simulation, 20-100 for load testing."
    )
    
    # Processor Configuration  
    st.subheader("Processor/Tokenizer")
    
    processor_type = st.selectbox(
        "Processor Type", 
        ["Custom processor", "Use model default"], 
        index=0,
        help="Determines how tokenization is handled for synthetic data creation. Custom processor: Specify a HuggingFace model ID or local path. Use model default: Uses a lightweight default processor (gpt2)."
    )
    
    if processor_type == "Custom processor":
        processor = st.text_input(
            "Processor Path",
            value="meta-llama/Llama-3.2-3B",
            placeholder="e.g., meta-llama/Llama-3.2-3B, gpt2, microsoft/DialoGPT-medium",
            help="HuggingFace model ID or local path to processor/tokenizer. Must match the model's processor/tokenizer for accuracy. Used for synthetic data creation and local token metrics. Supports both HuggingFace IDs and local paths."
        )
    else:
        processor = "gpt2"  # Use gpt2 as default - it's small and widely available
    
    # Data Configuration
    st.subheader("Data Configuration")
    
    data_type = st.selectbox(
        "Data Type", 
        ["emulated", "simple", "custom"], 
        index=0,
        help="Specifies the dataset source for benchmark requests. emulated: Synthetic data with configurable token counts. simple: Basic prompts without tokenizer requirements. custom: HuggingFace dataset, local files, or JSON config."
    )
    
    
    if data_type == "simple":
        st.info("Simple mode uses basic prompts without tokenizer requirements")
        data_config = "prompt_tokens=512,output_tokens=128"
        
    if data_type == "emulated":
        st.info("💡 Emulated mode generates synthetic data using the specified processor/tokenizer")
        if processor and ("llama" in processor.lower() or "meta-llama" in processor.lower()):
            st.warning("🔒 **Gated Model Detected**: Llama models require a HuggingFace token. Make sure to provide your HF token above.")
        
        if processor_type == "Use model default" or not processor:
            st.warning("⚠️ **Emulated mode works best with a Custom Processor.** Consider specifying a HuggingFace model ID for better token accuracy.")
        
        
        prompt_tokens = st.number_input(
            "Prompt Tokens",
            min_value=1,
            max_value=4096,
            value=512,
            step=1,
            help="Average number of tokens for input prompts. Used for synthetic data generation. Determines request size and complexity. Affects latency and resource usage. Additional options: prompt_tokens_stdev, prompt_tokens_min/max for range limits."
        )
        
        output_tokens = st.number_input(
            "Output Tokens", 
            min_value=1,
            max_value=2048,
            value=128,
            step=1,
            help="Average number of tokens for generated outputs. Controls response length in synthetic data. Impacts generation time and throughput. Affects token/sec measurements. Additional options: output_tokens_stdev, output_tokens_min/max for range limits."
        )
        
        data_config = {
            "type": "emulated",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens
        }
    else:
        custom_data = st.text_area(
            "Custom Data Config (JSON)",
            value='{"type": "custom", "path": "/path/to/dataset"}',
            height=100
        )
        try:
            data_config = json.loads(custom_data)
        except:
            st.error("Invalid JSON format")
            data_config = "prompt_tokens=512,output_tokens=128"
    
    # Output Format Selection
    st.subheader("Output Format")
    
    output_format = st.selectbox(
        "Output Format",
        ["YAML", "HTML"],
        index=0,
        help="Choose output format. YAML: Machine-readable data format for processing. HTML: Visual report format for viewing in browser."
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Run Benchmark")
    
    # Display current configuration
    with st.expander("📋 Current Configuration", expanded=False):
        config_display = {
            "Target": target,
            "Model": model_name,
            "Processor": processor if processor else "Model default",
            "Rate Type": rate_type,
            "Rate": rate if rate else "Auto",
            "Max Duration": f"{max_seconds}s",
            "Max Requests": max_requests,
            "Max Concurrency": max_concurrency,
            "Data Config": data_config
        }
        st.json(config_display)
    
    # Run benchmark button
    if st.button("🚀 Run Benchmark", type="primary", disabled=st.session_state.benchmark_running):
        # Clear any previous live metrics
        if 'live_metrics' in st.session_state:
            del st.session_state.live_metrics
        if 'final_benchmark_results' in st.session_state:
            del st.session_state.final_benchmark_results
        
        # Validation checks
        validation_errors = []
        if not target or not model_name:
            validation_errors.append("Please provide both target endpoint and model name")
        
        # Check for gated models without tokens
        if processor and ("llama" in processor.lower() or "meta-llama" in processor.lower()) and not hf_token:
            validation_errors.append("Llama models require a HuggingFace token. Please provide your HF token in the Authentication section.")
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            st.session_state.benchmark_running = True
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./results/{model_name}_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set output file extension based on format selection
            file_extension = "yaml" if output_format == "YAML" else "html"
            output_file = output_dir / f"benchmark-results.{file_extension}"
            
            # Build command
            cmd = [
                "guidellm", "benchmark",
                "--target", target,
                "--model", model_name,
                "--rate-type", rate_type,
                "--max-seconds", str(max_seconds),
                "--max-requests", str(max_requests),
                "--data", data_config if isinstance(data_config, str) else json.dumps(data_config),
                "--output-path", str(output_file)
            ]
            
            # Add processor if specified
            if processor:
                cmd.extend(["--processor", processor])
            
            # Add rate if specified
            if rate:
                cmd.extend(["--rate", str(rate)])
            
            # Set environment variables
            env = os.environ.copy()
            if api_key:
                env["GUIDELLM__OPENAI__API_KEY"] = api_key
            if hf_token:
                env["HUGGING_FACE_HUB_TOKEN"] = hf_token
            env["GUIDELLM__MAX_CONCURRENCY"] = str(max_concurrency)
            
            # Display command being executed (properly quoted for shell)
            import shlex
            display_cmd = " ".join(shlex.quote(arg) for arg in cmd)
            st.code(display_cmd, language="bash")
            
            # Create placeholder for output
            output_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Execute command
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                    bufsize=1
                )
                
                output_lines = []
                benchmark_stats = []
                stats_started = False
                start_time = time.time()
                
                # Create placeholders for real-time display
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Real-time output display
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line_stripped = line.strip()
                        output_lines.append(line_stripped)
                        current_time = time.time()
                        elapsed = current_time - start_time
                        progress = min(elapsed / max_seconds, 1.0)
                        
                        progress_bar.progress(progress)
                        status_text.text(f"Running... ({elapsed:.0f}s elapsed)")
                        
                        # Parse real-time benchmark progress and store in session state
                        if "│" in line_stripped and ("req/s" in line_stripped or "Lat" in line_stripped):
                            # Extract live metrics from the progress box
                            try:
                                # Parse lines like: │ [00:47:39] ⠦ 100% synchronous (complete) Req: 0.3 req/s, 3.88s Lat, 1.0 Conc, 14 Comp, 1 Inc, 0 Err │
                                if "req/s" in line_stripped and "Lat" in line_stripped:
                                    import re
                                    
                                    # Extract metrics using regex
                                    req_match = re.search(r'Req:\s*([\d.]+)\s*req/s', line_stripped)
                                    lat_match = re.search(r'([\d.]+)s\s*Lat', line_stripped)
                                    conc_match = re.search(r'([\d.]+)\s*Conc', line_stripped)
                                    comp_match = re.search(r'(\d+)\s*Comp', line_stripped)
                                    
                                    tok_match = re.search(r'Tok:\s*([\d.]+)\s*gen/s,\s*([\d.]+)\s*tot/s', line_stripped)
                                    ttft_match = re.search(r'([\d.]+)ms\s*TTFT', line_stripped)
                                    
                                    if req_match and lat_match and tok_match and ttft_match:
                                        # Store live metrics in session state for sidebar display
                                        st.session_state.live_metrics = {
                                            "requests_per_sec": req_match.group(1),
                                            "latency": f"{lat_match.group(1)}s",
                                            "concurrency": conc_match.group(1) if conc_match else "1",
                                            "completed": comp_match.group(1) if comp_match else "0",
                                            "gen_tokens_per_sec": tok_match.group(1),
                                            "total_tokens_per_sec": tok_match.group(2),
                                            "ttft": f"{ttft_match.group(1)}ms"
                                        }
                            except:
                                pass  # If parsing fails, continue
                        
                        # Check for progress bar updates
                        elif "Generating..." in line_stripped and "━" in line_stripped:
                            # Show progress info
                            with progress_placeholder.container():
                                st.info(f"🔄 {line_stripped}")
                        
                        # Check for phase updates
                        elif any(phase in line_stripped for phase in ["Creating backend", "Creating request loader", "Created loader"]):
                            with status_placeholder.container():
                                st.info(f"📋 {line_stripped}")
                        
                        # Check for benchmark stats table - be more flexible with patterns
                        elif any(pattern in line_stripped for pattern in ["Benchmark Stats:", "===============", "Rate Type", "|synchronous|", "|constant|", "|poisson|"]):
                            stats_started = True
                        
                        if stats_started:
                            benchmark_stats.append(line_stripped)
                        
                        # Display last 10 lines of output (reduced to make room for live metrics)
                        recent_output = "\n".join(output_lines[-10:])
                        output_placeholder.text_area(
                            "Console Output",
                            value=recent_output,
                            height=200,
                            key=f"output_{len(output_lines)}"
                        )
                
                process.wait()
                
                if process.returncode == 0:
                    st.success("✅ Benchmark completed successfully!")
                    
                    # Store final results in session state for sidebar display
                    if benchmark_stats:
                        final_stats = "\n".join(benchmark_stats)
                        st.session_state.final_benchmark_results = final_stats
                    else:
                        # Try to parse from all output lines
                        all_output = "\n".join(output_lines)
                        st.session_state.final_benchmark_results = all_output
                    
                    # Load and display results
                    if output_file.exists():
                        # Load results based on format
                        if output_format == "YAML":
                            with open(output_file, 'r') as f:
                                try:
                                    results = yaml.safe_load(f)
                                except:
                                    results = None
                        else:  # HTML format
                            results = None  # HTML files don't contain parseable result data
                        
                        # Store in session state with benchmark stats
                        final_stats = "\n".join(benchmark_stats) if benchmark_stats else None
                        result_entry = {
                            "timestamp": timestamp,
                            "model": model_name,
                            "target": target,
                            "config": config_display,
                            "results": results,
                            "benchmark_stats": final_stats,
                            "output_file": str(output_file),
                            "output_format": output_format
                        }
                        st.session_state.results_history.append(result_entry)
                        
                        st.balloons()
                    else:
                        st.warning("Benchmark completed but no results file found")
                else:
                    st.error(f"❌ Benchmark failed with return code {process.returncode}")
                    
            except Exception as e:
                st.error(f"❌ Error running benchmark: {str(e)}")
            
            finally:
                st.session_state.benchmark_running = False
                # Only clear progress elements, keep results visible
                try:
                    progress_bar.empty()
                    status_text.empty()
                    # Live metrics now stored in session state for sidebar
                    # DON'T clear output_placeholder - it shows console history
                    progress_placeholder.empty() 
                    status_placeholder.empty()
                except:
                    pass

    # Show live metrics below the button (main area)
    if hasattr(st.session_state, 'live_metrics') and st.session_state.benchmark_running:
        st.subheader("🔥 Live Performance Metrics")
        metrics = st.session_state.live_metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚀 Requests/sec", metrics["requests_per_sec"])
        with col2:
            st.metric("⚡ Tokens/sec", metrics["gen_tokens_per_sec"])
        with col3:
            st.metric("⏱️ Latency", metrics["latency"])
        with col4:
            st.metric("🎯 TTFT", metrics["ttft"])
    
    # Show final results table in main area
    elif hasattr(st.session_state, 'final_benchmark_results'):
        st.success("🎉 Benchmark completed! Check the sidebar for key metrics.")
        
        # Show complete results table in main area
        final_stats = st.session_state.final_benchmark_results
        try:
            for line in final_stats.split('\n'):
                if "synchronous" in line or "constant" in line or "poisson" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 10:
                        st.subheader("📊 Complete Results")
                        results_data = {
                            "Metric": ["Rate Type", "Requests/Second", "Concurrency", "Output Tokens/sec", "Total Tokens/sec", 
                                      "Mean Latency (ms)", "Median Latency (ms)", "P99 Latency (ms)", 
                                      "Mean TTFT (ms)", "Median TTFT (ms)", "P99 TTFT (ms)"],
                            "Value": [parts[0], parts[1], parts[2], parts[3], parts[4], 
                                     parts[5], parts[6], parts[7], parts[8], parts[9], parts[10]]
                        }
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Add HTML download button for the results table
                        if st.session_state.results_history:
                            latest_result = st.session_state.results_history[-1]
                            if "output_file" in latest_result and Path(latest_result["output_file"]).exists():
                                
                                # Create HTML version of the table for download
                                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>GuideLLM Benchmark Results - {latest_result['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #0e1117; color: white; }}
        h1 {{ color: #fafafa; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #333; padding: 12px; text-align: left; }}
        th {{ background-color: #262730; }}
        td {{ background-color: #0e1117; }}
        .metric {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📊 GuideLLM Benchmark Results</h1>
    <p><strong>Timestamp:</strong> {latest_result['timestamp']}</p>
    <p><strong>Model:</strong> {latest_result['model']}</p>
    <p><strong>Target:</strong> {latest_result['target']}</p>
    
    <table>
        <tr><th>Metric</th><th>Value</th></tr>"""
                                
                                # Add the table rows
                                for i, (metric, value) in enumerate(zip(results_data["Metric"], results_data["Value"])):
                                    html_content += f"<tr><td class='metric'>{metric}</td><td>{value}</td></tr>"
                                
                                html_content += """
    </table>
</body>
</html>"""
                                
                                # Single HTML download button
                                st.download_button(
                                    label="📊 Download HTML Report", 
                                    data=html_content,
                                    file_name=f"benchmark-{latest_result['timestamp']}.html",
                                    mime="text/html"
                                )
                        break
        except:
            pass

with col2:
    st.header("📊 Quick Stats")
    
    # Show final results if just completed
    if hasattr(st.session_state, 'final_benchmark_results'):
        final_stats = st.session_state.final_benchmark_results
        
        # Parse and show key metrics
        parsed_successfully = False
        try:
            for line in final_stats.split('\n'):
                if "synchronous" in line or "constant" in line or "poisson" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 10:
                        st.subheader("✅ Final Results")
                        st.metric("🚀 Requests/sec", parts[1])
                        st.metric("⚡ Tokens/sec", parts[3])
                        st.metric("⏱️ Latency", f"{parts[5]} ms")
                        st.metric("🎯 TTFT", f"{parts[8]} ms")
                        parsed_successfully = True
                        break
        except Exception as e:
            pass  # Silent error handling
        
        # Fallback: show basic info if parsing failed
        if not parsed_successfully:
            st.subheader("✅ Benchmark Completed")
            st.info("Results available in history below")
            # Try to show some basic stats from YAML results if available
            if st.session_state.results_history:
                latest = st.session_state.results_history[-1]
                if "results" in latest and latest["results"]:
                    results = latest["results"]
                    if "summary" in results:
                        summary = results["summary"]
                        st.metric("🚀 Throughput", f"{summary.get('throughput', 'N/A')}")
                        st.metric("⏱️ Mean Latency", f"{summary.get('mean_latency', 'N/A')}")
                        parsed_successfully = True
    
    # Always show general stats and historical data when available
    elif st.session_state.results_history:
        total_runs = len(st.session_state.results_history)
        latest_run = st.session_state.results_history[-1]
        
        st.metric("Total Runs", total_runs)
        st.metric("Latest Model", latest_run["model"])
        st.metric("Latest Timestamp", latest_run["timestamp"])
        
        # Beautiful latest results display
        if "benchmark_stats" in latest_run and latest_run["benchmark_stats"]:
            st.subheader("🏆 Latest Performance")
            
            # Parse the latest benchmark stats for display
            try:
                stats_lines = latest_run["benchmark_stats"].split('\n')
                for line in stats_lines:
                    if "synchronous" in line or "constant" in line or "poisson" in line:
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 10:
                            st.metric("🚀 Requests/sec", parts[1])
                            st.metric("⚡ Tokens/sec", parts[3])
                            st.metric("⏱️ Latency", f"{parts[5]} ms")
                            st.metric("🎯 TTFT", f"{parts[8]} ms")
                            break
            except:
                pass
        
        # Fallback to old results format
        elif "results" in latest_run and latest_run["results"]:
            results = latest_run["results"]
            if "summary" in results:
                summary = results["summary"]
                if "throughput" in summary:
                    st.metric("Throughput (req/s)", f"{summary['throughput']:.2f}")
                if "mean_latency" in summary:
                    st.metric("Mean Latency (ms)", f"{summary['mean_latency']:.2f}")
    else:
        st.info("No benchmark runs yet")

# Results section
if st.session_state.results_history:
    st.header("📈 Results History")
    
    # Results table
    results_data = []
    for result in st.session_state.results_history:
        row = {
            "Timestamp": result["timestamp"],
            "Model": result["model"],
            "Target": result["target"][:50] + "..." if len(result["target"]) > 50 else result["target"],
            "Rate Type": result["config"]["Rate Type"],
            "Duration": result["config"]["Max Duration"],
            "Format": result.get("output_format", "YAML")
        }
        
        # Add performance metrics if available
        if "results" in result and result["results"] and "summary" in result["results"]:
            summary = result["results"]["summary"]
            row["Throughput"] = f"{summary.get('throughput', 0):.2f} req/s"
            row["Mean Latency"] = f"{summary.get('mean_latency', 0):.2f} ms"
        
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Detailed results viewer
    st.subheader("🔍 Detailed Results")
    
    selected_run = st.selectbox(
        "Select run to view details",
        options=range(len(st.session_state.results_history)),
        format_func=lambda x: f"{st.session_state.results_history[x]['timestamp']} - {st.session_state.results_history[x]['model']}"
    )
    
    if selected_run is not None:
        result = st.session_state.results_history[selected_run]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration")
            st.json(result["config"])
        
        with col2:
            st.subheader("Results")
            
            # Show benchmark stats if available
            if "benchmark_stats" in result and result["benchmark_stats"]:
                st.subheader("📊 Benchmark Stats")
                
                # Parse and display beautifully
                try:
                    stats_lines = result["benchmark_stats"].split('\n')
                    data_row = None
                    for line in stats_lines:
                        if "synchronous" in line or "constant" in line or "poisson" in line:
                            data_row = line
                            break
                    
                    if data_row:
                        parts = [p.strip() for p in data_row.split("|")]
                        if len(parts) >= 10:
                            # Key metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🚀 Requests/sec", parts[1])
                                st.metric("⏱️ Mean Latency", f"{parts[5]} ms")
                            with col2:
                                st.metric("⚡ Tokens/sec", parts[3])
                                st.metric("🎯 TTFT", f"{parts[8]} ms")
                            
                            # Full details in expander
                            with st.expander("📋 Full Benchmark Stats"):
                                st.code(result["benchmark_stats"], language="text")
                        else:
                            st.code(result["benchmark_stats"], language="text")
                    else:
                        st.code(result["benchmark_stats"], language="text")
                except:
                    st.code(result["benchmark_stats"], language="text")
            
            # Show detailed results
            if "results" in result and result["results"]:
                with st.expander("📋 Detailed YAML Results"):
                    st.json(result["results"])
            else:
                st.info("No detailed results available")
        
        # Download button for results
        if "output_file" in result and Path(result["output_file"]).exists():
            # Get format from result or default to YAML for backward compatibility
            format_type = result.get("output_format", "YAML")
            
            with open(result["output_file"], "r") as f:
                results_content = f.read()
            
            # Set appropriate label, file extension, and MIME type
            if format_type == "HTML":
                label = "📊 Download Results (HTML)"
                file_extension = "html"
                mime_type = "text/html"
            else:  # YAML
                label = "📥 Download Results (YAML)"
                file_extension = "yaml"
                mime_type = "text/yaml"
            
            st.download_button(
                label=label,
                data=results_content,
                file_name=f"benchmark-{result['timestamp']}.{file_extension}",
                mime=mime_type
            )

# Footer
st.markdown("---")
st.markdown("PoC App by <a href='http://red.ht/cai-team' target='_blank'>red.ht/cai-team</a>&nbsp;&nbsp;-&nbsp;&nbsp;*Built with ❤️ using Streamlit and GuideLLM*", unsafe_allow_html=True)
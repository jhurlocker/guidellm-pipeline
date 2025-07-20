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
        placeholder="http://your-endpoint:8000/v1"
    )
    
    model_name = st.text_input(
        "Model Name",
        value="llama-3.2-3b",
        placeholder="Enter model identifier"
    )
    
    # Authentication
    st.subheader("Authentication")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Your API key (optional)"
    )
    
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password", 
        placeholder="For gated models (optional)"
    )
    
    # Advanced Parameters
    st.subheader("Benchmark Parameters")
    
    rate_type = st.selectbox(
        "Rate Type",
        ["synchronous", "throughput", "concurrent", "constant", "poisson", "sweep"],
        index=0
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
        value=300,
        step=10
    )
    
    max_requests = st.number_input(
        "Max Requests",
        min_value=1,
        max_value=10000,
        value=100,
        step=10
    )
    
    max_concurrency = st.number_input(
        "Max Concurrency",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )
    
    # Processor Configuration  
    st.subheader("Processor/Tokenizer")
    
    processor_type = st.selectbox(
        "Processor Type", 
        ["Use model default", "Custom processor"], 
        index=0
    )
    
    if processor_type == "Custom processor":
        processor = st.text_input(
            "Processor Path",
            value="microsoft/DialoGPT-medium",
            placeholder="e.g., gpt2, microsoft/DialoGPT-medium"
        )
    else:
        processor = None
    
    # Data Configuration
    st.subheader("Data Configuration")
    
    data_type = st.selectbox("Data Type", ["simple", "emulated", "custom"], index=0)
    
    if data_type == "simple":
        st.info("Simple mode uses basic prompts without tokenizer requirements")
        data_config = "prompt_tokens=512,output_tokens=128"
        
    elif data_type == "emulated":
        st.error("⚠️ Emulated mode requires a valid processor/tokenizer and may cause authentication errors!")
        prompt_tokens = st.number_input(
            "Prompt Tokens",
            min_value=1,
            max_value=4096,
            value=512,
            step=1
        )
        
        output_tokens = st.number_input(
            "Output Tokens", 
            min_value=1,
            max_value=2048,
            value=128,
            step=1
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
        if not target or not model_name:
            st.error("Please provide both target endpoint and model name")
        else:
            st.session_state.benchmark_running = True
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./results/{model_name}_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "benchmark-results.yaml"
            
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
                start_time = time.time()
                
                # Real-time output display
                for line in iter(process.stdout.readline, ''):
                    if line:
                        output_lines.append(line.strip())
                        current_time = time.time()
                        elapsed = current_time - start_time
                        progress = min(elapsed / max_seconds, 1.0)
                        
                        progress_bar.progress(progress)
                        status_text.text(f"Running... ({elapsed:.0f}s elapsed)")
                        
                        # Display last 20 lines of output
                        recent_output = "\n".join(output_lines[-20:])
                        output_placeholder.text_area(
                            "Real-time Output",
                            value=recent_output,
                            height=400,
                            key=f"output_{len(output_lines)}"
                        )
                
                process.wait()
                
                if process.returncode == 0:
                    st.success("✅ Benchmark completed successfully!")
                    
                    # Load and display results
                    if output_file.exists():
                        with open(output_file, 'r') as f:
                            results = yaml.safe_load(f)
                        
                        # Store in session state
                        result_entry = {
                            "timestamp": timestamp,
                            "model": model_name,
                            "target": target,
                            "config": config_display,
                            "results": results,
                            "output_file": str(output_file)
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
                progress_bar.empty()
                status_text.empty()

with col2:
    st.header("📊 Quick Stats")
    
    if st.session_state.results_history:
        total_runs = len(st.session_state.results_history)
        latest_run = st.session_state.results_history[-1]
        
        st.metric("Total Runs", total_runs)
        st.metric("Latest Model", latest_run["model"])
        st.metric("Latest Timestamp", latest_run["timestamp"])
        
        # Quick results from latest run
        if "results" in latest_run and latest_run["results"]:
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
            "Duration": result["config"]["Max Duration"]
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
            if "results" in result and result["results"]:
                st.json(result["results"])
            else:
                st.info("No detailed results available")
        
        # Download button for results
        if "output_file" in result and Path(result["output_file"]).exists():
            with open(result["output_file"], "r") as f:
                results_yaml = f.read()
            
            st.download_button(
                label="📥 Download Results (YAML)",
                data=results_yaml,
                file_name=f"benchmark-{result['timestamp']}.yaml",
                mime="text/yaml"
            )

# Footer
st.markdown("---")
st.markdown("PoC App by <a href='http://red.ht/cai-team' target='_blank'>red.ht/cai-team</a>&nbsp;&nbsp;-&nbsp;&nbsp;*Built with ❤️ using Streamlit and GuideLLM*", unsafe_allow_html=True)
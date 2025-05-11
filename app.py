import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import os
import google.generativeai as genai
from typing import List, Dict, Set, Optional, Tuple, Union, Any
import schemdraw
from schemdraw import elements
from schemdraw import logic
import matplotlib
matplotlib.use('Agg')

# Configure schemdraw settings
schemdraw.theme('default')

# Must be the first Streamlit command
st.set_page_config(
    page_title=" AI Digital Circuit Designer",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for developer info and contact
if 'show_dev_info' not in st.session_state:
    st.session_state.show_dev_info = False
if 'contact_message' not in st.session_state:
    st.session_state.contact_message = ""

# Add custom CSS with new styles
st.markdown("""
<style>
    /* Modern gradient background for header */
    .stApp header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Custom styling for main container */
    .main {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Developer Info Section */
    .dev-info-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .dev-profile {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .dev-profile-pic {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        border: 4px solid white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        margin-right: 20px;
    }
    
    .dev-info-header {
        font-size: 2rem;
        color: white;
        margin-bottom: 5px;
        text-align: center;
    }
    
    .dev-role {
        font-size: 1.2rem;
        color: #a8c7ff;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .dev-bio {
        color: #e0e0e0;
        text-align: center;
        margin: 15px 0;
        line-height: 1.6;
        font-size: 1.1rem;
    }
    
    .dev-skills {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin: 20px 0;
    }
    
    .skill-tag {
        background: rgba(255,255,255,0.1);
        padding: 5px 15px;
        border-radius: 15px;
        color: #a8c7ff;
        font-size: 0.9rem;
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 25px 0;
    }
    
    .social-link {
        display: inline-flex;
        align-items: center;
        padding: 10px 20px;
        background: rgba(255,255,255,0.1);
        color: white;
        text-decoration: none;
        border-radius: 25px;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .social-link:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    .contact-form {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    
    .copyright {
        text-align: center;
        color: #a8c7ff;
        margin-top: 20px;
        font-size: 0.9rem;
    }
    
    /* Developer badge button */
    .stButton>button.dev-badge {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        padding: 10px 20px !important;
        border-radius: 20px !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
        z-index: 1000;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button.dev-badge:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Enhance button styling */
    .stButton > button {
        background: linear-gradient(to right, #4776E6, #8E54E9);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Improve selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    
    /* Custom card styling */
    .custom-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* Enhance metric styling */
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Improve tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px;
        color: #4776E6;
        border: 1px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, #4776E6, #8E54E9) !important;
        color: white !important;
    }
    
    /* Enhance expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 5px;
    }
    
    /* Custom toggle switch */
    .stCheckbox > label > div[role="checkbox"] {
        background-color: #4776E6;
    }
    
    /* Improve slider styling */
    .stSlider > div > div > div > div {
        background-color: #4776E6;
    }
    
    /* Developer Info Section */
    .dev-info-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .dev-header {
        text-align: center;
        margin-bottom: 25px;
    }
    
    .dev-name {
        font-size: 2.5rem;
        color: white;
        margin-bottom: 5px;
        font-weight: bold;
    }
    
    .dev-role {
        font-size: 1.3rem;
        color: #a8c7ff;
        margin-bottom: 20px;
    }
    
    .dev-bio {
        color: #e0e0e0;
        text-align: center;
        margin: 20px 0;
        line-height: 1.8;
        font-size: 1.1rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .dev-skills {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 12px;
        margin: 25px 0;
        padding: 0 20px;
    }
    
    .skill-tag {
        background: rgba(255,255,255,0.1);
        padding: 8px 18px;
        border-radius: 20px;
        color: #a8c7ff;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .skill-tag:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 30px 0;
    }
    
    .social-link {
        display: inline-flex;
        align-items: center;
        padding: 12px 25px;
        background: rgba(255,255,255,0.1);
        color: white;
        text-decoration: none;
        border-radius: 25px;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .social-link:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    .contact-section {
        background: rgba(255,255,255,0.05);
        padding: 25px;
        border-radius: 15px;
        margin: 30px auto;
        max-width: 600px;
    }
    
    .contact-header {
        color: white;
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 15px;
    }
    
    .contact-text {
        color: #a8c7ff;
        text-align: center;
        margin-bottom: 20px;
        font-size: 1.1rem;
    }
    
    .copyright {
        text-align: center;
        color: #a8c7ff;
        margin-top: 30px;
        font-size: 1rem;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header with animation
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 30px;'>
    <h1 style='color: white; font-size: 48px; margin-bottom: 10px; font-weight: bold;'>
        🔌 AI Digital Circuit Designer
    </h1>
    <p style='color: #e0e0e0; font-size: 20px; margin-bottom: 20px;'>
        Design, Simulate, and Analyze Digital Circuits with AI
    </p>
</div>
""", unsafe_allow_html=True)

# Create containers for developer info
dev_badge_container = st.container()
dev_info_container = st.container()

# Add the developer badge button
with dev_badge_container:
    if st.button("👨‍💻  Meet Jasim – The Mind Behind the Code", key="dev_badge"):
        st.session_state.show_dev_info = not st.session_state.show_dev_info

# Show developer info if button is clicked
if st.session_state.show_dev_info:
    with dev_info_container:
        # Main container with gradient background
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 20px; text-align: center;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 5px;'>Muhammad Jasim</h1>
            <h2 style='font-size: 1.3rem; color: #a8c7ff; margin-bottom: 20px;'>AI/ML Engineer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Bio section
        st.markdown("### About Me")
        st.write("Passionate about combining artificial intelligence with digital circuit design. Specializing in developing innovative AI solutions and creating intelligent digital systems.")
        
        # Skills section using columns
        st.markdown("### Skills")
        skills = ["🤖 Artificial Intelligence", "🧠 Machine Learning", "⚡ AI Agent", 
                 "🔧 System Architecture", "💻 Python Development", "📊 Data Analysis"]
        
        cols = st.columns(3)
        for i, skill in enumerate(skills):
            with cols[i % 3]:
                st.markdown(f"""
                <div style='background: rgba(30, 60, 114, 0.3); padding: 10px; border-radius: 15px; text-align: center; margin: 5px;'>
                    {skill}
                </div>
                """, unsafe_allow_html=True)
        
        # Social Links
        st.markdown("### Connect With Me")
        social_cols = st.columns(3)
        
        with social_cols[0]:
            st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/muhammad-jasim-b21802287")
        with social_cols[1]:
            st.link_button("🌐 Portfolio", "https://jasimai.vercel.app")
        with social_cols[2]:
            st.link_button("📺 YouTube", "https://youtube.com/@aibyjasim?si=ZtjTM1QOnnIwHXMV")
        
        # Contact section
        st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h3 style='color: white; font-size: 1.4rem; margin-bottom: 15px;'>📬 Get in Touch</h3>
            <p style='color: #a8c7ff;'>Have a question or want to collaborate? Let's connect!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Contact form
        with st.expander("💌 Send me a message"):
            col1, col2 = st.columns(2)
            with col1:
                contact_name = st.text_input("Your Name")
            with col2:
                contact_email = st.text_input("Your Email")
            contact_message = st.text_area("Your Message")
            if st.button("Send Message", type="primary"):
                st.success("Thanks for reaching out! I'll get back to you soon.")
        
        # Copyright
        st.markdown("""
        <p style='text-align: center; color: #a8c7ff; margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);'>
            © 2024 AI by Jasim | AI & Digital Circuit Innovation
        </p>
        """, unsafe_allow_html=True)

# Welcome message with enhanced styling
st.markdown("""
<style>
    .welcome-header {
        text-align: center;
        padding: 2rem 0;
    }
    .welcome-header h1 {
        color: #1e3c72;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .welcome-header p {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .welcome-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .welcome-section h4 {
        color: #1e3c72;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .welcome-section ol {
        color: #666;
        line-height: 1.6;
        margin-left: 1.5rem;
    }
    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .example-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        color: #666;
    }
    .ready-prompt {
        text-align: center;
        margin-top: 2rem;
        color: #1e3c72;
        font-size: 1.1rem;
    }
</style>

<div class="welcome-header">
    <h1>👋 Welcome to the Digital Circuit Designer!</h1>
    <p>Design, simulate, and analyze digital circuits with the power of AI</p>
</div>

<div class="welcome-section">
    <h4>🚀 Quick Start Guide:</h4>
    <ol>
        <li>Type your circuit description or choose a pre-built option</li>
        <li>Click "Generate Circuit" to create your design</li>
        <li>Use the interactive tools to simulate and analyze</li>
    </ol>
</div>

<div class="welcome-section">
    <h4>💡 Example Prompts:</h4>
    <div class="example-grid">
        <div class="example-card">
            "Create a 4-bit adder with carry out"
        </div>
        <div class="example-card">
            "Design a D flip-flop with async reset"
        </div>
        <div class="example-card">
            "Build a 2-bit counter circuit"
        </div>
    </div>
</div>

<div class="ready-prompt">
    🎯 Ready to start? Type your circuit description below or choose from our pre-built options!
</div>
""", unsafe_allow_html=True)

import numpy as np
import networkx as nx
import json
import base64
from PIL import Image
import time
from dataclasses import dataclass, field
import random
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============== Configuration ==============
# Set your Google API Key in Streamlit secrets or as an environment variable
def get_api_key():
    """Get API key from secrets or environment variables."""
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        st.error("⚠️ No API key found. Please set GEMINI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
        return None
    return api_key

# Configure the Gemini API securely with retries
def configure_api():
    """Configure the Gemini API with the provided key."""
    try:
        # First try to get API key from secrets
        api_key = st.secrets.get("GEMINI_API_KEY")
        
        # If not in secrets, try environment variable
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        # If still no API key, show a friendly input box
        if not api_key:
            st.warning("⚠️ No API key found in secrets or environment variables.")
            api_key = st.text_input(
                "Please enter your Gemini API key:",
                type="password",
                help="Get your API key from Google AI Studio (https://makersuite.google.com/app/apikey)",
                key="api_key_input"
            )
            if api_key:
                # Save to session state for reuse
                st.session_state['GEMINI_API_KEY'] = api_key
            else:
                return False
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Test the configuration with minimal content
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Test")
            if response and response.text:
                st.success("✅ API configured successfully!")
                return True
        except Exception as e:
            if "API_KEY_INVALID" in str(e):
                st.error("❌ Invalid API key. Please check your key and try again.")
            else:
                st.error(f"❌ Error testing API: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"❌ Configuration error: {str(e)}")
        return False

# Initialize API configuration
API_CONFIGURED = configure_api()

def get_gemini_model():
    """Get the Gemini model with proper configuration and error handling."""
    try:
        # Check if API is configured
        if not API_CONFIGURED and not configure_api():
            st.warning("Please configure the API key to use AI features.")
            return None
            
        # Get API key from session state or reconfigure
        api_key = st.session_state.get('GEMINI_API_KEY') or st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            if not configure_api():
                return None
        
        # Configure generation parameters
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Create and configure model
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )
        model.safety_settings = safety_settings
        
        return model
        
    except Exception as e:
        st.error(f"Error getting Gemini model: {str(e)}")
        return None

# ============== Circuit Components ==============
@dataclass
class LogicGate:
    gate_type: str  # AND, OR, NOT, NAND, NOR, XOR, XNOR, BUFFER
    inputs: List[str] = field(default_factory=list)  # Names of input connections
    output: str = ""  # Name of output connection
    position: Tuple[float, float] = (0, 0)  # (x, y) position for visualization
    
    def evaluate(self, input_values: Dict[str, bool]) -> bool:
        """Evaluate the gate's output based on its inputs."""
        try:
            if self.gate_type == "AND":
                return all(input_values[inp] for inp in self.inputs)
            elif self.gate_type == "OR":
                return any(input_values[inp] for inp in self.inputs)
            elif self.gate_type == "NOT":
                return not input_values[self.inputs[0]]
            elif self.gate_type == "NAND":
                return not all(input_values[inp] for inp in self.inputs)
            elif self.gate_type == "NOR":
                return not any(input_values[inp] for inp in self.inputs)
            elif self.gate_type == "XOR":
                return sum(input_values[inp] for inp in self.inputs) % 2 == 1
            elif self.gate_type == "XNOR":
                return sum(input_values[inp] for inp in self.inputs) % 2 == 0
            elif self.gate_type == "BUFFER":
                return input_values[self.inputs[0]]
            else:
                raise ValueError(f"Unknown gate type: {self.gate_type}")
        except KeyError as e:
            st.error(f"Missing input value: {e}")
            return False

@dataclass
class Circuit:
    name: str
    gates: List[LogicGate] = field(default_factory=list)
    inputs: Set[str] = field(default_factory=set)  # Circuit input names
    outputs: Set[str] = field(default_factory=set)  # Circuit output names
    
    def add_gate(self, gate: LogicGate):
        self.gates.append(gate)
        if gate.output and gate.output not in self.inputs:
            self.outputs.add(gate.output)
        for inp in gate.inputs:
            if not any(g.output == inp for g in self.gates):
                self.inputs.add(inp)
    
    def simulate(self, input_values: Dict[str, bool]) -> Dict[str, bool]:
        """Simulate the circuit with the given input values."""
        all_values = input_values.copy()
        
        # Sort gates to ensure inputs are computed before they're needed
        dependency_order = self._sort_gates_by_dependency()
        
        for gate in dependency_order:
            all_values[gate.output] = gate.evaluate(all_values)
        
        # Return only output values
        return {key: all_values[key] for key in self.outputs}
    
    def _sort_gates_by_dependency(self) -> List[LogicGate]:
        """Sort gates so inputs are evaluated before they're needed as inputs."""
        # Create a graph to represent dependencies
        graph = nx.DiGraph()
        for i, gate in enumerate(self.gates):
            graph.add_node(i)
        
        # Add edges for dependencies
        for i, gate_i in enumerate(self.gates):
            for j, gate_j in enumerate(self.gates):
                if i != j and gate_i.output in gate_j.inputs:
                    graph.add_edge(i, j)
        
        # Topological sort
        try:
            sorted_indices = list(nx.topological_sort(graph))
            return [self.gates[i] for i in sorted_indices]
        except nx.NetworkXUnfeasible:
            st.error("Circuit contains cyclic dependencies!")
            return self.gates  # Fallback to original order
    
    def to_dict(self) -> Dict:
        """Convert circuit to dictionary for serialization."""
        return {
            "name": self.name,
            "gates": [
                {
                    "gate_type": gate.gate_type,
                    "inputs": gate.inputs,
                    "output": gate.output,
                    "position": gate.position
                }
                for gate in self.gates
            ],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Circuit':
        """Create circuit from dictionary."""
        circuit = cls(name=data["name"])
        circuit.inputs = set(data.get("inputs", []))
        circuit.outputs = set(data.get("outputs", []))
        for gate_data in data.get("gates", []):
            gate = LogicGate(
                gate_type=gate_data["gate_type"],
                inputs=gate_data["inputs"],
                output=gate_data["output"],
                position=tuple(gate_data["position"])
            )
            circuit.gates.append(gate)
        return circuit

def visualize_circuit(circuit: Circuit) -> plt.Figure:
    """Create a visualization of the circuit using NetworkX and Matplotlib."""
    G = nx.DiGraph()
    
    # Add input nodes
    for inp in circuit.inputs:
        G.add_node(inp, node_type="input")
    
    # Add gate nodes and connections
    for i, gate in enumerate(circuit.gates):
        gate_id = f"gate_{i}_{gate.gate_type}"
        G.add_node(gate_id, node_type="gate", gate_type=gate.gate_type)
        
        # Connect inputs to this gate
        for inp in gate.inputs:
            G.add_edge(inp, gate_id)
        
        # Connect this gate to its output
        if gate.output:
            G.add_edge(gate_id, gate.output)
    
    # Add output nodes that aren't already in the graph
    for out in circuit.outputs:
        if out not in G:
            G.add_node(out, node_type="output")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use hierarchical layout for clearer visualization
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with different colors by type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "")
        if node_type == "input":
            node_colors.append("skyblue")
        elif node_type == "output":
            node_colors.append("lightgreen")
        else:
            node_colors.append("salmon")  # gates
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, ax=ax)
    
    # Add labels
    node_labels = {}
    for node in G.nodes():
        if G.nodes[node].get("node_type") == "gate":
            gate_type = G.nodes[node].get("gate_type", "")
            node_labels[node] = gate_type
        else:
            node_labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax)
    
    plt.axis('off')
    plt.tight_layout()
    return fig

def visualize_enhanced_circuit(circuit: Circuit) -> plt.Figure:
    """Create an enhanced visualization of the circuit using NetworkX and Matplotlib."""
    try:
        # Create a graph representation
        G = nx.DiGraph()
        
        # Add input nodes
        for inp in circuit.inputs:
            G.add_node(inp, node_type="input")
        
        # Add gate nodes and connections
        for i, gate in enumerate(circuit.gates):
            gate_id = f"gate_{i}_{gate.gate_type}"
            G.add_node(gate_id, node_type="gate", gate_type=gate.gate_type)
            
            # Connect inputs to this gate
            for inp in gate.inputs:
                G.add_edge(inp, gate_id)
            
            # Connect this gate to its output
            if gate.output:
                G.add_edge(gate_id, gate.output)
        
        # Add output nodes that aren't already in the graph
        for out in circuit.outputs:
            if out not in G:
                G.add_node(out, node_type="output")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Try to use kamada_kawai_layout, fall back to spring_layout if scipy not available
        try:
            pos = nx.kamada_kawai_layout(G)
        except ImportError:
            pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes with different colors by type
        input_nodes = [node for node in G.nodes() if G.nodes[node].get("node_type") == "input"]
        output_nodes = [node for node in G.nodes() if G.nodes[node].get("node_type") == "output"]
        gate_nodes = [node for node in G.nodes() if G.nodes[node].get("node_type") == "gate"]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color="skyblue", 
                             node_size=700, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color="lightgreen", 
                             node_size=700, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=gate_nodes, node_color="salmon", 
                             node_size=900, ax=ax)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, width=2, ax=ax)
        
        # Add labels
        node_labels = {}
        for node in G.nodes():
            if G.nodes[node].get("node_type") == "gate":
                gate_type = G.nodes[node].get("gate_type", "")
                node_labels[node] = gate_type
            else:
                node_labels[node] = node
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, 
                              font_weight="bold", ax=ax)
        
        plt.title(f"{circuit.name} Circuit Diagram", fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error visualizing circuit: {str(e)}")
        # Create a simple error visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error visualizing circuit:\n{str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

def draw_schematic_with_schemdraw(circuit: Circuit) -> BytesIO:
    """Create a professional schematic diagram using schemdraw."""
    try:
        # Create a drawing
        d = schemdraw.Drawing()
        
        # Set up spacing
        spacing = 2
        
        # Add title
        d += (elements.Dot().label(circuit.name))
        
        # Place inputs on the left
        input_signals = sorted(list(circuit.inputs))
        for i, inp_name in enumerate(input_signals):
            # Add input components
            d += (elements.Dot().label(inp_name))
            d += elements.LED2()
            d += elements.Line().length(spacing)
            
            # Move down for next input
            if i < len(input_signals) - 1:
                d += elements.Line().down().length(spacing)
        
        # Place gates
        for i, gate in enumerate(circuit.gates):
            # Add the appropriate gate
            gate_element = None
            if gate.gate_type == "AND":
                gate_element = logic.And()
            elif gate.gate_type == "OR":
                gate_element = logic.Or()
            elif gate.gate_type == "NOT":
                gate_element = logic.Not()
            elif gate.gate_type == "NAND":
                gate_element = logic.Nand()
            elif gate.gate_type == "NOR":
                gate_element = logic.Nor()
            elif gate.gate_type == "XOR":
                gate_element = logic.Xor()
            elif gate.gate_type == "XNOR":
                gate_element = logic.Xnor()
            elif gate.gate_type == "BUFFER":
                gate_element = logic.Buf()
            
            if gate_element:
                d += gate_element.label(gate.output)
                d += elements.Line().length(spacing)
            
            # Move down for next gate
            if i < len(circuit.gates) - 1:
                d += (elements.Line().down().length(spacing))
        
        # Move right for outputs
        d += (elements.Line().right().length(spacing))
        
        # Place outputs on the right
        output_signals = sorted(list(circuit.outputs))
        for i, out_name in enumerate(output_signals):
            with d.orient('right'):
                # Add connecting line
                d += elements.Line().length(spacing)
                
                # Add LED with color based on output state
                is_high = out_name in st.session_state.get('output_values', {}) and st.session_state.output_values.get(out_name, False)
                d += elements.LED2().color('green' if is_high else 'red')
                
                # Add output label
                d += (elements.Dot().label(out_name))
            
            # Move down for next output
            if i < len(output_signals) - 1:
                d += (elements.Line().down().length(spacing))
        
        # Save to buffer
        buf = BytesIO()
        d.save(buf, dpi=300)
        buf.seek(0)
        return buf
            
    except Exception as e:
        st.error(f"Error drawing schematic: {str(e)}")
        # Create a simple error image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        return buffer

def draw_professional_schematic(circuit: Circuit) -> BytesIO:
    """Draw a professional-looking circuit schematic with proper gates and connections."""
    try:
        # Create a drawing with a good size and white background
        d = schemdraw.Drawing(unit=2.0)
        
        # Calculate layout dimensions
        num_inputs = len(circuit.inputs)
        num_outputs = len(circuit.outputs)
        num_gates = len(circuit.gates)
        
        # Layout parameters
        input_spacing = 2.0
        gate_spacing = 3.0
        layer_spacing = 5.0
        
        # Create a graph to determine gate layers
        G = nx.DiGraph()
        for gate in circuit.gates:
            G.add_node(gate.output)
            for inp in gate.inputs:
                if inp in [g.output for g in circuit.gates]:
                    G.add_edge(inp, gate.output)
        
        # Determine gate layers using topological sort
        try:
            layers = {}
            for i, node in enumerate(nx.topological_sort(G)):
                gate = next((g for g in circuit.gates if g.output == node), None)
                if gate:
                    layer = max([layers.get(inp, -1) for inp in gate.inputs], default=-1) + 1
                    layers[gate.output] = layer
        except nx.NetworkXUnfeasible:
            layers = {gate.output: i for i, gate in enumerate(circuit.gates)}
        
        # Place inputs
        input_positions = {}
        input_signals = sorted(list(circuit.inputs))
        for i, inp_name in enumerate(input_signals):
            y_pos = (num_inputs - 1) / 2 - i
            
            # Add input label and dot
            d += (elements.Dot().label(inp_name, 'left').at((0, y_pos * input_spacing)))
            
            # Add connecting line
            d += (elements.Line().right().length(1).at((0, y_pos * input_spacing)))
            input_positions[inp_name] = (1, y_pos * input_spacing)
        
        # Place gates by layer
        gate_positions = {}
        max_layer = max(layers.values(), default=0)
        
        for layer in range(max_layer + 1):
            layer_gates = [g for g in circuit.gates if layers.get(g.output, 0) == layer]
            
            for i, gate in enumerate(layer_gates):
                # Calculate position
                x_pos = 3 + layer * layer_spacing
                y_pos = ((len(layer_gates) - 1) / 2 - i) * gate_spacing
                
                # Create gate element based on type
                gate_elem = None
                if gate.gate_type == "AND":
                    gate_elem = logic.And()
                elif gate.gate_type == "OR":
                    gate_elem = logic.Or()
                elif gate.gate_type == "NOT":
                    gate_elem = logic.Not()
                elif gate.gate_type == "NAND":
                    gate_elem = logic.Nand()
                elif gate.gate_type == "NOR":
                    gate_elem = logic.Nor()
                elif gate.gate_type == "XOR":
                    gate_elem = logic.Xor()
                elif gate.gate_type == "XNOR":
                    gate_elem = logic.Xnor()
                elif gate.gate_type == "BUFFER":
                    gate_elem = logic.Dot()
                
                # Position and add the gate
                d += gate_elem.at((x_pos, y_pos)).label(gate.output, 'right')
                
                # Store position for connections
                gate_positions[gate.output] = (x_pos + 1, y_pos)  # Output position
                
                # Connect inputs
                for inp in gate.inputs:
                    start_pos = input_positions.get(inp) or gate_positions.get(inp)
                    if start_pos:
                        d += (elements.Line()
                             .at(start_pos)
                             .to((x_pos - 0.5, y_pos)))
        
        # Place outputs
        output_signals = sorted(list(circuit.outputs))
        for i, out_name in enumerate(output_signals):
            if out_name in gate_positions:
                y_pos = (num_outputs - 1) / 2 - i
                x_pos = 3 + (max_layer + 1) * layer_spacing
                
                # Get source position
                start_pos = gate_positions[out_name]
                
                # Add connecting line and output
                d += (elements.Line()
                     .at(start_pos)
                     .to((x_pos, y_pos * input_spacing)))
                d += (elements.Dot()
                     .at((x_pos, y_pos * input_spacing))
                     .label(out_name, 'right'))
        
        # Save to buffer
        img_buffer = BytesIO()
        d.save(img_buffer, dpi=300)
        img_buffer.seek(0)
        return img_buffer
        
    except Exception as e:
        st.error(f"Error drawing schematic: {str(e)}")
        # Create a simple error image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        return buffer

def generate_circuit_from_prompt(prompt: str) -> Circuit:
    """Generate a circuit based on a natural language prompt using Gemini."""
    if not API_CONFIGURED:
        st.warning("API not configured. Using fallback circuit...")
        return fallback_circuit_selection(prompt)
        
    try:
        model = get_gemini_model()
        if not model:
            st.warning("Could not initialize model. Using fallback circuit...")
            return fallback_circuit_selection(prompt)
            
        # Enhanced system prompt for better circuit generation
        system_prompt = """You are a digital circuit design expert. Create a circuit based on this description.
        Requirements:
        1. Circuit name should be clear and descriptive
        2. Use standard logic gates: AND, OR, NOT, NAND, NOR, XOR, XNOR
        3. All gates must have proper inputs and outputs
        4. Connections must be logically valid
        5. No floating inputs or outputs allowed
        
        Response format (JSON only):
        {
            "circuit_name": "Name of Circuit",
            "gates": [
                {
                    "gate_type": "AND",
                    "inputs": ["A", "B"],
                    "output": "X"
                }
            ]
        }"""
        
        combined_prompt = f"{system_prompt}\n\nUser request: {prompt}\n\nRespond with valid JSON only."
        
        # Generate with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(combined_prompt)
                if not response or not response.text:
                    if attempt == max_retries - 1:
                        st.warning("No response from model. Using fallback circuit...")
                        return fallback_circuit_selection(prompt)
                    continue
                    
                content = response.text.strip()
                
                # Clean up the response to extract JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                # Parse and validate JSON
                circuit_data = json.loads(content)
                
                # Validate required fields
                if not circuit_data.get("circuit_name") or not circuit_data.get("gates"):
                    raise ValueError("Missing required fields in circuit data")
                
                # Create circuit
                circuit = Circuit(name=circuit_data["circuit_name"])
                
                # Add gates with improved positioning
                for i, gate_data in enumerate(circuit_data["gates"]):
                    # Validate gate data
                    if not all(k in gate_data for k in ["gate_type", "inputs", "output"]):
                        raise ValueError(f"Invalid gate data: {gate_data}")
                    
                    # Calculate better positions for clearer visualization
                    x_pos = 0.2 + (i % 3) * 0.3  # Spread gates horizontally
                    y_pos = 0.2 + (i // 3) * 0.3  # Stack vertically when more than 3 gates
                    
                    gate = LogicGate(
                        gate_type=gate_data["gate_type"],
                        inputs=gate_data["inputs"],
                        output=gate_data["output"],
                        position=(x_pos, y_pos)
                    )
                    circuit.add_gate(gate)
                
                # Validate the circuit
                if not circuit.gates:
                    raise ValueError("No gates created")
                
                st.success("Circuit generated successfully!")
                return circuit
                
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to parse circuit description: {str(e)}")
                    return fallback_circuit_selection(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Error in circuit generation: {str(e)}")
                    return fallback_circuit_selection(prompt)
                
    except Exception as e:
        st.error(f"Unexpected error in circuit generation: {str(e)}")
        return fallback_circuit_selection(prompt)

def fallback_circuit_selection(prompt: str) -> Circuit:
    """Select an appropriate predefined circuit based on the prompt."""
    prompt_lower = prompt.lower()
    
    if "4-bit" in prompt_lower and "adder" in prompt_lower:
        return generate_4_bit_adder()
    elif "full" in prompt_lower and "adder" in prompt_lower:
        return create_full_adder()
    elif "half" in prompt_lower and "adder" in prompt_lower:
        return create_half_adder()
    elif "flip" in prompt_lower or "flop" in prompt_lower:
        return create_d_flip_flop()
    elif "latch" in prompt_lower:
        return create_sr_latch()
    elif "counter" in prompt_lower:
        return create_2_bit_counter()
    elif "adder" in prompt_lower:
        return create_full_adder()
    else:
        # Default fallback
        return create_half_adder()

def get_enhanced_circuit_explanation(circuit: Circuit) -> Dict[str, str]:
    """Get an enhanced explanation of the circuit with multiple sections."""
    try:
        # Get basic explanation
        basic_explanation = get_circuit_explanation(circuit)
        
        # Parse it into sections for better UI
        return {
            "overview": "This circuit performs digital logic operations to process binary signals.",
            "operation": basic_explanation,
            "use_cases": "This type of circuit is commonly used in digital electronics, computers, and control systems.",
            "limitations": "The circuit operates on ideal logic levels and doesn't account for real-world issues like signal noise or propagation delay.",
            "performance": "The circuit uses standard digital logic gates with typical propagation delays in the nanosecond range."
        }
    except Exception as e:
        # Fallback if there's an error
        return {
            "overview": f"This is a {circuit.name} circuit with {len(circuit.gates)} gates.",
            "operation": "The circuit processes input signals through logic gates to produce output signals.",
            "use_cases": "Digital logic applications",
            "limitations": "Ideal digital logic model",
            "performance": "Standard digital performance characteristics"
        }

def analyze_circuit_timing(circuit: Circuit) -> Dict:
    """Analyze circuit timing characteristics."""
    # Mock implementation - in a real system this would do actual timing analysis
    timing_results = {
        "critical_path": {
            "delay": 4.2,
            "output": next(iter(circuit.outputs)) if circuit.outputs else "Output"
        },
        "max_frequency": 238.1,
        "setup_time": 1.2,
        "hold_time": 0.5,
        "propagation_delays": {}
    }
    
    # Add mock propagation delays for each gate
    gate_count = len(circuit.gates)
    for i, gate in enumerate(circuit.gates):
        if gate.output:
            # Generate slightly random delays for realism
            timing_results["propagation_delays"][gate.output] = 1.0 + (i % 3) * 0.5
    
    return timing_results

def create_timing_diagram(circuit: Circuit, input_sequences: Dict[str, List[bool]]) -> go.Figure:
    """Create an interactive timing diagram for the given circuit and input sequences."""
    # Calculate number of time steps
    if not input_sequences:
        return go.Figure()
    
    num_steps = max(len(seq) for seq in input_sequences.values())
    if num_steps == 0:
        return go.Figure()
    
    # Prepare figure
    fig = go.Figure()
    
    # First, plot input signals
    y_pos = 0
    time_points = list(range(num_steps))
    
    # Generate outputs for each time step
    output_sequences = {out: [False] * num_steps for out in circuit.outputs}
    for t in range(num_steps):
        # Get input values for this time step
        inputs = {}
        for inp, seq in input_sequences.items():
            if t < len(seq):
                inputs[inp] = seq[t]
            else:
                inputs[inp] = False
        
        # Simulate circuit for this time step
        if all(inp in inputs for inp in circuit.inputs):
            outputs = circuit.simulate(inputs)
            
            # Record outputs
            for out, val in outputs.items():
                output_sequences[out][t] = val
    
    # Plot input signals
    for name, values in input_sequences.items():
        y_values = [y_pos + int(val) for val in values[:num_steps]]
        fig.add_trace(go.Scatter(
            x=time_points,
            y=y_values,
            mode='lines',
            name=name,
            line=dict(shape='hv', width=2)
        ))
        y_pos += 2
    
    # Plot output signals
    for name, values in output_sequences.items():
        y_values = [y_pos + int(val) for val in values]
        fig.add_trace(go.Scatter(
            x=time_points, 
            y=y_values,
            mode='lines',
            name=name,
            line=dict(shape='hv', width=2, dash='dot')
        ))
        y_pos += 2
    
    # Update layout
    fig.update_layout(
        title="Circuit Timing Diagram",
        xaxis_title="Time Step",
        yaxis_visible=False,
        height=100 + 50 * (len(input_sequences) + len(circuit.outputs)),
        showlegend=True
    )
    
    return fig

def generate_truth_table(circuit: Circuit) -> pd.DataFrame:
    """Generate a complete truth table for the circuit."""
    inputs = sorted(list(circuit.inputs))
    outputs = sorted(list(circuit.outputs))
    
    # Prepare data structure for truth table
    num_rows = 2 ** len(inputs)
    data = {inp: [False] * num_rows for inp in inputs}
    data.update({out: [False] * num_rows for out in outputs})
    
    # Generate all possible input combinations
    for row in range(num_rows):
        # Convert row number to binary and use as input values
        for i, inp in enumerate(inputs):
            # Check if the i-th bit is set in the row number
            data[inp][row] = (row & (1 << (len(inputs) - i - 1))) != 0
    
    # Simulate circuit for each row
    for row in range(num_rows):
        input_values = {inp: data[inp][row] for inp in inputs}
        output_values = circuit.simulate(input_values)
        
        # Record output values
        for out in outputs:
            data[out][row] = output_values.get(out, False)
    
    # Convert boolean values to 0/1 for better readability
    for col in data:
        data[col] = [1 if val else 0 for val in data[col]]
    
    return pd.DataFrame(data)

@dataclass
class SimulationResult:
    """Store the result of a circuit simulation."""
    timestamp: float
    circuit_name: str
    inputs: Dict[str, bool]
    outputs: Dict[str, bool]

def update_settings(tool_type, setting_name):
    """Update settings for oscilloscope, logic analyzer, or multimeter."""
    if tool_type == "oscilloscope":
        if setting_name == "timebase" and "osc_timebase" in st.session_state:
            st.session_state.oscilloscope_settings["timebase"] = st.session_state.osc_timebase
        elif setting_name == "voltage_scale" and "osc_voltage" in st.session_state:
            st.session_state.oscilloscope_settings["voltage_scale"] = st.session_state.osc_voltage
        elif setting_name == "trigger_mode" and "osc_trigger" in st.session_state:
            st.session_state.oscilloscope_settings["trigger_mode"] = st.session_state.osc_trigger
        elif setting_name == "trigger_level" and "main_osc_trigger_level" in st.session_state:
            st.session_state.oscilloscope_settings["trigger_level"] = st.session_state.main_osc_trigger_level
    
    elif tool_type == "logic_analyzer":
        if setting_name == "sample_rate" and "la_sample_rate" in st.session_state:
            st.session_state.logic_analyzer_settings["sample_rate"] = st.session_state.la_sample_rate
        elif setting_name == "trigger_channel" and "main_la_trigger_channel" in st.session_state:
            st.session_state.logic_analyzer_settings["trigger_channel"] = st.session_state.main_la_trigger_channel
        elif setting_name == "trigger_condition" and "sidebar_la_trigger_condition" in st.session_state:
            st.session_state.logic_analyzer_settings["trigger_condition"] = st.session_state.sidebar_la_trigger_condition
        elif setting_name == "buffer_size" and "la_buffer_size" in st.session_state:
            st.session_state.logic_analyzer_settings["buffer_size"] = st.session_state.la_buffer_size
    
    elif tool_type == "multimeter":
        if setting_name == "mode" and "mm_mode" in st.session_state:
            st.session_state.multimeter_settings["mode"] = st.session_state.mm_mode
        elif setting_name == "range" and "mm_range" in st.session_state:
            st.session_state.multimeter_settings["range"] = st.session_state.mm_range

def init_session_state():
    """Initialize session state variables if they don't exist."""
    # Basic session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_circuit' not in st.session_state:
        st.session_state.current_circuit = None
    if 'saved_circuits' not in st.session_state:
        st.session_state.saved_circuits = []
    if 'input_values' not in st.session_state:
        st.session_state.input_values = {}
    if 'simulation_history' not in st.session_state:
        st.session_state.simulation_history = []
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Design"
    
    # Analysis tool states
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {
            'timing': {},
            'power': {},
            'reliability': {},
            'signals': {}
        }
    
    # Oscilloscope settings
    if 'oscilloscope_settings' not in st.session_state:
        st.session_state.oscilloscope_settings = {
            'timebase': '100 ns/div',
            'voltage_scale': '2 V/div',
            'trigger_mode': 'Auto',
            'trigger_level': 2.5,
            'trigger_source': 'None',
            'running': False,
            'data': {},
            'cursors': {'x1': 0, 'x2': 100, 'y1': 0, 'y2': 5},
            'measurements': {}
        }
    
    # Logic Analyzer states
    if 'logic_analyzer_settings' not in st.session_state:
        st.session_state.logic_analyzer_settings = {
            'sample_rate': 100,  # MHz
            'trigger_channel': '',
            'trigger_condition': 'Rising Edge',
            'buffer_size': 1000,
            'channels': [],
            'running': False,
            'data': {},
            'trigger_level': 2.5,
            'time_scale': 100  # ns/div
        }
    
    # Multimeter states
    if 'multimeter_settings' not in st.session_state:
        st.session_state.multimeter_settings = {
            'mode': 'Voltage',
            'range': 'Auto',
            'probe_point': '',
            'reference': 'GND',
            'running': False,
            'measurements': {},
            'history': [],
            'auto_range': True,
            'sampling_rate': 1000
        }
    
    # AI Assistant states
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = {
            'messages': [],
            'context': {},
            'analysis_history': [],
            'suggestions': [],
            'last_circuit': None
        }
    
    # Signal Analysis states
    if 'signal_analysis' not in st.session_state:
        st.session_state.signal_analysis = {
            'captured_signals': {},
            'measurements': {},
            'triggers': {},
            'markers': {},
            'cursors': {'x1': 0, 'x2': 100, 'y1': 0, 'y2': 5}
        }

def render_ai_assistant_tab():
    st.markdown("""
    <div class='custom-card' style='margin-bottom: 20px;'>
        <h2 style='color: #1e3c72; margin-bottom: 20px;'>💬 AI Circuit Analysis Assistant</h2>
        <p style='color: #666; font-size: 16px; margin-bottom: 15px;'>
            Get instant help with circuit analysis, troubleshooting, and learning concepts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Create two columns for a better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #1e3c72; margin-bottom: 15px;'>🤔 Ask Your Question</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced text input with placeholder
        user_input = st.text_area(
            "",
            placeholder="Example questions:\n• How does this circuit work?\n• What are the key components?\n• Explain the logic behind this design\n• What are common applications?",
            key="ai_assistant_input",
            height=100
        )
        
        # Quick question buttons with improved styling
        st.markdown("""
        <div style='margin: 15px 0;'>
            <p style='color: #1e3c72; font-weight: bold; margin-bottom: 10px;'>⚡ Quick Questions</p>
        </div>
        """, unsafe_allow_html=True)
        
        quick_q_cols = st.columns(3)
        with quick_q_cols[0]:
            if st.button("❓ How it works", use_container_width=True):
                user_input = "Can you explain how this circuit works in detail?"
        with quick_q_cols[1]:
            if st.button("🔍 Analyze state", use_container_width=True):
                user_input = "Analyze the current state of the circuit and explain what's happening."
        with quick_q_cols[2]:
            if st.button("📊 Truth table", use_container_width=True):
                user_input = "Generate and explain the truth table for this circuit."
    
    with col2:
        # Context information about current circuit with improved styling
        if st.session_state.current_circuit:
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 15px;'>🔍 Current Circuit</h3>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style='color: #666; margin-bottom: 10px;'><strong>Name:</strong> {st.session_state.current_circuit.name}</p>
                <p style='color: #666; margin-bottom: 10px;'><strong>Gates:</strong> {len(st.session_state.current_circuit.gates)}</p>
                <p style='color: #666; margin-bottom: 10px;'><strong>Inputs:</strong> {', '.join(st.session_state.current_circuit.inputs)}</p>
                <p style='color: #666;'><strong>Outputs:</strong> {', '.join(st.session_state.current_circuit.outputs)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Send button with improved styling
    if st.button("🚀 Send Question", key="send_question", type="primary", use_container_width=True):
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("🤔 Analyzing your question..."):
                try:
                    # Try to get AI response
                    response = get_ai_response(user_input, st.session_state.current_circuit)
                    
                    # If API is not available, use pre-defined responses
                    if "API key" in response or "error" in response.lower():
                        # Fallback to pre-defined responses
                        if "how" in user_input.lower() and "work" in user_input.lower():
                            response = get_circuit_explanation(st.session_state.current_circuit)
                        elif "truth table" in user_input.lower():
                            response = "Here's the truth table analysis:\n\n" + str(generate_truth_table(st.session_state.current_circuit))
                        elif "state" in user_input.lower() or "current" in user_input.lower():
                            response = f"Current circuit state:\nInputs: {st.session_state.input_values}\nOutputs: {st.session_state.current_circuit.simulate(st.session_state.input_values)}"
                        else:
                            response = get_circuit_explanation(st.session_state.current_circuit)
                    
                    # Add response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"An error occurred, but I'll try to help anyway! {str(e)}")
                    # Provide basic circuit information as fallback
                    response = get_circuit_explanation(st.session_state.current_circuit)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear input
            st.text_area("", value="", key="ai_assistant_input_clear")
    
    # Display chat history with improved styling
    if st.session_state.messages:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #1e3c72; margin-bottom: 20px;'>💭 Conversation History</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown("""
                <div style='background: #f0f7ff; padding: 15px; border-radius: 15px; margin: 10px 0; border-bottom-right-radius: 5px;'>
                    <p style='color: #1e3c72; margin-bottom: 5px; font-weight: bold;'>You</p>
                    <p style='color: #666; margin: 0;'>{}</p>
                </div>
                """.format(msg['content']), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 15px; margin: 10px 0; border-bottom-left-radius: 5px;'>
                    <p style='color: #1e3c72; margin-bottom: 5px; font-weight: bold;'>Assistant</p>
                    <p style='color: #666; margin: 0;'>{}</p>
                </div>
                """.format(msg['content']), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 30px; background: #f8f9fa; border-radius: 10px; margin: 20px 0;'>
            <img src='https://img.icons8.com/fluency/96/000000/chat.png' style='width: 64px; margin-bottom: 20px;'/>
            <h3 style='color: #1e3c72; margin-bottom: 10px;'>No messages yet</h3>
            <p style='color: #666;'>Start by asking a question about the circuit!</p>
        </div>
        """, unsafe_allow_html=True)

def get_ai_response(question: str, circuit: Circuit = None) -> str:
    """Generate an AI response with improved error handling and fallbacks."""
    try:
        model = get_gemini_model()
        if not model:
            return "⚠️ AI features are currently unavailable. Please configure your API key to use this feature."
        
        # Prepare context about the current circuit if available
        circuit_context = ""
        if circuit:
            circuit_context = f"""
            The user is currently working with a {circuit.name} circuit. 
            This circuit has {len(circuit.gates)} gates, {len(circuit.inputs)} inputs, and {len(circuit.outputs)} outputs.
            Gates in the circuit: {[f"{g.gate_type} ({g.output})" for g in circuit.gates]}
            Inputs: {list(circuit.inputs)}
            Outputs: {list(circuit.outputs)}
            """
        
        # Construct the prompt
        prompt = f"""
        You are a digital electronics expert assistant. Please provide a helpful, educational response to this question:
        
        {question}
        
        {circuit_context}
        
        Provide a thorough explanation with examples where appropriate. If discussing a circuit, include practical applications.
        """
        
        # Generate the response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
                if attempt == max_retries - 1:
                    return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
            except Exception as e:
                if "API_KEY_INVALID" in str(e):
                    return "⚠️ Invalid API key. Please check your API key configuration."
                if attempt == max_retries - 1:
                    return f"I apologize, but an error occurred: {str(e)}"
                time.sleep(1)  # Wait before retry
        
        return "I apologize, but I couldn't generate a response at this time. Please try again later."
        
    except Exception as e:
        return f"I apologize, but an error occurred: {str(e)}"

def main():
    init_session_state()
    
    # Add a clearer header section with simple instructions
    st.title("💻 AI Digital Circuit Designer")
    st.markdown("""
    <div style='background-color:rgba(29, 32, 39, 0.22); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3>Welcome to the Digital Circuit Designer!</h3>
        <p>📝 <b>Get started:</b> Type a circuit description and click "Generate Circuit" or choose from the pre-built options.</p>
        <p>⚡ <b>Example prompts:</b> "Create a 4-bit adder","why look bro make future hhahahah", "Design a D flip-flop bro ", "Make a half adder circuit"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add main tabs for different sections of the app
    main_tabs = st.tabs([
        "🎨 Design", 
        "📊 Properties", 
        "▶️ Simulation", 
        "📈 Analysis", 
        "🔍 Oscilloscope", 
        "📱 Logic Analyzer", 
        "⚡ Multimeter", 
        "🤖 AI Assistant"
    ])
    
    # Design Tab
    with main_tabs[0]:
        # Create a modern two-column layout with cards
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>🎨 Create Your Circuit</h3>
            """, unsafe_allow_html=True)
            
            # Enhanced text input
            prompt = st.text_area(
                "Describe your circuit:",
                height=100,
                placeholder="Example: Create a 4-bit adder with carry in and carry out",
                help="Be as specific as possible in your description"
            )
            
            # Quick-select section with modern styling
            st.markdown("""
            <div style='margin: 20px 0;'>
                <h4 style='color: #1e3c72; margin-bottom: 15px;'>🚀 Quick Select</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # First row of quick select buttons
            quick_select_cols = st.columns(3)
            with quick_select_cols[0]:
                if st.button("Half Adder", use_container_width=True):
                    st.session_state.current_circuit = create_half_adder()
                    st.rerun()
            with quick_select_cols[1]:
                if st.button("Full Adder", use_container_width=True):
                    st.session_state.current_circuit = create_full_adder()
                    st.rerun()
            with quick_select_cols[2]:
                if st.button("4-Bit Adder", use_container_width=True):
                    st.session_state.current_circuit = generate_4_bit_adder()
                    st.rerun()
            
            # Second row of quick select buttons
            quick_select_cols2 = st.columns(3)
            with quick_select_cols2[0]:
                if st.button("SR Latch", use_container_width=True):
                    st.session_state.current_circuit = create_sr_latch()
                    st.rerun()
            with quick_select_cols2[1]:
                if st.button("D Flip-Flop", use_container_width=True):
                    st.session_state.current_circuit = create_d_flip_flop()
                    st.rerun()
            with quick_select_cols2[2]:
                if st.button("2-Bit Counter", use_container_width=True):
                    st.session_state.current_circuit = create_2_bit_counter()
                    st.rerun()
            
            # Generate button with enhanced styling
            st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
            if st.button("🔮 Generate Custom Circuit", type="primary", use_container_width=True):
                with st.spinner("✨ Creating your circuit... Please wait"):
                    try:
                        circuit = generate_circuit_from_prompt(prompt)
                        st.session_state.current_circuit = circuit
                        st.success(f"✅ {circuit.name} created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error creating circuit: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Display current circuit with enhanced styling
            if st.session_state.current_circuit:
                st.markdown(f"""
                <div class='custom-card'>
                    <h3 style='color: #1e3c72; margin-bottom: 20px;'>
                        📊 {st.session_state.current_circuit.name}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Show circuit visualization in a card
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                fig = visualize_enhanced_circuit(st.session_state.current_circuit)
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Circuit explanation in a modern expandable card
                with st.expander("📖 Circuit Explanation", expanded=True):
                    explanation = get_circuit_explanation(st.session_state.current_circuit)
                    st.markdown(f"""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <h4 style='color: #1e3c72; margin-bottom: 10px;'>How it works:</h4>
                        <p style='color: #666; font-size: 16px;'>{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='custom-card' style='text-align: center; padding: 40px;'>
                    <img src='https://img.icons8.com/fluency/96/000000/circuit.png' style='width: 96px; margin-bottom: 20px;'/>
                    <h3 style='color: #1e3c72; margin-bottom: 10px;'>No Circuit Created Yet</h3>
                    <p style='color: #666; font-size: 16px;'>
                        Please select a pre-built circuit or generate a custom one to begin.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Properties Tab
    with main_tabs[1]:
        if st.session_state.current_circuit:
            st.markdown(f"""
            <div class='custom-card'>
                <h2 style='color: #1e3c72; margin-bottom: 20px;'>
                    📊 Circuit Properties: {st.session_state.current_circuit.name}
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Circuit details in a modern card
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>🔍 Basic Properties</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a property table with modern styling
            properties = {
                "Circuit Name": st.session_state.current_circuit.name,
                "Number of Gates": str(len(st.session_state.current_circuit.gates)),
                "Number of Inputs": str(len(st.session_state.current_circuit.inputs)),
                "Number of Outputs": str(len(st.session_state.current_circuit.outputs)),
                "Gate Types Used": ", ".join(set(gate.gate_type for gate in st.session_state.current_circuit.gates))
            }
            
            # Display properties in a modern grid
            prop_cols = st.columns(3)
            for i, (prop, value) in enumerate(properties.items()):
                with prop_cols[i % 3]:
                    st.markdown(f"""
                    <div style='background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin: 10px 0;'>
                        <p style='color: #666; font-size: 14px; margin-bottom: 5px;'>{prop}</p>
                        <h4 style='color: #1e3c72; margin: 0;'>{value}</h4>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Input and Output details in modern cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class='custom-card'>
                    <h3 style='color: #1e3c72; margin-bottom: 20px;'>📥 Input Signals</h3>
                """, unsafe_allow_html=True)
                
                if st.session_state.current_circuit.inputs:
                    for inp in sorted(list(st.session_state.current_circuit.inputs)):
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            <span style='color: #1e3c72;'>🔹 {inp}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No inputs defined.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='custom-card'>
                    <h3 style='color: #1e3c72; margin-bottom: 20px;'>📤 Output Signals</h3>
                """, unsafe_allow_html=True)
                
                if st.session_state.current_circuit.outputs:
                    for out in sorted(list(st.session_state.current_circuit.outputs)):
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            <span style='color: #1e3c72;'>🔸 {out}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No outputs defined.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Gates details in a modern table
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>⚡ Gates in Circuit</h3>
            """, unsafe_allow_html=True)
            
            if st.session_state.current_circuit.gates:
                gates_data = []
                for i, gate in enumerate(st.session_state.current_circuit.gates):
                    gates_data.append({
                        "Gate #": f"Gate {i+1}",
                        "Type": gate.gate_type,
                        "Inputs": ", ".join(gate.inputs),
                        "Output": gate.output
                    })
                gates_df = pd.DataFrame(gates_data)
                
                # Style the dataframe
                st.markdown("""
                <style>
                    .dataframe {
                        border: none !important;
                        border-radius: 8px !important;
                        overflow: hidden !important;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
                    }
                    .dataframe th {
                        background-color: #1e3c72 !important;
                        color: white !important;
                        font-weight: 500 !important;
                        text-align: center !important;
                    }
                    .dataframe td {
                        text-align: center !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                st.dataframe(gates_df, use_container_width=True)
            else:
                st.info("No gates in circuit.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Circuit Schematic section with modern styling
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>📝 Circuit Schematic</h3>
            """, unsafe_allow_html=True)
            
            try:
                schematic_buffer = draw_professional_schematic(st.session_state.current_circuit)
                schematic_image = Image.open(schematic_buffer)
                st.image(schematic_image, caption=f"{st.session_state.current_circuit.name} Schematic", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate schematic: {str(e)}")
                st.image("https://via.placeholder.com/400x300?text=Schematic+Unavailable", 
                         caption="Schematic unavailable", use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Export options with modern styling
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>💾 Export Circuit</h3>
            """, unsafe_allow_html=True)
            
            export_cols = st.columns(2)
            with export_cols[0]:
                if st.button("📥 Export as JSON", use_container_width=True):
                    circuit_json = json.dumps(st.session_state.current_circuit.to_dict(), indent=2)
                    b64 = base64.b64encode(circuit_json.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{st.session_state.current_circuit.name}.json" class="download-button">Download JSON</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with export_cols[1]:
                if st.button("💾 Save Circuit", use_container_width=True):
                    if st.session_state.current_circuit not in st.session_state.saved_circuits:
                        st.session_state.saved_circuits.append(st.session_state.current_circuit)
                        st.success(f"✅ Circuit '{st.session_state.current_circuit.name}' saved successfully!")
                    else:
                        st.info("This circuit is already saved.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Truth table section with modern styling
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>📊 Truth Table</h3>
            """, unsafe_allow_html=True)
            
            if st.checkbox("📋 Show Truth Table"):
                if len(st.session_state.current_circuit.inputs) <= 8:
                    truth_table = generate_truth_table(st.session_state.current_circuit)
                    st.dataframe(truth_table, use_container_width=True)
                else:
                    st.warning(f"⚠️ Truth table would be too large ({2**len(st.session_state.current_circuit.inputs)} rows). Please use a circuit with 8 or fewer inputs.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class='custom-card' style='text-align: center; padding: 40px;'>
                <img src='https://img.icons8.com/fluency/96/000000/circuit.png' style='width: 96px; margin-bottom: 20px;'/>
                <h3 style='color: #1e3c72; margin-bottom: 10px;'>No Circuit Loaded</h3>
                <p style='color: #666; font-size: 16px;'>
                    Please create or load a circuit first to view its properties.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Simulation Tab - Replace with new implementation
    with main_tabs[2]:
        render_simulation_tab()
    
    # Analysis Tab
    with main_tabs[3]:
        st.subheader("📊 Circuit Analysis")
        if st.session_state.current_circuit:
            analysis_tabs = st.tabs(["Timing", "Power", "Reliability"])
            
            with analysis_tabs[0]:
                st.markdown("### Timing Analysis")
                if st.button("Analyze Timing", key="analyze_timing"):
                    with st.spinner("Analyzing timing characteristics..."):
                        timing_results = analyze_circuit_timing(st.session_state.current_circuit)
                        st.json(timing_results)
            
            with analysis_tabs[1]:
                st.markdown("### Power Analysis")
                if st.button("Analyze Power", key="analyze_power"):
                    with st.spinner("Analyzing power consumption..."):
                        st.markdown("""
                        **Power Consumption:**
                        - Static Power: 0.1mW
                        - Dynamic Power: 1.2mW
                        - Total Power: 1.3mW
                        """)
            
            with analysis_tabs[2]:
                st.markdown("### Reliability Analysis")
                if st.button("Analyze Reliability", key="analyze_reliability"):
                    with st.spinner("Analyzing reliability metrics..."):
                        st.markdown("""
                        **Reliability Metrics:**
                        - MTBF: 100,000 hours
                        - Failure Rate: 0.001%
                        - Temperature Range: -40°C to 85°C
                        """)
        else:
            st.info("No circuit loaded. Please create or load a circuit first.")
    
    # Oscilloscope Tab
    with main_tabs[4]:
        st.subheader("🔍 Oscilloscope")
        if st.session_state.current_circuit:
            # Settings
            settings_cols = st.columns(4)
            with settings_cols[0]:
                st.selectbox("Timebase", 
                           options=["1 ns/div", "10 ns/div", "100 ns/div", "1 µs/div"],
                           key="osc_timebase")
            with settings_cols[1]:
                st.selectbox("Voltage Scale",
                           options=["1 V/div", "2 V/div", "5 V/div"],
                           key="osc_voltage")
            with settings_cols[2]:
                st.selectbox("Trigger Mode",
                           options=["Auto", "Normal", "Single"],
                           key="osc_trigger")
            with settings_cols[3]:
                st.slider("Trigger Level", 0.0, 5.0, 2.5, 0.1,
                         key="osc_trigger_level")
            
            # Display area
            st.plotly_chart(create_timing_diagram(st.session_state.current_circuit,
                                                {"A": [0,1,0,1], "B": [0,0,1,1]}),
                           use_container_width=True)
        else:
            st.info("No circuit loaded. Please create or load a circuit first.")
    
    # Logic Analyzer Tab
    with main_tabs[5]:
        render_logic_analyzer_tab()
    
    # Multimeter Tab
    with main_tabs[6]:
        render_multimeter_tab()
    
    # AI Assistant Tab
    with main_tabs[7]:
        render_ai_assistant_tab()

# For the missing trigger_conditions variable that might be used elsewhere
trigger_conditions = ["Rising Edge", "Falling Edge", "Level High", "Level Low"]

# Define a function for generating complex circuits from prompts
def generate_complex_circuit_from_prompt(prompt: str) -> Circuit:
    """Generate a more complex circuit using the generate_circuit_from_prompt function."""
    try:
        return generate_circuit_from_prompt(prompt)
    except Exception as e:
        st.error(f"Error generating complex circuit: {str(e)}")
        return create_half_adder()  # Fallback to a simple circuit

# Add these functions to your app.py file after the dataclass definitions 
# and before the main function

def get_circuit_explanation(circuit: Circuit) -> str:
    """Generate a human-readable explanation of the circuit's function."""
    try:
        # Check for common circuit patterns
        if circuit.name.lower().find("half adder") >= 0:
            return "A half adder adds two single binary digits and produces a sum and carry output. It uses an XOR gate for the sum and an AND gate for the carry."
        
        elif circuit.name.lower().find("full adder") >= 0:
            return "A full adder adds three binary digits (two inputs and a carry-in) and produces a sum and carry output. It combines two half adders with an OR gate for the carry output."
        
        elif circuit.name.lower().find("bit adder") >= 0:
            return "This multi-bit adder can add binary numbers with multiple bits, propagating the carry from one position to the next. It consists of connected full adders for each bit position."
        
        elif circuit.name.lower().find("flip-flop") >= 0 or circuit.name.lower().find("flipflop") >= 0:
            return "This flip-flop is a sequential circuit that stores a single bit of data. It uses feedback to maintain its state until clocked to change."
        
        elif circuit.name.lower().find("latch") >= 0:
            return "This latch is a bistable multivibrator that can be used to store one bit of information. It has two stable states based on feedback between gates."
        
        elif circuit.name.lower().find("counter") >= 0:
            return "This counter circuit advances through a sequence of binary states when clocked. It uses flip-flops to store the current count value."
        
        else:
            # Generic explanation
            return f"This circuit contains {len(circuit.gates)} logic gates that process {len(circuit.inputs)} inputs to produce {len(circuit.outputs)} outputs through Boolean logic operations."
    
    except Exception as e:
        return f"This is a digital logic circuit with {len(circuit.gates)} gates. (Error generating detailed explanation: {str(e)})"

def create_half_adder() -> Circuit:
    """Create a half adder circuit."""
    circuit = Circuit(name="Half Adder")
    
    # Add gates
    sum_gate = LogicGate(gate_type="XOR", inputs=["A", "B"], output="S", position=(0.6, 0.3))
    carry_gate = LogicGate(gate_type="AND", inputs=["A", "B"], output="C", position=(0.6, 0.7))
    
    # Add gates to circuit
    circuit.add_gate(sum_gate)
    circuit.add_gate(carry_gate)
    
    return circuit

def create_full_adder() -> Circuit:
    """Create a full adder circuit."""
    circuit = Circuit(name="Full Adder")
    
    # Create the gates
    xor1 = LogicGate(gate_type="XOR", inputs=["A", "B"], output="XOR1", position=(0.3, 0.2))
    xor2 = LogicGate(gate_type="XOR", inputs=["XOR1", "Cin"], output="S", position=(0.6, 0.2))
    and1 = LogicGate(gate_type="AND", inputs=["A", "B"], output="AND1", position=(0.3, 0.5))
    and2 = LogicGate(gate_type="AND", inputs=["XOR1", "Cin"], output="AND2", position=(0.6, 0.5))
    or1 = LogicGate(gate_type="OR", inputs=["AND1", "AND2"], output="Cout", position=(0.8, 0.5))
    
    # Add gates to circuit
    circuit.add_gate(xor1)
    circuit.add_gate(xor2)
    circuit.add_gate(and1)
    circuit.add_gate(and2)
    circuit.add_gate(or1)
    
    return circuit

def generate_4_bit_adder() -> Circuit:
    """Create a 4-bit ripple carry adder circuit."""
    circuit = Circuit(name="4-Bit Adder")
    
    # Create 4 full adders
    for i in range(4):
        # Create the gates for each bit position
        xor1 = LogicGate(gate_type="XOR", inputs=[f"A{i}", f"B{i}"], output=f"XOR1_{i}", position=(0.2 + i*0.2, 0.1 + i*0.1))
        xor2 = LogicGate(gate_type="XOR", inputs=[f"XOR1_{i}", f"Cin_{i}"], output=f"S{i}", position=(0.4 + i*0.2, 0.1 + i*0.1))
        and1 = LogicGate(gate_type="AND", inputs=[f"A{i}", f"B{i}"], output=f"AND1_{i}", position=(0.2 + i*0.2, 0.3 + i*0.1))
        and2 = LogicGate(gate_type="AND", inputs=[f"XOR1_{i}", f"Cin_{i}"], output=f"AND2_{i}", position=(0.4 + i*0.2, 0.3 + i*0.1))
        or1 = LogicGate(gate_type="OR", inputs=[f"AND1_{i}", f"AND2_{i}"], output=f"Cout_{i}", position=(0.6 + i*0.2, 0.3 + i*0.1))
        
        # Add gates to circuit
        circuit.add_gate(xor1)
        circuit.add_gate(xor2)
        circuit.add_gate(and1)
        circuit.add_gate(and2)
        circuit.add_gate(or1)
        
        # Connect carry to next bit, except for the last bit
        if i < 3:
            # Rename the output of the OR gate to be the Cin of the next bit
            or1.output = f"Cin_{i+1}"
    
    # Manually add the initial carry in
    circuit.inputs.add("Cin_0")
    
    # Make sure the final carry out is included in the outputs
    circuit.outputs.add("Cout_3")
    
    return circuit

def create_sr_latch() -> Circuit:
    """Create an SR latch circuit."""
    circuit = Circuit(name="SR Latch")
    
    # Create the gates
    nor1 = LogicGate(gate_type="NOR", inputs=["S", "Q_bar"], output="Q", position=(0.3, 0.3))
    nor2 = LogicGate(gate_type="NOR", inputs=["R", "Q"], output="Q_bar", position=(0.7, 0.7))
    
    # Add gates to circuit
    circuit.add_gate(nor1)
    circuit.add_gate(nor2)
    
    return circuit

def create_d_flip_flop() -> Circuit:
    """Create a D flip-flop circuit."""
    circuit = Circuit(name="D Flip-Flop")
    
    # Create the gates
    not_gate = LogicGate(gate_type="NOT", inputs=["D"], output="D_bar", position=(0.2, 0.7))
    nand1 = LogicGate(gate_type="NAND", inputs=["D", "CLK"], output="X", position=(0.4, 0.3))
    nand2 = LogicGate(gate_type="NAND", inputs=["D_bar", "CLK"], output="Y", position=(0.4, 0.7))
    nand3 = LogicGate(gate_type="NAND", inputs=["X", "Q_bar"], output="Q", position=(0.6, 0.3))
    nand4 = LogicGate(gate_type="NAND", inputs=["Y", "Q"], output="Q_bar", position=(0.6, 0.7))
    
    # Add gates to circuit
    circuit.add_gate(not_gate)
    circuit.add_gate(nand1)
    circuit.add_gate(nand2)
    circuit.add_gate(nand3)
    circuit.add_gate(nand4)
    
    return circuit

def create_2_bit_counter() -> Circuit:
    """Create a 2-bit counter circuit."""
    circuit = Circuit(name="2-Bit Counter")
    
    # Create the gates for the first flip-flop (LSB)
    not1 = LogicGate(gate_type="NOT", inputs=["Q0"], output="Q0_bar", position=(0.3, 0.2))
    and1 = LogicGate(gate_type="AND", inputs=["CLK", "1"], output="CLK_FF0", position=(0.2, 0.3))
    dff1 = LogicGate(gate_type="BUFFER", inputs=["Q0_bar"], output="Q0", position=(0.5, 0.3))
    
    # Create the gates for the second flip-flop (MSB)
    not2 = LogicGate(gate_type="NOT", inputs=["Q1"], output="Q1_bar", position=(0.6, 0.5))
    and2 = LogicGate(gate_type="AND", inputs=["Q0", "CLK"], output="CLK_FF1", position=(0.5, 0.6))
    dff2 = LogicGate(gate_type="BUFFER", inputs=["Q1_bar"], output="Q1", position=(0.8, 0.6))
    
    # Add gates to circuit
    circuit.add_gate(not1)
    circuit.add_gate(and1)
    circuit.add_gate(dff1)
    circuit.add_gate(not2)
    circuit.add_gate(and2)
    circuit.add_gate(dff2)
    
    # Add constant signal
    circuit.inputs.add("1")
    
    return circuit

def get_educational_content(circuit: Circuit) -> Dict[str, str]:
    """Generate educational content about the circuit."""
    content = {
        "basic_concept": "",
        "how_it_works": "",
        "real_world_applications": "",
        "learning_tips": "",
        "quiz_questions": []
    }
    
    # Determine circuit type and provide appropriate content
    circuit_name = circuit.name.lower()
    
    if "half adder" in circuit_name:
        content["basic_concept"] = "A half adder is a fundamental digital circuit that adds two single binary digits."
        content["how_it_works"] = """
        1. Uses XOR gate for Sum output (S)
        2. Uses AND gate for Carry output (C)
        3. Truth Table:
           A B | S C
           0 0 | 0 0
           0 1 | 1 0
           1 0 | 1 0
           1 1 | 0 1
        """
        content["real_world_applications"] = """
        - Basic building block in arithmetic logic units (ALU)
        - Used in calculators and processors
        - Foundation for more complex adder circuits
        """
        content["learning_tips"] = """
        - Start by understanding XOR operation for sum
        - Remember AND operation determines carry
        - Practice with binary addition examples
        """
        content["quiz_questions"] = [
            {"question": "What is the sum when inputs A=1 and B=1?", "answer": "0"},
            {"question": "When does the carry output become 1?", "answer": "When both inputs are 1"},
            {"question": "How many outputs does a half adder have?", "answer": "2 (Sum and Carry)"}
        ]
    
    elif "full adder" in circuit_name:
        content["basic_concept"] = "A full adder adds three binary digits (including carry from previous addition)."
        content["how_it_works"] = """
        1. Combines two half adders and an OR gate
        2. Processes three inputs: A, B, and Carry-in (Cin)
        3. Produces Sum (S) and Carry-out (Cout)
        4. Uses cascaded XOR gates for sum calculation
        """
        content["real_world_applications"] = """
        - Multi-bit binary addition in processors
        - Digital arithmetic circuits
        - Part of binary multipliers
        """
        content["learning_tips"] = """
        - Understand half adder first
        - Focus on carry propagation
        - Practice with three-input combinations
        """
        content["quiz_questions"] = [
            {"question": "What happens when all inputs are 1?", "answer": "Sum=1, Carry=1"},
            {"question": "How many inputs does a full adder have?", "answer": "3"},
            {"question": "What's the difference from a half adder?", "answer": "It has an extra input for carry-in"}
        ]
    
    elif "flip-flop" in circuit_name:
        content["basic_concept"] = "A flip-flop is a sequential circuit that stores one bit of data."
        content["how_it_works"] = """
        1. Changes state based on clock signal
        2. Maintains state until next clock edge
        3. Uses feedback for state retention
        """
        content["real_world_applications"] = """
        - Memory elements in digital systems
        - Registers and counters
        - Sequential logic circuits
        """
        content["learning_tips"] = """
        - Focus on clock behavior
        - Understand state transitions
        - Practice timing diagrams
        """
        content["quiz_questions"] = [
            {"question": "When does a D flip-flop update its output?", "answer": "On the clock edge"},
            {"question": "What happens to the output when clock is low?", "answer": "Maintains previous state"},
            {"question": "How many stable states does a flip-flop have?", "answer": "2"}
        ]
    
    else:
        # Generic content for other circuits
        content["basic_concept"] = f"This is a {circuit.name} with {len(circuit.gates)} gates."
        content["how_it_works"] = """
        1. Processes digital inputs through logic gates
        2. Combines basic logic operations
        3. Produces output based on gate configurations
        """
        content["real_world_applications"] = """
        - Digital logic systems
        - Computer hardware
        - Electronic control systems
        """
        content["learning_tips"] = """
        - Study the truth table
        - Understand each gate's function
        - Practice with different inputs
        """
        content["quiz_questions"] = [
            {"question": f"How many inputs does this circuit have?", "answer": str(len(circuit.inputs))},
            {"question": f"How many outputs does it produce?", "answer": str(len(circuit.outputs))},
            {"question": "What types of gates are used?", "answer": ", ".join(set(gate.gate_type for gate in circuit.gates))}
        ]
    
    return content

# Update the render_simulation_tab function to include educational content
def render_simulation_tab():
    if st.session_state.current_circuit:
        st.markdown(f"""
        <div class='custom-card'>
            <h2 style='color: #1e3c72; margin-bottom: 20px;'>
                ▶️ Circuit Simulation: {st.session_state.current_circuit.name}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Add tabs for simulation and learning with modern styling
        sim_tabs = st.tabs([
            "🎮 Interactive Simulation",
            "📚 Learn & Practice",
            "❓ Quiz"
        ])
        
        # Interactive Simulation Tab
        with sim_tabs[0]:
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>🔄 Interactive Circuit</h3>
            """, unsafe_allow_html=True)
            
            try:
                schematic_buffer = draw_professional_schematic(st.session_state.current_circuit)
                schematic_image = Image.open(schematic_buffer)
                st.image(schematic_image, caption=f"{st.session_state.current_circuit.name} Interactive Schematic", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate interactive schematic: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Input controls with modern styling
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>🎛️ Input Controls</h3>
            """, unsafe_allow_html=True)
            
            inputs = sorted(list(st.session_state.current_circuit.inputs))
            cols_per_row = 4
            
            for i in range(0, len(inputs), cols_per_row):
                cols = st.columns(min(cols_per_row, len(inputs) - i))
                for j in range(min(cols_per_row, len(inputs) - i)):
                    with cols[j]:
                        inp = inputs[i + j]
                        current_value = st.session_state.input_values.get(inp, False)
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                            <p style='color: #1e3c72; margin-bottom: 5px; font-weight: bold;'>{inp}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.input_values[inp] = st.toggle(
                            f"{'ON' if current_value else 'OFF'}", 
                            value=current_value,
                            key=f"sim_input_{inp}"
                        )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Run simulation button with modern styling
            st.markdown("""
            <div class='custom-card' style='text-align: center;'>
            """, unsafe_allow_html=True)
            
            if st.button("▶️ Run Simulation", key="run_sim_button", type="primary", use_container_width=True):
                with st.spinner("🔄 Simulating circuit..."):
                    time.sleep(0.5)
                    outputs = st.session_state.current_circuit.simulate(st.session_state.input_values)
                    
                    # Store simulation result
                    sim_result = SimulationResult(
                        timestamp=time.time(),
                        circuit_name=st.session_state.current_circuit.name,
                        inputs=st.session_state.input_values.copy(),
                        outputs=outputs.copy()
                    )
                    st.session_state.simulation_history.append(sim_result)
                    
                    # Display outputs with enhanced visual indicators
                    st.markdown("""
                    <div style='margin-top: 20px;'>
                        <h3 style='color: #1e3c72; margin-bottom: 20px;'>📊 Output Values</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    output_cols = st.columns(min(4, len(outputs)))
                    for i, (out_name, out_value) in enumerate(outputs.items()):
                        with output_cols[i % 4]:
                            st.markdown(
                                f"""
                                <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
                                    <h4 style='color: #1e3c72; margin-bottom: 10px;'>{out_name}</h4>
                                    <div style='width: 60px; height: 60px; background: linear-gradient(135deg, {'#00ff87, #60efff' if out_value else '#ff6b6b, #feca57'}); 
                                              border-radius: 50%; margin: 10px auto; display: flex; align-items: center; justify-content: center;
                                              box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                                        <span style='color: white; font-size: 24px; font-weight: bold;'>{1 if out_value else 0}</span>
                                    </div>
                                    <p style='color: #666; margin-top: 10px;'>{'HIGH' if out_value else 'LOW'}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Learn & Practice Tab
        with sim_tabs[1]:
            # Get educational content
            edu_content = get_educational_content(st.session_state.current_circuit)
            
            # Display educational content with modern styling
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>📚 Basic Concept</h3>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                    <p style='color: #666; font-size: 16px;'>{}</p>
                </div>
            </div>
            """.format(edu_content["basic_concept"]), unsafe_allow_html=True)
            
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>⚙️ How It Works</h3>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                    <pre style='color: #666; font-size: 16px; white-space: pre-wrap;'>{}</pre>
                </div>
            </div>
            """.format(edu_content["how_it_works"]), unsafe_allow_html=True)
            
            # Interactive visualization with modern styling
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>🎬 Interactive Visualization</h3>
            """, unsafe_allow_html=True)
            
            if st.button("▶️ Show Signal Flow Animation", key="show_animation"):
                st.markdown("""
                <div style='text-align: center; padding: 40px; background: #f8f9fa; border-radius: 8px;'>
                    <div style='font-size: 48px; margin-bottom: 20px;'>🔄</div>
                    <p style='color: #666; font-size: 16px;'>
                        Signal flow animation would appear here
                        <br>(Visual representation of how signals propagate through the circuit)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Applications and tips with modern styling
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class='custom-card'>
                    <h3 style='color: #1e3c72; margin-bottom: 20px;'>🌟 Real-World Applications</h3>
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <p style='color: #666; font-size: 16px;'>{}</p>
                    </div>
                </div>
                """.format(edu_content["real_world_applications"]), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='custom-card'>
                    <h3 style='color: #1e3c72; margin-bottom: 20px;'>💡 Learning Tips</h3>
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <p style='color: #666; font-size: 16px;'>{}</p>
                    </div>
                </div>
                """.format(edu_content["learning_tips"]), unsafe_allow_html=True)
            
            # Timing diagram with modern styling
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>📈 Timing Diagram</h3>
            """, unsafe_allow_html=True)
            
            if st.button("📊 Generate Timing Diagram", key="gen_timing"):
                fig = create_timing_diagram(st.session_state.current_circuit, 
                                         {"A": [0,1,0,1], "B": [0,0,1,1]})
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Quiz Tab
        with sim_tabs[2]:
            st.markdown("""
            <div class='custom-card'>
                <h3 style='color: #1e3c72; margin-bottom: 20px;'>🎯 Test Your Knowledge</h3>
            """, unsafe_allow_html=True)
            
            if "quiz_score" not in st.session_state:
                st.session_state.quiz_score = 0
            
            if "quiz_submitted" not in st.session_state:
                st.session_state.quiz_submitted = False
            
            # Display quiz questions with modern styling
            for i, q in enumerate(edu_content["quiz_questions"]):
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='color: #1e3c72; font-weight: bold; margin-bottom: 10px;'>Q{i+1}: {q['question']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                user_answer = st.text_input("Your answer:", key=f"quiz_q{i}")
                
                if st.session_state.quiz_submitted:
                    if user_answer.lower() == q['answer'].lower():
                        st.markdown("""
                        <div style='background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            ✅ Correct!
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            ❌ Incorrect. The answer is: {q['answer']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Quiz submission buttons with modern styling
            if not st.session_state.quiz_submitted:
                if st.button("📝 Submit Quiz", key="submit_quiz", type="primary"):
                    st.session_state.quiz_submitted = True
                    st.rerun()
            else:
                if st.button("🔄 Try Again", key="reset_quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = 0
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class='custom-card' style='text-align: center; padding: 40px;'>
            <img src='https://img.icons8.com/fluency/96/000000/circuit.png' style='width: 96px; margin-bottom: 20px;'/>
            <h3 style='color: #1e3c72; margin-bottom: 10px;'>No Circuit Loaded</h3>
            <p style='color: #666; font-size: 16px;'>
                Please create or load a circuit first to start simulation.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Implement the Logic Analyzer tab
def render_logic_analyzer_tab():
    st.subheader("🔍 Logic Analyzer")
    
    if not st.session_state.current_circuit:
        st.info("No circuit loaded. Please create or load a circuit first.")
        return
    
    # Settings panel
    with st.expander("Logic Analyzer Settings", expanded=True):
        settings_cols = st.columns(4)
        
        with settings_cols[0]:
            st.selectbox("Sample Rate",
                        options=["10 MHz", "50 MHz", "100 MHz", "200 MHz", "500 MHz", "1 GHz"],
                        index=2,
                        key="la_sample_rate")
            
        with settings_cols[1]:
            available_channels = list(st.session_state.current_circuit.inputs) + \
                               list(st.session_state.current_circuit.outputs)
            st.multiselect("Active Channels",
                          options=available_channels,
                          default=available_channels[:4],
                          key="la_active_channels")
            
        with settings_cols[2]:
            st.selectbox("Trigger Source",
                        options=["None"] + available_channels,
                        index=0,
                        key="la_trigger_source")
            
        with settings_cols[3]:
            st.selectbox("Trigger Type",
                        options=["Rising Edge", "Falling Edge", "Both Edges", "High Level", "Low Level"],
                        index=0,
                        key="la_trigger_type")
    
    # Control buttons
    control_cols = st.columns(4)
    with control_cols[0]:
        if st.button("Start Capture", type="primary", key="la_start"):
            st.session_state.logic_analyzer_settings['running'] = True
            
    with control_cols[1]:
        if st.button("Stop", key="la_stop"):
            st.session_state.logic_analyzer_settings['running'] = False
            
    with control_cols[2]:
        if st.button("Single Capture", key="la_single"):
            st.session_state.logic_analyzer_settings['running'] = True
            # Simulate single capture
            time.sleep(0.5)
            st.session_state.logic_analyzer_settings['running'] = False
            
    with control_cols[3]:
        if st.button("Clear", key="la_clear"):
            st.session_state.logic_analyzer_settings['data'] = {}
    
    # Display area
    st.markdown("### Signal Display")
    
    # Generate sample data for visualization
    if st.session_state.logic_analyzer_settings['running'] or \
       len(st.session_state.logic_analyzer_settings.get('data', {})) > 0:
        
        # Create figure for signal display
        fig = go.Figure()
        
        # Generate sample data for each channel
        time_points = np.linspace(0, 1000, 1000)
        for i, channel in enumerate(st.session_state.logic_analyzer_settings.get('channels', [])):
            # Generate digital signal
            if channel in st.session_state.input_values:
                base_value = st.session_state.input_values[channel]
            else:
                base_value = False
            
            signal = np.zeros_like(time_points)
            for j in range(0, len(time_points), 100):
                signal[j:j+50] = 1 if base_value else 0
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=time_points,
                y=signal + i*2,  # Offset each channel
                mode='lines',
                name=channel,
                line=dict(shape='hv')
            ))
        
        # Update layout
        fig.update_layout(
            title="Logic Analyzer Display",
            xaxis_title="Time (ns)",
            yaxis_title="Channel",
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            yaxis=dict(
                ticktext=st.session_state.logic_analyzer_settings.get('channels', []),
                tickvals=list(range(len(st.session_state.logic_analyzer_settings.get('channels', []))))
            )
        )
        
        # Add cursors
        fig.add_vline(x=250, line_dash="dash", line_color="red", annotation_text="Cursor 1")
        fig.add_vline(x=750, line_dash="dash", line_color="blue", annotation_text="Cursor 2")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Measurements panel
        st.markdown("### Measurements")
        measure_cols = st.columns(4)
        
        with measure_cols[0]:
            st.metric("Time Between Cursors", "500 ns")
        with measure_cols[1]:
            st.metric("Frequency", "2 MHz")
        with measure_cols[2]:
            st.metric("Pulse Width", "250 ns")
        with measure_cols[3]:
            st.metric("Duty Cycle", "50%")
        
        # Protocol decoder
        st.markdown("### Protocol Decoder")
        decoder_cols = st.columns(2)
        
        with decoder_cols[0]:
            st.selectbox("Protocol",
                        options=["None", "SPI", "I2C", "UART", "Custom"],
                        key="la_protocol")
            
        with decoder_cols[1]:
            if st.button("Decode"):
                st.info("Protocol decoding would appear here")
    else:
        st.info("Click 'Start Capture' to begin capturing signals")

# Implement the Multimeter tab
def render_multimeter_tab():
    st.subheader("⚡ Digital Multimeter")
    
    if not st.session_state.current_circuit:
        st.info("No circuit loaded. Please create or load a circuit first.")
        return
    
    # Settings
    settings_cols = st.columns(3)
    
    with settings_cols[0]:
        mode = st.selectbox("Measurement Mode",
                           options=["Logic Level", "Voltage", "Timing", "Statistics"],
                           key="mm_mode")
        
    with settings_cols[1]:
        available_points = list(st.session_state.current_circuit.inputs) + \
                         list(st.session_state.current_circuit.outputs)
        probe_point = st.selectbox("Measurement Point",
                                 options=["Select Point"] + available_points,
                                 key="mm_point")
        
    with settings_cols[2]:
        st.selectbox("Range",
                    options=["Auto", "TTL (0-5V)", "CMOS (0-3.3V)", "Custom"],
                    key="mm_range")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        adv_cols = st.columns(3)
        with adv_cols[0]:
            st.slider("Sample Rate (kHz)", 1, 1000, 100, key="mm_sample_rate")
        with adv_cols[1]:
            st.slider("Averaging", 1, 100, 10, key="mm_averaging")
        with adv_cols[2]:
            st.checkbox("Auto-ranging", value=True, key="mm_auto_range")
    
    # Display measurements if a point is selected
    if probe_point != "Select Point":
        # Get the signal value
        signal_value = False
        if probe_point in st.session_state.input_values:
            signal_value = st.session_state.input_values[probe_point]
        elif probe_point in st.session_state.current_circuit.outputs and \
             hasattr(st.session_state, 'simulation_history') and \
             st.session_state.simulation_history:
            signal_value = st.session_state.simulation_history[-1].outputs.get(probe_point, False)
        
        # Digital display
        st.markdown(
            f"""
            <div style="background-color:black; color:white; padding:20px; border-radius:10px; text-align:center; margin:20px 0;">
                <h2 style="margin:0; font-family:monospace;">{probe_point}</h2>
                <div style="background-color:#303030; margin:10px 0; padding:15px; border-radius:5px;">
                    <h1 style="color:#00ff00; font-family:monospace; margin:0; font-size:48px;">
                        {"HIGH (5V)" if signal_value else "LOW (0V)"}
                    </h1>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:10px;">
                    <div style="background-color:#404040; padding:5px 10px; border-radius:5px; font-family:monospace;">
                        Mode: {mode}
                    </div>
                    <div style="background-color:#404040; padding:5px 10px; border-radius:5px; font-family:monospace;">
                        Status: {"VALID" if signal_value is not None else "NO SIGNAL"}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Measurements display
        measure_cols = st.columns(4)
        
        with measure_cols[0]:
            st.metric("Logic Level", "HIGH" if signal_value else "LOW",
                     delta="5V" if signal_value else "0V")
            
        with measure_cols[1]:
            st.metric("Frequency", "N/A" if not signal_value else "50 MHz",
                     delta="+/-0.1 MHz" if signal_value else None)
            
        with measure_cols[2]:
            st.metric("Pulse Width", "N/A" if not signal_value else "10 ns",
                     delta="±1ns" if signal_value else None)
            
        with measure_cols[3]:
            st.metric("Edge Time", "N/A" if not signal_value else "2 ns",
                     delta="±0.2ns" if signal_value else None)
        
        # Waveform display
        st.markdown("### Signal Waveform")
        
        # Generate sample waveform
        time_points = np.linspace(0, 100, 1000)
        if signal_value:
            # Generate a signal with some transitions
            signal = np.zeros_like(time_points)
            for i in range(0, len(time_points), 100):
                signal[i:i+50] = 5.0
        else:
            signal = np.zeros_like(time_points)
        
        # Add some noise
        signal += np.random.normal(0, 0.05, size=len(time_points))
        
        # Create waveform plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_points,
            y=signal,
            mode='lines',
            name=probe_point,
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Signal Waveform: {probe_point}",
            xaxis_title="Time (ns)",
            yaxis_title="Voltage (V)",
            height=300,
            showlegend=True,
            plot_bgcolor='white',
            yaxis=dict(range=[-0.5, 5.5])
        )
        
        # Add threshold lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                     annotation_text="LOW Threshold")
        fig.add_hline(y=2.0, line_dash="dash", line_color="green",
                     annotation_text="HIGH Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("### Signal Statistics")
        stats_df = pd.DataFrame({
            "Measurement": ["Minimum", "Maximum", "Average", "RMS", "Peak-to-Peak"],
            "Value": [
                "0.0 V",
                "5.0 V" if signal_value else "0.2 V",
                f"{2.5 if signal_value else 0.1:.1f} V",
                f"{3.5 if signal_value else 0.15:.1f} V",
                "5.0 V" if signal_value else "0.2 V"
            ],
            "Status": ["Normal"] * 5
        })
        st.dataframe(stats_df, use_container_width=True)

# Add this at the end of the file to ensure the app runs
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.button("Restart App"):
            st.rerun()


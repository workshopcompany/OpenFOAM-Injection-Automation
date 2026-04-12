import streamlit as st
import requests
import json
import time
import google.generativeai as genai

# --- 1. SECURITY CONFIGURATION ---
# IMPORTANT: Do not hardcode keys. 
# Local: Create '.streamlit/secrets.toml'
# Streamlit Cloud: Set in 'Settings > Secrets'
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    ZAPIER_WEBHOOK_URL = st.secrets["ZAPIER_WEBHOOK_URL"]
    
    # Gemini AI Setup
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error("Missing Secrets: Please configure GEMINI_API_KEY and ZAPIER_WEBHOOK_URL.")
    st.stop()

# --- 2. MATERIAL DATABASE ---
MATERIAL_DB = {
    "MIM (Metal Injection Molding)": {
        "metals": ["SUS630 (17-4PH)", "Inconel", "Ti", "Al", "Cu"],
        "binders": ["Wax-base", "POM-base"]
    },
    "Super Engineering Plastics": ["PEEK", "PPS"],
    "Hot & Bio (Eco/EV)": ["PLA", "PHA", "EV Thermal Plastic"],
    "General Plastics": ["ABS", "PC", "PP", "POM", "PA66+GF30", "PC+ABS"]
}

st.set_page_config(page_title="MIM AI Simulator", layout="wide")

# --- 3. AI LOGIC FUNCTION ---
def get_ai_injection_setup(material_name):
    prompt = f"""
    You are an expert injection molding engineer. 
    Provide optimal simulation parameters for {material_name}.
    MIM-specific constraints: 
    - If Wax-base: Nozzle 175C, Mold 35C, Speed 30-40mm/s.
    - If POM-base: Nozzle 180C, Mold 80C, Speed 40-50mm/s.
    
    Return ONLY a JSON object:
    {{
        "density": <float in kg/m3>,
        "thermal_conductivity": <float in W/mK>,
        "heat_capacity": <float in J/kgK>,
        "nozzle_temp": <int in Celsius>,
        "mold_temp": <int in Celsius>,
        "injection_speed": <int in mm/s>,
        "gate_qty": 1,
        "gate_loc": "[0.001, 0.001, 0.001]"
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_str)
    except Exception as e:
        st.error(f"AI Suggestion Error: {e}")
        return None

# --- 4. UI LAYOUT ---
st.title("🚀 MIM & Plastic Injection AI Simulator")
st.caption("14+ Years Materials Engineering Expertise Integrated")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Geometry & Gate Upload")
    uploaded_file = st.file_uploader("Upload STL File", type=['stl'])
    if uploaded_file:
        # AUTOMATICALLY RENAME TO mold.stl FOR OPENFOAM CONSISTENCY
        uploaded_file.name = "mold.stl"
        st.info(f"File processed as: **{uploaded_file.name}**")

with col2:
    st.subheader("2. Material Selection")
    category = st.selectbox("Select Category", list(MATERIAL_DB.keys()))
    
    if category == "MIM (Metal Injection Molding)":
        m = st.selectbox("Metal Type", MATERIAL_DB[category]["metals"])
        b = st.selectbox("Binder System", MATERIAL_DB[category]["binders"])
        target_material = f"{m} {b}"
    else:
        target_material = st.selectbox("Material Type", MATERIAL_DB[category])

# --- 5. AI SUGGESTION & FORM ---
st.markdown("---")
st.subheader("3. AI-Powered Process Optimization")

if st.button("Generate Optimized Setup"):
    with st.spinner("Gemini AI calculating optimal parameters..."):
        setup_data = get_ai_injection_setup(target_material)
        if setup_data:
            st.session_state['current_setup'] = setup_data

if 'current_setup' in st.session_state:
    setup = st.session_state['current_setup']
    
    with st.form("confirm_setup"):
        st.warning("⚡ AI suggested the following parameters. Adjust if needed.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 🧪 Material Properties")
            rho = st.number_input("Density (kg/m³)", value=float(setup['density']))
            k = st.number_input("Thermal Cond. (W/m·K)", value=float(setup['thermal_conductivity']))
            cp = st.number_input("Heat Capacity (J/kg·K)", value=float(setup['heat_capacity']))
            
        with c2:
            st.markdown("#### ⚙️ Machine Settings")
            t_n = st.number_input("Nozzle Temperature (℃)", value=int(setup['nozzle_temp']))
            t_m = st.number_input("Mold Temperature (℃)", value=int(setup['mold_temp']))
            v_i = st.number_input("Injection Speed (mm/s)", value=int(setup['injection_speed']))
            
        with c3:
            st.markdown("#### 🎯 Gate Setup")
            g_q = st.number_input("Number of Gates", value=int(setup['gate_qty']))
            g_l = st.text_input("Gate Coordinates [X, Y, Z]", value=setup['gate_loc'])

        if st.form_submit_button("Approve & Trigger Simulation", type="primary"):
            final_payload = {
                "material": target_material,
                "properties": {"rho": rho, "k": k, "cp": cp},
                "machine": {"nozzle": t_n, "mold": t_m, "speed": v_i},
                "gate": {"qty": g_q, "loc": g_l}
            }
            # Send to GitHub Actions via Zapier
            try:
                requests.post(ZAPIER_WEBHOOK_URL, json=final_payload)
                st.success("✅ Job dispatched! Check GitHub Actions for progress.")
                st.session_state['sim_status'] = "Running"
            except Exception as e:
                st.error(f"Webhook Error: {e}")

# --- 6. RESULTS VISUALIZATION ---
st.markdown("---")
if st.session_state.get('sim_status') == "Running":
    st.subheader("4. Real-time Flow & Thermal Visualization")
    st.info("Generating filling sequences and thermal gradient maps...")
    
    step = st.slider("Select Filling Step (1-10)", 1, 10, 1)
    # Placeholder: Replace with actual GitHub artifact URL logic later
    st.image(f"https://via.placeholder.com/1000x500.png?text=Step+{step}:+Filling+Pattern+%26+Thermal+Map", use_container_width=True)

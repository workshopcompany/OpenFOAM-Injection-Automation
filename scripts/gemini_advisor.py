import os
import json
import urllib.request

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Local backup database
MATERIAL_DB = {
    "PP":      {"nu": 1e-3,  "rho": 900,  "Tmelt": 230, "Tmold": 40},
    "ABS":     {"nu": 2e-3,  "rho": 1050, "Tmelt": 240, "Tmold": 60},
    "Nylon66": {"nu": 5e-4,  "rho": 1140, "Tmelt": 280, "Tmold": 80},
}

def get_material_properties(material_name: str) -> dict:
    """Fetch properties via Gemini API, fallback to local DB if error"""
    
    # Check local DB first
    for key in MATERIAL_DB:
        if key.upper() in material_name.upper():
            props = MATERIAL_DB[key].copy()
            props["source"] = "local_db"
            props["material"] = key
            return props

    if not GEMINI_API_KEY:
        return _fallback(material_name)

    prompt = f"""
    You are an expert in Plastic Injection Molding.
    Material Name: {material_name}
    Return ONLY a JSON object with the following keys:
    {{
      "material": "{material_name}",
      "nu": kinematic_viscosity_m2s (float),
      "rho": density_kg_m3 (float),
      "Tmelt": melt_temp_celsius (int),
      "Tmold": mold_temp_celsius (int),
      "description": "one line summary",
      "source": "gemini"
    }}
    Note: nu = dynamic_viscosity / density.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{GEMINI_URL}?key={GEMINI_API_KEY}", data=data, 
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            return json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
    except:
        return _fallback(material_name)

def _fallback(name):
    return {"material": name, "nu": 1e-3, "rho": 1000, "Tmelt": 220, "Tmold": 50, 
            "description": "Default values used", "source": "fallback"}

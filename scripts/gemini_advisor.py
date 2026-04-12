import os
import json
import urllib.request
import urllib.parse

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

MATERIAL_DB = {
    "PP":      {"nu": 1e-3,  "rho": 900,  "Tmelt": 230, "Tmold": 40},
    "ABS":     {"nu": 2e-3,  "rho": 1050, "Tmelt": 240, "Tmold": 60},
    "Nylon66": {"nu": 5e-4,  "rho": 1140, "Tmelt": 280, "Tmold": 80},
    "POM":     {"nu": 8e-4,  "rho": 1410, "Tmelt": 200, "Tmold": 90},
    "PC":      {"nu": 3e-3,  "rho": 1200, "Tmelt": 300, "Tmold": 85},
    "HDPE":    {"nu": 9e-4,  "rho": 960,  "Tmelt": 220, "Tmold": 35},
    "PET":     {"nu": 6e-4,  "rho": 1370, "Tmelt": 265, "Tmold": 70},
}

def get_material_properties(material_name: str) -> dict:
    """Gemini API로 수지 물성 추천, 실패 시 로컬 DB 사용"""
    
    # 1. 로컬 DB에서 먼저 확인
    for key in MATERIAL_DB:
        if key.upper() in material_name.upper():
            local = MATERIAL_DB[key].copy()
            local["source"] = "local_db"
            local["material"] = key
            return local

    # 2. Gemini API 호출
    if not GEMINI_API_KEY:
        return _fallback_properties(material_name)

    prompt = f"""
당신은 플라스틱 사출 성형 전문가입니다.
재료명: {material_name}

아래 JSON 형식으로만 답변하세요. 다른 텍스트 없이 JSON만 출력하세요:
{{
  "material": "{material_name}",
  "nu": 운동점도_m2s (숫자),
  "rho": 밀도_kg_m3 (숫자),
  "Tmelt": 용융온도_섭씨 (숫자),
  "Tmold": 권장금형온도_섭씨 (숫자),
  "description": "재료 특성 한줄 설명",
  "source": "gemini"
}}

참고: 운동점도 nu = 동적점도(Pa·s) / 밀도(kg/m3)
일반적인 사출 조건에서의 값을 사용하세요.
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512}
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            # JSON 파싱
            text = text.strip().strip("```json").strip("```").strip()
            props = json.loads(text)
            return props
    except Exception as e:
        print(f"[Gemini API 오류] {e} → 로컬 DB fallback")
        return _fallback_properties(material_name)


def _fallback_properties(material_name: str) -> dict:
    """알 수 없는 재료 기본값"""
    return {
        "material": material_name,
        "nu": 1e-3,
        "rho": 1000,
        "Tmelt": 220,
        "Tmold": 50,
        "description": "알 수 없는 재료 — 기본값 사용",
        "source": "fallback"
    }


if __name__ == "__main__":
    import sys
    material = sys.argv[1] if len(sys.argv) > 1 else "PP"
    props = get_material_properties(material)
    print(json.dumps(props, ensure_ascii=False, indent=2))

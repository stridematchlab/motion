# app.py
# StrideMatch MotionLab V2 – version corrigée avec scoring fonctionnel
# Prérequis: ffmpeg installé, Google Sheet public/lecture.

import streamlit as st
import numpy as np
import pandas as pd
import cv2, os, tempfile, subprocess, base64
from math import atan2, degrees
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import mediapipe as mp
import streamlit.components.v1 as components

# ---------------- CONFIG ----------------
st.set_page_config(page_title="StrideMatch MotionLab V2", layout="wide", page_icon="👟")

# ---------------- PASSWORD PROTECTION ----------------
def check_password():
    """Returns `True` if the user has the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Get password from secrets or use default
        correct_password = st.secrets.get("password", "StrideMatch2024!")
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Login page
    st.markdown("""
    <div style="text-align:center;padding:50px 20px">
        <div style="font-size:4rem;margin-bottom:20px">👟</div>
        <h1 style="background:linear-gradient(90deg,#06b6d4,#3b82f6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-size:2.5rem;font-weight:800;margin-bottom:10px">
            StrideMatch MotionLab
        </h1>
        <p style="color:#6b7280;font-size:1.1rem">🔒 Private Access</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.text_input(
            "Enter Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Your access password"
        )
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Incorrect password. Please try again.")
        
        st.markdown("""
        <div style="text-align:center;margin-top:30px;color:#9ca3af;font-size:0.85rem">
            Contact your administrator for access credentials
        </div>
        """, unsafe_allow_html=True)
    
    return False

# Check password before showing the app
if not check_password():
    st.stop()

# ---------------- MAIN APP (after authentication) ----------------
OUT_W, OUT_H = 720, 405
BLUE = (255, 102, 0)
GREEN = (57, 255, 20)
LABEL_BG = GREEN
LABEL_TXT = (25, 28, 35)
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ZkIlRaVfCIKHbXIXWb2OTJCzI0PblVUqrKEkT6IRC0s/edit#gid=0"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------- STYLE ----------------
st.markdown("""
<style>
h1{ text-align:center;font-size:2.3rem;font-weight:800;
     background:linear-gradient(90deg,#06b6d4,#3b82f6);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
[data-testid="stVideo"] video{max-width:720px!important;display:block;margin:0 auto;}
div.stButton > button:first-child{
  background:linear-gradient(90deg,#4f46e5,#3b82f6);
  color:white;border:none;border-radius:9999px;padding:0.7rem 1.35rem;
  box-shadow:0 8px 20px rgba(59,130,246,.25);font-weight:700}
div.stButton > button:first-child:hover{filter:brightness(.98)}
.kpi{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;
     background:#f9fafb;border-radius:18px;padding:16px}
.kpi > div{background:#fff;border:1px solid #eef2f7;border-radius:12px;
           padding:10px 12px;min-width:130px;text-align:center}
.card{background:#fff;border-radius:16px;box-shadow:0 10px 28px rgba(0,0,0,.06);
      padding:18px 22px;margin:14px auto;max-width:780px}
.badge{display:inline-block;background:linear-gradient(90deg,#60a5fa,#3b82f6);
       color:#fff;padding:.25rem .7rem;border-radius:9999px;font-weight:700}
.progress{height:10px;background:#e5e7eb;border-radius:9999px}
.progress > span{display:block;height:10px;border-radius:9999px}
.meta{color:#1f2937}
</style>
""", unsafe_allow_html=True)

st.title("StrideMatch MotionLab")

# ---------------- UTILS ----------------
def angle(a,b,c):
    ang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))
    ang = abs((ang + 360) % 360)
    return ang if ang <= 180 else 360 - ang

def p(lm, i, w, h):
    return (int(lm[i].x*w), int(lm[i].y*h))

def draw_label(img, text, org):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y-th-4), (x+tw+6, y), LABEL_BG, -1)
    cv2.putText(img, text, (x+3, y-3), font, scale, LABEL_TXT, thickness, cv2.LINE_AA)

def ffmpeg_optimize(path):
    try:
        out = path.replace(".mp4","_fast.mp4")
        subprocess.run(["ffmpeg","-y","-i",path,"-movflags","faststart","-vcodec","libx264",
                        "-pix_fmt","yuv420p","-preset","ultrafast","-an",out],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out if os.path.exists(out) else path
    except:
        return path

def embed_video(data, speed):
    b64 = base64.b64encode(data).decode("utf-8")
    html_code = f"""
    <video id="vid" width="720" height="405" controls playsinline style="border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.08);background:#000">
      <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    </video>
    <script>
    var vid = document.getElementById('vid');
    if(vid) {{ vid.playbackRate = {speed}; }}
    </script>
    """
    components.html(html_code, height=460)

@st.cache_data(show_spinner=False, ttl=300)
def load_catalog_from_gsheet(sheet_url: str) -> pd.DataFrame:
    """Charge le catalogue depuis Google Sheets - mise en cache 5 minutes"""
    sheet_id = sheet_url.split("/d/")[1].split("/")[0]
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df = pd.read_csv(export_url)
        # Nettoyer les noms de colonnes
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Erreur chargement catalogue: {e}")
        return pd.DataFrame()


def score_shoe(row, user_surface, user_pronation, user_preference, bio_data=None):
    """
    Calcule un score de compatibilité de 0 à 100 pour une chaussure.
    """
    score = 0.0
    max_possible = 100.0
    
    # Extraire et nettoyer les données de la chaussure
    shoe_terrain = str(row.get("terrain", "")).lower().strip()
    shoe_stability = str(row.get("stability", "")).lower().strip()
    shoe_cushioning = str(row.get("cushioning", "")).lower().strip()
    
    # Poids
    try:
        weight_str = str(row.get("weight_g", "300"))
        shoe_weight = float(weight_str.replace("g", "").replace(",", ".").strip())
    except:
        shoe_weight = 300.0
    
    # Drop
    try:
        drop_str = str(row.get("drop_mm", "10"))
        shoe_drop = float(drop_str.replace("mm", "").replace(",", ".").strip())
    except:
        shoe_drop = 10.0
    
    # Stack
    try:
        stack_val = row.get("stack_mm", row.get("stack_mm_total", "30"))
        shoe_stack = float(str(stack_val).replace("mm", "").replace(",", ".").strip())
    except:
        shoe_stack = 30.0
    
    # ===== TERRAIN (30 points) =====
    terrain_score = 0
    if user_surface == "trail":
        if shoe_terrain == "trail":
            terrain_score = 30
        elif "trail" in shoe_terrain:
            terrain_score = 25
        elif shoe_terrain in ["mixed", "mixte"]:
            terrain_score = 15
        else:
            terrain_score = 0
    elif user_surface == "road":
        if shoe_terrain in ["road", "route"]:
            terrain_score = 30
        elif "road" in shoe_terrain or "route" in shoe_terrain:
            terrain_score = 25
        elif shoe_terrain in ["mixed", "mixte"]:
            terrain_score = 15
        else:
            terrain_score = 0
    elif user_surface == "mixed":
        if shoe_terrain in ["mixed", "mixte"]:
            terrain_score = 30
        elif shoe_terrain in ["road", "route", "trail"]:
            terrain_score = 20
        else:
            terrain_score = 10
    score += terrain_score
    
    # ===== PRONATION (35 points) - C'est le critère le plus important =====
    pronation_score = 0
    if user_pronation == "overpronation":
        # Besoin de stabilité/support
        if "stability" in shoe_stability or "support" in shoe_stability:
            pronation_score = 35
        elif shoe_stability == "guidance":
            pronation_score = 25
        elif shoe_stability == "neutral":
            pronation_score = 10
        else:
            pronation_score = 5
    elif user_pronation == "neutral":
        # Chaussure neutre idéale
        if shoe_stability == "neutral":
            pronation_score = 35
        elif shoe_stability in ["light stability", "guidance"]:
            pronation_score = 25
        elif "stability" in shoe_stability:
            pronation_score = 15
        else:
            pronation_score = 20
    elif user_pronation == "underpronation":
        # Besoin d'amorti et flexibilité, PAS de stabilité
        if shoe_stability == "neutral" and shoe_cushioning in ["high", "max", "soft", "plush"]:
            pronation_score = 35
        elif shoe_cushioning in ["high", "max", "soft", "plush"]:
            pronation_score = 30
        elif shoe_stability == "neutral":
            pronation_score = 25
        elif "stability" in shoe_stability:
            pronation_score = 5  # Stabilité = mauvais pour supination
        else:
            pronation_score = 15
    score += pronation_score
    
    # ===== PRÉFÉRENCE (20 points) =====
    pref_score = 0
    if user_preference == "comfort":
        if shoe_cushioning in ["high", "max", "plush", "soft"]:
            pref_score += 12
        elif shoe_cushioning in ["medium", "balanced"]:
            pref_score += 6
        if shoe_stack >= 32:
            pref_score += 8
        elif shoe_stack >= 26:
            pref_score += 4
    elif user_preference == "stability":
        if "stability" in shoe_stability or "support" in shoe_stability:
            pref_score += 12
        elif shoe_stability == "guidance":
            pref_score += 8
        if shoe_drop >= 8:
            pref_score += 8
        elif shoe_drop >= 4:
            pref_score += 4
    elif user_preference == "responsiveness":
        if shoe_weight < 250:
            pref_score += 10
        elif shoe_weight < 280:
            pref_score += 7
        elif shoe_weight < 300:
            pref_score += 4
        if shoe_cushioning in ["responsive", "firm"]:
            pref_score += 10
        elif shoe_drop <= 6:
            pref_score += 5
    score += pref_score
    
    # ===== POIDS GÉNÉRAL (10 points) =====
    weight_score = 0
    if shoe_weight < 240:
        weight_score = 10
    elif shoe_weight < 270:
        weight_score = 8
    elif shoe_weight < 300:
        weight_score = 6
    elif shoe_weight < 330:
        weight_score = 4
    else:
        weight_score = 2
    score += weight_score
    
    # ===== BIOMÉCANIQUE (5 points bonus) =====
    if bio_data and isinstance(bio_data, dict) and len(bio_data) > 0:
        contact_time = bio_data.get("contact_time", 250)
        
        if contact_time > 280:  # Talonneur
            if shoe_cushioning in ["high", "max", "plush"] and shoe_drop >= 8:
                score += 5
            elif shoe_cushioning in ["high", "max", "plush"] or shoe_drop >= 8:
                score += 3
        elif contact_time < 220:  # Avant-pied
            if shoe_drop <= 6:
                score += 5
            elif shoe_drop <= 8:
                score += 3
        else:
            score += 3
    else:
        score += 2.5
    
    # Normaliser à 100
    final_score = min(100, (score / max_possible) * 100)
    return round(final_score, 1)


# ============== INITIALISATION SESSION STATE ==============
if "profile_gender" not in st.session_state:
    st.session_state["profile_gender"] = "Male"
if "profile_surface" not in st.session_state:
    st.session_state["profile_surface"] = "road"
if "profile_pronation" not in st.session_state:
    st.session_state["profile_pronation"] = "neutral"
if "profile_preference" not in st.session_state:
    st.session_state["profile_preference"] = "comfort"


# ---------------- TABS ----------------
tab_profile, tab_analyse, tab_results, tab_reco = st.tabs(["Profile","Analysis","Results","Recommendations"])

# ---------------- PROFIL ----------------
with tab_profile:
    st.subheader("🏃 Your Runner Profile")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.radio("Gender", ["Male","Female"], horizontal=True, 
                          index=0 if st.session_state["profile_gender"]=="Male" else 1)
        age = st.number_input("Age", 12, 80, 35)
        height = st.number_input("Height (cm)", 140, 210, 175)
    with c2:
        weight = st.number_input("Weight (kg)", 40, 150, 75)
        level = st.selectbox("Level 🏅", ["Beginner","Intermediate","Advanced"])
        surface = st.selectbox("Surface 🌍", ["road","trail","mixed"],
                               index=["road","trail","mixed"].index(st.session_state["profile_surface"]) if st.session_state["profile_surface"] in ["road","trail","mixed"] else 0)
    with c3:
        distance_hebdo = st.selectbox("Weekly Distance 🏃", 
                                      ["0-10 km","10-20 km","20-30 km","30-40 km","40-50 km","+60 km"])
        pronation = st.selectbox("Pronation 🦶", ["neutral","overpronation","underpronation"],
                                 index=["neutral","overpronation","underpronation"].index(st.session_state["profile_pronation"]))
        preference = st.selectbox("Preference ⚙️", ["comfort","stability","responsiveness"],
                                  index=["comfort","stability","responsiveness"].index(st.session_state["profile_preference"]) if st.session_state["profile_preference"] in ["comfort","stability","responsiveness"] else 0)
    
    # IMPORTANT: Sauvegarder dans session_state
    st.session_state["profile_gender"] = gender
    st.session_state["profile_surface"] = surface
    st.session_state["profile_pronation"] = pronation
    st.session_state["profile_preference"] = preference

# ---------------- ANALYSE ----------------
with tab_analyse:
    st.subheader("Biomechanical Analysis 🎥")
    
    video = st.file_uploader("Profile video (5–10 s, 720p recommended)", type=["mp4","mov","avi"])
    speed = st.radio("Playback speed", ["0.2×","0.5×","1×"], horizontal=True)
    rate = {"0.2×":0.2,"0.5×":0.5,"1×":1}[speed]

    if st.button("🚀 Start Analysis", key="btn_analyze"):
        if not video:
            st.warning("Please upload a video first.")
        else:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(video.read()); f.flush()
            cap = cv2.VideoCapture(f.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (OUT_W, OUT_H))

            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
            progress = st.progress(0)

            # Enhanced biomechanical tracking arrays
            knee_angles_left = []
            knee_angles_right = []
            hip_angles_left = []
            hip_angles_right = []
            ankle_angles_left = []
            ankle_angles_right = []
            trunk_angles = []
            arm_swing_left = []
            arm_swing_right = []
            com_y = []  # Center of mass vertical position
            left_ankle_y, right_ankle_y = [], []
            mid_snapshot = None

            for i in range(nframes):
                ok, frame = cap.read()
                if not ok: break
                frame = cv2.resize(frame, (OUT_W, OUT_H))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    L_hip,L_knee,L_ankle = p(lm,23,OUT_W,OUT_H),p(lm,25,OUT_W,OUT_H),p(lm,27,OUT_W,OUT_H)
                    R_hip,R_knee,R_ankle = p(lm,24,OUT_W,OUT_H),p(lm,26,OUT_W,OUT_H),p(lm,28,OUT_W,OUT_H)
                    aL = angle(L_hip,L_knee,L_ankle); aR = angle(R_hip,R_knee,R_ankle)
                    knee_angles_left.append(aL)
                    knee_angles_right.append(aR)

                    mp_drawing.draw_landmarks(
                        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=GREEN, thickness=-1, circle_radius=4),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=BLUE, thickness=2)
                    )

                    L_shoulder,L_elbow,L_wrist = p(lm,11,OUT_W,OUT_H),p(lm,13,OUT_W,OUT_H),p(lm,15,OUT_W,OUT_H)
                    R_shoulder,R_elbow,R_wrist = p(lm,12,OUT_W,OUT_H),p(lm,14,OUT_W,OUT_H),p(lm,16,OUT_W,OUT_H)
                    
                    # Hip angles for stride analysis
                    L_hip_angle = angle(L_shoulder, L_hip, L_knee)
                    R_hip_angle = angle(R_shoulder, R_hip, R_knee)
                    hip_angles_left.append(L_hip_angle)
                    hip_angles_right.append(R_hip_angle)
                    
                    # Ankle angles for strike pattern
                    L_ankle_angle = angle(L_knee, L_ankle, p(lm, 31, OUT_W, OUT_H))  # Left foot index
                    R_ankle_angle = angle(R_knee, R_ankle, p(lm, 32, OUT_W, OUT_H))  # Right foot index
                    ankle_angles_left.append(L_ankle_angle)
                    ankle_angles_right.append(R_ankle_angle)
                    
                    # Trunk lean (forward/backward)
                    mid_shoulder = ((L_shoulder[0]+R_shoulder[0])//2, (L_shoulder[1]+R_shoulder[1])//2)
                    mid_hip = ((L_hip[0]+R_hip[0])//2, (L_hip[1]+R_hip[1])//2)
                    trunk_angle = degrees(atan2(mid_shoulder[0]-mid_hip[0], mid_hip[1]-mid_shoulder[1]))
                    trunk_angles.append(trunk_angle)
                    
                    # Arm swing amplitude
                    arm_swing_left.append(L_wrist[1])
                    arm_swing_right.append(R_wrist[1])
                    
                    # Vertical position of center of mass (approximated by hip center)
                    com_y.append(mid_hip[1])
                    
                    L_hip_center = ((L_hip[0]+R_hip[0])//2, (L_hip[1]+R_hip[1])//2)
                    draw_label(frame, f"{int(angle(L_hip_center,L_shoulder,L_elbow))}", (L_shoulder[0]+8, L_shoulder[1]-8))
                    draw_label(frame, f"{int(angle(L_hip_center,R_shoulder,R_elbow))}", (R_shoulder[0]+8, R_shoulder[1]-8))
                    draw_label(frame, f"{int(angle(L_shoulder,L_elbow,L_wrist))}", (L_elbow[0]+8, L_elbow[1]-8))
                    draw_label(frame, f"{int(angle(R_shoulder,R_elbow,R_wrist))}", (R_elbow[0]+8, R_elbow[1]-8))
                    draw_label(frame, f"{int(angle(L_knee,L_hip,L_ankle))}", (L_hip[0]+8, L_hip[1]-8))
                    draw_label(frame, f"{int(angle(R_knee,R_hip,R_ankle))}", (R_hip[0]+8, R_hip[1]-8))
                    draw_label(frame, f"{int(aL)}", (L_knee[0]+8, L_knee[1]-8))
                    draw_label(frame, f"{int(aR)}", (R_knee[0]+8, R_knee[1]-8))
                    draw_label(frame, f"{int(angle(L_hip,L_ankle,L_knee))}", (L_ankle[0]+8, L_ankle[1]-8))
                    draw_label(frame, f"{int(angle(R_hip,R_ankle,R_knee))}", (R_ankle[0]+8, R_ankle[1]-8))

                    left_ankle_y.append(L_ankle[1]); right_ankle_y.append(R_ankle[1])
                    
                    if mid_snapshot is None and i >= nframes // 3:
                        body_x = (L_hip[0] + R_hip[0] + L_shoulder[0] + R_shoulder[0]) / 4
                        if OUT_W * 0.35 <= body_x <= OUT_W * 0.65:
                            mid_snapshot = frame.copy()

                writer.write(frame)
                progress.progress(int((i+1)/max(1,nframes)*100))

            cap.release(); writer.release()
            video_path = ffmpeg_optimize(out_path)

            def count_peaks(y):
                y = pd.Series(y).rolling(3, center=True).mean().dropna()
                return int(((y.shift(1) > y) & (y.shift(-1) > y)).sum())

            steps = max(count_peaks(left_ankle_y), count_peaks(right_ankle_y))
            duration_sec = nframes / float(fps or 30)
            cadence = int(steps / duration_sec * 60) if duration_sec > 0 else 0
            
            # Individual knee angles
            knee_left_mean = float(np.nanmean(knee_angles_left)) if knee_angles_left else 0
            knee_right_mean = float(np.nanmean(knee_angles_right)) if knee_angles_right else 0
            knee_mean = (knee_left_mean + knee_right_mean) / 2

            # Vertical oscillation
            osc = 0.0
            if len(com_y) > 5:
                com_smooth = pd.Series(com_y).rolling(5, center=True).mean().dropna()
                osc = float((com_smooth.max() - com_smooth.min()) / 4)
            
            # Ground contact time
            def contact_time(y):
                if len(y) < 3 or fps == 0: return 0
                thr = np.percentile(y, 35)
                contact_frames = (np.array(y) >= thr).sum()
                return int(contact_frames / fps * 1000 / max(1, (len(y)/steps if steps else 1)))
            ct_left = contact_time(left_ankle_y)
            ct_right = contact_time(right_ankle_y)
            ct = int((ct_left + ct_right) / 2)
            
            # Motion type: Aerial vs Grounded
            flight_ratio = 0.0
            if len(left_ankle_y) > 10 and steps > 0:
                # Estimate flight time based on ankle position patterns
                ankle_combined = (np.array(left_ankle_y) + np.array(right_ankle_y)) / 2
                flight_threshold = np.percentile(ankle_combined, 25)
                flight_frames = (ankle_combined < flight_threshold).sum()
                flight_ratio = flight_frames / len(ankle_combined)
            motion_type = "Aerial" if flight_ratio > 0.3 else "Grounded"
            
            # Strike pattern analysis
            strike_pattern = "Midfoot"
            if len(ankle_angles_left) > 0 and len(ankle_angles_right) > 0:
                avg_ankle_angle = (np.nanmean(ankle_angles_left) + np.nanmean(ankle_angles_right)) / 2
                if avg_ankle_angle > 100:
                    strike_pattern = "Heel"
                elif avg_ankle_angle < 85:
                    strike_pattern = "Forefoot"
                else:
                    strike_pattern = "Midfoot"
            
            # Trunk lean
            trunk_lean = float(np.nanmean(trunk_angles)) if trunk_angles else 0
            trunk_stability = float(np.nanstd(trunk_angles)) if trunk_angles else 0
            
            # Arm swing efficiency
            arm_swing_amplitude = 0.0
            if len(arm_swing_left) > 0 and len(arm_swing_right) > 0:
                left_amp = np.max(arm_swing_left) - np.min(arm_swing_left)
                right_amp = np.max(arm_swing_right) - np.min(arm_swing_right)
                arm_swing_amplitude = (left_amp + right_amp) / 2
            
            # Energy efficiency score (based on vertical oscillation and trunk stability)
            energy_score = 100
            energy_score -= min(30, osc * 3)  # Penalize high oscillation
            energy_score -= min(20, trunk_stability * 2)  # Penalize trunk instability
            energy_score -= min(20, abs(trunk_lean) * 2)  # Penalize excessive lean
            if cadence < 160:
                energy_score -= 10
            energy_score = max(0, min(100, energy_score))
            
            # Symmetry calculation
            knee_sym = 100 - abs(knee_left_mean - knee_right_mean)
            contact_sym = 100 - abs(ct_left - ct_right) / max(ct_left, ct_right, 1) * 100
            overall_sym = (knee_sym + contact_sym) / 2
            
            # Hip mobility
            hip_rom_left = np.max(hip_angles_left) - np.min(hip_angles_left) if hip_angles_left else 0
            hip_rom_right = np.max(hip_angles_right) - np.min(hip_angles_right) if hip_angles_right else 0
            hip_mobility = (hip_rom_left + hip_rom_right) / 2

            remarks=[]
            remarks.append("⚡ Efficient cadence." if cadence>=160 else "🌀 Cadence could be slightly increased.")
            remarks.append("✅ Good symmetry." if overall_sym > 95 else "⚖️ Slight asymmetry detected.")
            remarks.append("⬇️ Controlled oscillation." if osc<=9 else "⬆️ High vertical oscillation.")
            if strike_pattern == "Heel":
                remarks.append("🦶 Heel striker pattern detected.")
            elif strike_pattern == "Forefoot":
                remarks.append("👣 Forefoot striker pattern detected.")
            else:
                remarks.append("🦶 Midfoot strike pattern.")
            remarks.append(f"🎯 {motion_type} running style.")
            if trunk_lean > 5:
                remarks.append("↗️ Forward trunk lean detected.")
            elif trunk_lean < -5:
                remarks.append("↙️ Backward trunk lean detected.")
            else:
                remarks.append("✅ Good trunk posture.")
            if energy_score >= 80:
                remarks.append("⚡ High energy efficiency.")
            elif energy_score >= 60:
                remarks.append("🔋 Moderate energy efficiency.")
            else:
                remarks.append("🔋 Energy efficiency could be improved.")

            st.session_state["video_path"] = video_path
            st.session_state["bio"] = {
                "knee_mean": round(knee_mean, 1),
                "knee_left": round(knee_left_mean, 1),
                "knee_right": round(knee_right_mean, 1),
                "cadence": cadence,
                "osc": round(osc, 1),
                "sym": round(overall_sym, 1),
                "contact_time": ct,
                "contact_left": ct_left,
                "contact_right": ct_right,
                "motion_type": motion_type,
                "strike_pattern": strike_pattern,
                "trunk_lean": round(trunk_lean, 1),
                "trunk_stability": round(trunk_stability, 1),
                "arm_swing": round(arm_swing_amplitude, 1),
                "energy_score": round(energy_score, 1),
                "hip_mobility": round(hip_mobility, 1),
                "flight_ratio": round(flight_ratio * 100, 1)
            }
            st.session_state["snapshot"] = mid_snapshot
            st.session_state["remarks"] = remarks

            video_file_path = st.session_state["video_path"]
            if os.path.exists(video_file_path):
                file_size = os.path.getsize(video_file_path)
                if file_size > 50 * 1024 * 1024:
                    st.warning("⚠️ Large video detected.")
                    st.video(video_file_path)
                else:
                    with open(video_file_path, "rb") as vf:
                        embed_video(vf.read(), rate)
            else:
                st.error("Error: analyzed video not found.")
    
    if "bio" in st.session_state and "video_path" in st.session_state:
        st.success("✅ Analysis complete! Check the Results tab.")

# ---------------- RESULTATS ----------------
with tab_results:
    st.subheader("📊 Biomechanical Results")
    if "bio" not in st.session_state:
        st.info("No analysis available. Run an analysis in the previous tab.")
    else:
        b = st.session_state["bio"]
        def v(x,dec=1):
            try: return 0 if np.isnan(x) else round(float(x),dec)
            except: return x if isinstance(x, str) else 0
        
        # Overall Running Score
        overall_score = int(b.get('energy_score', 50))
        st.markdown(f"""
        <div style="text-align:center;margin:20px 0 30px 0">
            <div style="font-size:3rem;margin-bottom:10px">🏆</div>
            <div style="font-size:2.5rem;font-weight:800;color:#1f2937">{overall_score}/100</div>
            <div style="font-size:1rem;color:#6b7280;margin-top:5px">Your Running Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main KPIs
        st.markdown("### 🎯 Key Metrics")
        st.markdown(f"""
        <div class="kpi">
          <div>⏱ Cadence<br><b>{v(b['cadence'])}</b> steps/min</div>
          <div>⚡ Energy Score<br><b>{v(b['energy_score'])}</b>/100</div>
          <div>🎭 Motion Type<br><b>{b.get('motion_type', 'N/A')}</b></div>
          <div>👟 Strike Pattern<br><b>{b.get('strike_pattern', 'N/A')}</b></div>
          <div>⚖️ Symmetry<br><b>{v(b['sym'])}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Knee Analysis
        st.markdown("### 🦵 Knee Analysis")
        st.markdown(f"""
        <div class="kpi">
          <div>🦵 Left Knee<br><b>{v(b.get('knee_left', 0))}°</b></div>
          <div>🦵 Right Knee<br><b>{v(b.get('knee_right', 0))}°</b></div>
          <div>📊 Average<br><b>{v(b['knee_mean'])}°</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Ground Contact
        st.markdown("### 🦶 Ground Contact")
        st.markdown(f"""
        <div class="kpi">
          <div>🕐 Left Contact<br><b>{v(b.get('contact_left', 0), 0)}</b> ms</div>
          <div>🕐 Right Contact<br><b>{v(b.get('contact_right', 0), 0)}</b> ms</div>
          <div>📊 Average<br><b>{v(b['contact_time'], 0)}</b> ms</div>
          <div>✈️ Flight Ratio<br><b>{v(b.get('flight_ratio', 0))}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Body Mechanics
        st.markdown("### 🏃 Body Mechanics")
        st.markdown(f"""
        <div class="kpi">
          <div>🔄 Oscillation<br><b>{v(b['osc'])}</b></div>
          <div>📐 Trunk Lean<br><b>{v(b.get('trunk_lean', 0))}°</b></div>
          <div>🎯 Trunk Stability<br><b>{v(b.get('trunk_stability', 0))}</b></div>
          <div>💪 Arm Swing<br><b>{v(b.get('arm_swing', 0))}</b> px</div>
          <div>🦴 Hip Mobility<br><b>{v(b.get('hip_mobility', 0))}°</b></div>
        </div>
        """, unsafe_allow_html=True)

        if "remarks" in st.session_state:
            st.markdown("""
            <div style="background:#f9fafb;border-radius:12px;padding:12px 16px;margin:16px 0">
                <h4 style="color:#374151;margin:0 0 8px 0;font-size:1.1rem;font-weight:600">💡 Technical Remarks</h4>
                <div style="color:#4b5563;font-size:0.95rem;line-height:1.6">
            """, unsafe_allow_html=True)
            for r in st.session_state["remarks"]:
                st.markdown(f"<div style='margin:4px 0'>• {r}</div>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # PDF Download
        if st.session_state.get("bio"):
            pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            c = canvas.Canvas(pdf_path, pagesize=A4)
            w,h = A4; y = h-2*cm
            c.setFont("Helvetica-Bold",16); c.drawString(2*cm,y,"StrideMatch MotionLab Report"); y -= 1.2*cm
            
            # Key Metrics
            c.setFont("Helvetica-Bold",13); c.drawString(2*cm,y,"Key Metrics"); y -= 0.8*cm
            c.setFont("Helvetica",11)
            c.drawString(2*cm,y,f"Cadence: {b['cadence']} steps/min"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Energy Score: {b.get('energy_score', 0)}/100"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Motion Type: {b.get('motion_type', 'N/A')}"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Strike Pattern: {b.get('strike_pattern', 'N/A')}"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Symmetry: {b['sym']}%"); y -= 0.9*cm
            
            # Knee Analysis
            c.setFont("Helvetica-Bold",13); c.drawString(2*cm,y,"Knee Analysis"); y -= 0.8*cm
            c.setFont("Helvetica",11)
            c.drawString(2*cm,y,f"Left Knee: {b.get('knee_left', 0)} degrees"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Right Knee: {b.get('knee_right', 0)} degrees"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Average: {b['knee_mean']} degrees"); y -= 0.9*cm
            
            # Ground Contact
            c.setFont("Helvetica-Bold",13); c.drawString(2*cm,y,"Ground Contact"); y -= 0.8*cm
            c.setFont("Helvetica",11)
            c.drawString(2*cm,y,f"Left Contact: {b.get('contact_left', 0)} ms"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Right Contact: {b.get('contact_right', 0)} ms"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Average: {b['contact_time']} ms"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Flight Ratio: {b.get('flight_ratio', 0)}%"); y -= 0.9*cm
            
            # Body Mechanics
            c.setFont("Helvetica-Bold",13); c.drawString(2*cm,y,"Body Mechanics"); y -= 0.8*cm
            c.setFont("Helvetica",11)
            c.drawString(2*cm,y,f"Vertical Oscillation: {b['osc']}"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Trunk Lean: {b.get('trunk_lean', 0)} degrees"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Trunk Stability: {b.get('trunk_stability', 0)}"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Arm Swing Amplitude: {b.get('arm_swing', 0)} px"); y -= 0.6*cm
            c.drawString(2*cm,y,f"Hip Mobility: {b.get('hip_mobility', 0)} degrees"); y -= 0.9*cm
            
            # Technical Remarks
            c.setFont("Helvetica-Bold",13); c.drawString(2*cm,y,"Technical Remarks"); y -= 0.8*cm
            c.setFont("Helvetica",11)
            for r in st.session_state.get("remarks", []):
                # Remove emoji for PDF
                r_clean = r.encode('ascii', 'ignore').decode('ascii').strip()
                if r_clean:
                    c.drawString(2*cm,y,"- "+r_clean); y -= 0.6*cm
                    if y < 3*cm:
                        c.showPage(); y = h-2*cm; c.setFont("Helvetica",11)

            if st.session_state.get("snapshot") is not None:
                snap_bgr = st.session_state["snapshot"]
                tmp_png_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                cv2.imwrite(tmp_png_path, snap_bgr)
                img = ImageReader(tmp_png_path)
                iw, ih = img.getSize()
                scale = min((w-4*cm)/iw, (h/2)/ih)
                c.drawImage(img, 2*cm, max(3*cm, y-ih*scale-0.5*cm), iw*scale, ih*scale, preserveAspectRatio=True, mask='auto')

            c.showPage(); c.save()
            
            with open(pdf_path,"rb") as fp:
                pdf_data = fp.read()
            
            st.markdown("""
            <style>
            div[data-testid="stDownloadButton"] > button {
                background: linear-gradient(90deg,#4f46e5,#3b82f6) !important;
                color: white !important;
                border: none !important;
                border-radius: 9999px !important;
                padding: 0.7rem 1.35rem !important;
                box-shadow: 0 8px 20px rgba(59,130,246,.25) !important;
                font-weight: 700 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name="stridematch_report.pdf",
                mime="application/pdf",
                key="download_pdf_direct"
            )

# ---------------- RECOMMANDATIONS ----------------
with tab_reco:
    st.markdown("<h2 style='text-align:center'>🧠 Your Recommendations</h2>", unsafe_allow_html=True)
    
    current_surface = st.session_state.get("profile_surface", "road")
    current_pronation = st.session_state.get("profile_pronation", "neutral")
    current_preference = st.session_state.get("profile_preference", "comfort")
    
    if "bio" not in st.session_state:
        st.warning("⚠️ Please complete a biomechanical analysis first for personalized recommendations.")
        bio_data = {}
    else:
        bio_data = st.session_state["bio"]
    
    # Charger le catalogue
    cat = load_catalog_from_gsheet(GOOGLE_SHEET_URL)
    
    if cat.empty:
        st.error("❌ Catalog unavailable. Please check that the Google Sheet is publicly readable.")
    else:
        # IMPORTANT: Créer une COPIE du DataFrame pour ne pas modifier le cache
        cat_copy = cat.copy()
        
        # Calculer les scores pour chaque chaussure
        scores = []
        for idx, row in cat_copy.iterrows():
            score = score_shoe(row, current_surface, current_pronation, current_preference, bio_data)
            scores.append(score)
        
        cat_copy["match_score"] = scores
        
        # Trier par score décroissant et prendre le top 7
        cat_sorted = cat_copy.sort_values("match_score", ascending=False).head(7).reset_index(drop=True)
        
        runner_emoji = "🏃‍♂️" if st.session_state.get("profile_gender","Male")=="Male" else "🏃‍♀️"
        
        # Définir les couleurs en gradation du vert au rouge pour les 7 positions
        gradient_colors = [
            "#10b981",  # 1er - Vert vif
            "#34d399",  # 2ème - Vert clair
            "#a3e635",  # 3ème - Vert-jaune
            "#facc15",  # 4ème - Jaune
            "#fb923c",  # 5ème - Orange
            "#f87171",  # 6ème - Rouge clair
            "#ef4444",  # 7ème - Rouge
        ]
        
        for position, (idx, row) in enumerate(cat_sorted.iterrows()):
            brand = str(row.get("brand", "")).strip()
            model = str(row.get("model", "")).strip()
            pct = int(row.get("match_score", 0))
            terrain = str(row.get("terrain", "")).strip()
            drop = str(row.get("drop_mm", "")).strip()
            weight_g = str(row.get("weight_g", "")).strip()
            stack = str(row.get("stack_mm", row.get("stack_mm_total", ""))).strip()
            link1 = str(row.get("link 1", row.get("link1", "#"))).strip()
            link2 = str(row.get("link 2", row.get("link2", ""))).strip()
            image_url = str(row.get("image_url", "")).strip()
            cushioning = str(row.get("cushioning", "")).strip()
            stability = str(row.get("stability", "")).strip()
            notes = str(row.get("notes", "")).strip()

            # Couleur selon la POSITION dans le classement (pas le score absolu)
            bar_color = gradient_colors[min(position, len(gradient_colors)-1)]

            meta_parts = []
            if stability or cushioning:
                meta_parts.append(f"{runner_emoji} {stability} / {cushioning}")
            else:
                meta_parts.append(f"{runner_emoji}")
            if terrain: meta_parts.append(f"🌍 {terrain}")
            if stack: meta_parts.append(f"⛰ {stack} mm")
            if drop: meta_parts.append(f"⏬ {drop} mm")
            if weight_g: meta_parts.append(f"⚖️ {weight_g} g")
            meta = " • ".join(meta_parts)
            
            notes_html = ""
            if notes and notes != "nan":
                notes_list = [n.strip() for n in notes.split(",") if n.strip()]
                if notes_list:
                    notes_capsules = " ".join([f'<span style="display:inline-block;background:#f3f4f6;color:#374151;padding:4px 12px;border-radius:20px;font-size:0.85rem;margin:2px 4px 2px 0;white-space:nowrap">{n}</span>' for n in notes_list])
                    notes_html = f'<div style="margin-top:10px;margin-bottom:10px">{notes_capsules}</div>'
            
            # Image HTML - only if valid URL
            image_html = ""
            if image_url and image_url != "nan" and image_url != "" and image_url.startswith("http"):
                image_html = f'<div style="text-align:center;margin-bottom:12px"><img src="{image_url}" style="max-width:120px;max-height:80px;object-fit:contain;border-radius:8px" onerror="this.style.display=\'none\'" /></div>'
            
            # Buy buttons HTML with proper styling
            buy_buttons = []
            if link1 and link1 != "nan" and link1 != "#" and link1 != "" and link1.startswith("http"):
                buy_buttons.append(f'<a href="{link1}" target="_blank" style="display:inline-block;background:linear-gradient(90deg,#4f46e5,#3b82f6);color:white;padding:8px 16px;border-radius:20px;text-decoration:none;font-weight:600;margin:4px;font-size:0.9rem">🛒 Buy with Partner 1</a>')
            if link2 and link2 != "nan" and link2 != "" and link2 != "#" and link2.startswith("http"):
                buy_buttons.append(f'<a href="{link2}" target="_blank" style="display:inline-block;background:linear-gradient(90deg,#10b981,#059669);color:white;padding:8px 16px;border-radius:20px;text-decoration:none;font-weight:600;margin:4px;font-size:0.9rem">🛒 Buy with Partner 2</a>')
            
            if not buy_buttons:
                buy_links_html = ""
            else:
                buy_links_html = f'<div style="text-align:center;margin-top:12px">{"".join(buy_buttons)}</div>'

            st.markdown(f"""
            <div class="card">
              {image_html}
              <div style="display:flex;justify-content:center;align-items:center;margin-bottom:8px">
                <span class="badge">❤️ {pct}%</span>
              </div>
              <h4 style="margin:0 0 8px 0;text-align:center">👟 {brand} {model}</h4>
              <div class="progress"><span style="width:{pct}%;background:{bar_color}"></span></div>
              <p class="meta" style="margin:.6rem 0 0.2rem">{meta}</p>
              {notes_html}
              {buy_links_html}
            </div>
            """, unsafe_allow_html=True)

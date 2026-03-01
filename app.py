"""
app.py — Network Intrusion Detection System
============================================
Streamlit version of the IDS project.
Hosted on Streamlit Cloud at: https://share.streamlit.io

Pages:
  🏠 Login        → simple password gate
  📊 Dashboard    → dataset stats + charts
  🔍 Detection    → 3-step form → Random Forest prediction
  📋 History      → past detections stored in session
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the FIRST streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network IDS — Random Forest",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background */
  .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 100%); }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(0,255,100,0.04);
    border-right: 1px solid rgba(0,255,100,0.15);
  }

  /* Cards */
  .ids-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,255,100,0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
  }

  /* Attack badges */
  .badge-dos    { background:#ff505022; color:#ff5050; border:1px solid #ff5050; padding:4px 12px; border-radius:12px; font-weight:700; font-size:0.85rem; }
  .badge-normal { background:#00ff6422; color:#00ff64; border:1px solid #00ff64; padding:4px 12px; border-radius:12px; font-weight:700; font-size:0.85rem; }
  .badge-probe  { background:#ffc80022; color:#ffc800; border:1px solid #ffc800; padding:4px 12px; border-radius:12px; font-weight:700; font-size:0.85rem; }
  .badge-r2l    { background:#ff640022; color:#ff6400; border:1px solid #ff6400; padding:4px 12px; border-radius:12px; font-weight:700; font-size:0.85rem; }
  .badge-u2r    { background:#c800ff22; color:#c800ff; border:1px solid #c800ff; padding:4px 12px; border-radius:12px; font-weight:700; font-size:0.85rem; }

  /* Result box */
  .result-normal { background:#00ff6415; border:2px solid #00ff64; border-radius:16px; padding:24px; text-align:center; }
  .result-attack { background:#ff505015; border:2px solid #ff5050; border-radius:16px; padding:24px; text-align:center; }

  /* Step progress */
  .step-active   { background:#00ff6422; border:1px solid #00ff64; border-radius:8px; padding:8px 16px; color:#00ff64; font-weight:700; }
  .step-inactive { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:8px 16px; color:rgba(255,255,255,0.4); }
  .step-done     { background:#00b4d822; border:1px solid #00b4d8; border-radius:8px; padding:8px 16px; color:#00b4d8; font-weight:700; }

  h1, h2, h3 { color: #fff !important; }
  p, li       { color: rgba(255,255,255,0.75) !important; }
  label       { color: rgba(255,255,255,0.8) !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS  (must match training order exactly)
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # Step 1 — Basic connection
    ('duration',          'int',   0,    60000, 0,    'Length (seconds) of the connection'),
    ('protocol_type',     'int',   0,    2,     0,    'Protocol: 0=TCP  1=UDP  2=ICMP'),
    ('flag',              'int',   0,    10,    0,    'Connection status flag (encoded 0–10)'),
    ('src_bytes',         'int',   0,    999999,0,    'Bytes sent from source → destination'),
    ('dst_bytes',         'int',   0,    999999,0,    'Bytes sent from destination → source'),
    ('land',              'int',   0,    1,     0,    '1 if src/dst host & port are the same'),
    ('wrong_fragment',    'int',   0,    3,     0,    'Number of wrong fragments'),
    ('urgent',            'int',   0,    3,     0,    'Number of urgent packets'),
    ('hot',               'int',   0,    100,   0,    'Number of hot indicators in content'),
    ('num_failed_logins', 'int',   0,    5,     0,    'Number of failed login attempts'),
    ('logged_in',         'int',   0,    1,     0,    '1 if successfully logged in'),
    ('num_compromised',   'int',   0,    999,   0,    'Number of compromised conditions'),

    # Step 2 — System activity
    ('root_shell',        'int',   0,    1,     0,    '1 if a root shell was obtained'),
    ('su_attempted',      'int',   0,    1,     0,    '1 if su root command was attempted'),
    ('num_file_creations','int',   0,    100,   0,    'Number of file creation operations'),
    ('num_shells',        'int',   0,    5,     0,    'Number of shell prompts obtained'),
    ('num_access_files',  'int',   0,    9,     0,    'Ops on access control files'),
    ('is_guest_login',    'int',   0,    1,     0,    '1 if the login is a guest login'),
    ('count',             'int',   0,    512,   1,    'Connections to same host (2-sec window)'),
    ('srv_count',         'int',   0,    512,   1,    'Connections to same service (2-sec window)'),
    ('serror_rate',       'float', 0.0,  1.0,   0.0,  '% connections with SYN errors'),
    ('rerror_rate',       'float', 0.0,  1.0,   0.0,  '% connections with REJ errors'),
    ('same_srv_rate',     'float', 0.0,  1.0,   1.0,  '% connections to the same service'),
    ('diff_srv_rate',     'float', 0.0,  1.0,   0.0,  '% connections to different services'),

    # Step 3 — Destination host window
    ('srv_diff_host_rate',         'float', 0.0, 1.0, 0.0, '% connections to different hosts (same service)'),
    ('dst_host_count',             'int',   0,   255, 1,   'Connections to same dst host (100-conn window)'),
    ('dst_host_srv_count',         'int',   0,   255, 1,   'Connections to same dst service'),
    ('dst_host_diff_srv_rate',     'float', 0.0, 1.0, 0.0, '% connections to different services on dst host'),
    ('dst_host_same_src_port_rate','float', 0.0, 1.0, 0.0, '% connections using same src port to dst host'),
    ('dst_host_srv_diff_host_rate','float', 0.0, 1.0, 0.0, '% connections to different hosts for same service'),
]

FEATURE_NAMES = [f[0] for f in FEATURES]
STEP1 = FEATURES[:12]
STEP2 = FEATURES[12:24]
STEP3 = FEATURES[24:]

# Attack colours for charts
ATTACK_COLORS = {
    'dos':    '#ff5050',
    'normal': '#00ff64',
    'probe':  '#ffc800',
    'r2l':    '#ff6400',
    'u2r':    '#c800ff',
}


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached so it loads only once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    st.cache_resource caches the model object across all users/sessions.
    It loads once when the app starts, not on every page refresh.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    try:
        m = joblib.load(model_path)
        return m
    except Exception as e:
        st.error(f"❌ Could not load model.h5 — {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATASET STATS  (cached so CSV is read only once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset_stats():
    """Returns a dict of attack counts from dataset.csv."""
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    try:
        df = pd.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        counts = df['Attack Type'].value_counts().to_dict()
        return counts, len(df)
    except Exception:
        return {'dos': 391458, 'normal': 97278, 'probe': 4107, 'r2l': 1126, 'u2r': 52}, 494021


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
# st.session_state persists values across page interactions (like Django sessions)
def init_state():
    defaults = {
        'logged_in':   False,
        'username':    '',
        'page':        'login',       # login | dashboard | detect | history
        'detect_step': 1,             # 1 | 2 | 3
        'input_data':  {},            # accumulated feature values
        'history':     [],            # list of past detection dicts
        'last_result': None,          # most recent prediction string
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()
model = load_model()


# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def go(page, detect_step=1):
    st.session_state.page = page
    st.session_state.detect_step = detect_step
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡 Network IDS")
        st.markdown("**Random Forest Classifier**")
        st.markdown("---")

        if st.session_state.logged_in:
            st.markdown(f"👤 **{st.session_state.username}**")
            st.markdown("")

            if st.button("📊 Dashboard",  use_container_width=True): go('dashboard')
            if st.button("🔍 Detection",  use_container_width=True): go('detect', 1)
            if st.button("📋 History",    use_container_width=True): go('history')
            st.markdown("---")
            if st.button("🚪 Logout",     use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username  = ''
                go('login')

        st.markdown("---")
        st.markdown("""
        <small style='color:rgba(255,255,255,0.3)'>
        GEC Kushalnagar<br>
        4GL21CS048 — Sudeep K<br>
        Random Forest · KDD CUP 99
        </small>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:24px;'>
          <div style='font-size:3rem;'>🛡</div>
          <h1 style='color:#00ff64; letter-spacing:3px; margin:0;'>LOGIN</h1>
          <p style='color:rgba(255,255,255,0.4); font-size:0.85rem;'>
            Intrusion Detection System · Random Forest ML
          </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", placeholder="Enter password", type="password")
            submitted = st.form_submit_button("🔐 Sign In", use_container_width=True)

        if submitted:
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.session_state.username  = username
                go('dashboard')
            else:
                st.error("❌ Invalid credentials. Try admin / admin")

        st.markdown("""
        <p style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.75rem; margin-top:12px;'>
          Demo credentials: admin / admin
        </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    counts, total = load_dataset_stats()

    st.markdown("# 📊 Data Insights")
    st.markdown(f"**KDD CUP 99 Dataset** — {total:,} records · 5 attack categories · 30 features")
    st.markdown("---")

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    cols = st.columns(5)
    labels = [
        ('DoS',    'dos',    '#ff5050', '💥'),
        ('Normal', 'normal', '#00ff64', '✅'),
        ('Probe',  'probe',  '#ffc800', '🔎'),
        ('R2L',    'r2l',    '#ff6400', '🌐'),
        ('U2R',    'u2r',    '#c800ff', '👑'),
    ]
    for col, (name, key, color, icon) in zip(cols, labels):
        cnt = counts.get(key, 0)
        pct = cnt / total * 100
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid {color}33;
                        border-radius:12px; padding:16px; text-align:center;'>
              <div style='font-size:1.6rem;'>{icon}</div>
              <div style='color:{color}; font-weight:700; font-size:1rem;'>{name}</div>
              <div style='color:#fff; font-size:1.3rem; font-weight:700;'>{cnt:,}</div>
              <div style='color:rgba(255,255,255,0.4); font-size:0.75rem;'>{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Attack Type Distribution")
        fig = go.Figure(go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=[ATTACK_COLORS.get(k, '#888') for k in counts.keys()],
            text=[f"{v:,}" for v in counts.values()],
            textposition='outside',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='rgba(255,255,255,0.7)',
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Distribution (Pie)")
        fig2 = go.Figure(go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            marker_colors=[ATTACK_COLORS.get(k, '#888') for k in counts.keys()],
            hole=0.45,
            textinfo='label+percent',
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='rgba(255,255,255,0.8)',
            margin=dict(t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Model Scores ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("""
        <div style='text-align:center; background:rgba(0,255,100,0.05);
                    border:1px solid rgba(0,255,100,0.2); border-radius:12px; padding:20px;'>
          <div style='color:rgba(255,255,255,0.5); font-size:0.8rem;'>Training Accuracy</div>
          <div style='color:#00ff64; font-size:2.5rem; font-weight:700;'>99.996%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div style='text-align:center; background:rgba(0,180,216,0.05);
                    border:1px solid rgba(0,180,216,0.2); border-radius:12px; padding:20px;'>
          <div style='color:rgba(255,255,255,0.5); font-size:0.8rem;'>Testing Accuracy</div>
          <div style='color:#00b4d8; font-size:2.5rem; font-weight:700;'>99.977%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div style='text-align:center; background:rgba(255,200,0,0.05);
                    border:1px solid rgba(255,200,0,0.2); border-radius:12px; padding:20px;'>
          <div style='color:rgba(255,255,255,0.5); font-size:0.8rem;'>Decision Trees</div>
          <div style='color:#ffc800; font-size:2.5rem; font-weight:700;'>100</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance ─────────────────────────────────────────────────────
    st.markdown("#### 🏆 Top 10 Most Important Features")
    feat_importance = {
        'count': 0.2079, 'dst_bytes': 0.1387, 'logged_in': 0.1123,
        'dst_host_count': 0.0843, 'src_bytes': 0.0613, 'srv_count': 0.0594,
        'protocol_type': 0.0491, 'same_srv_rate': 0.0429,
        'diff_srv_rate': 0.0427, 'dst_host_srv_diff_host_rate': 0.0398,
    }
    fi_df = pd.DataFrame(list(feat_importance.items()), columns=['Feature', 'Importance'])
    fi_fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale=['#00b4d8', '#00ff64'])
    fi_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='rgba(255,255,255,0.7)',
        yaxis=dict(autorange='reversed', showgrid=False),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        coloraxis_showscale=False,
        margin=dict(t=10, b=10), height=320,
    )
    st.plotly_chart(fi_fig, use_container_width=True)

    st.markdown("---")
    if st.button("🚀 Start Detection →", use_container_width=False, type="primary"):
        st.session_state.input_data  = {}
        st.session_state.detect_step = 1
        go('detect', 1)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DETECTION (3 Steps)
# ══════════════════════════════════════════════════════════════════════════════
def render_step_header(current):
    """Draws the 3-step progress indicator."""
    steps = ['Step 1\nBasic Connection', 'Step 2\nSystem Activity', 'Step 3\nHost Window']
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current:
                cls = 'step-done';     icon = '✅'
            elif i == current:
                cls = 'step-active';   icon = '▶'
            else:
                cls = 'step-inactive'; icon = f'{i}'
            st.markdown(
                f"<div class='{cls}' style='text-align:center;'>"
                f"{icon} {label.replace(chr(10),' — ')}</div>",
                unsafe_allow_html=True
            )
    st.markdown("<br>", unsafe_allow_html=True)


def input_widget(feat_tuple):
    """
    Renders one Streamlit input widget per feature.
    Returns the value the user entered.
    """
    name, typ, mn, mx, default, hint = feat_tuple
    prev = st.session_state.input_data.get(name, default)

    if typ == 'float':
        val = st.number_input(
            f"**{name}**",
            min_value=float(mn), max_value=float(mx),
            value=float(prev), step=0.01,
            help=hint, key=f"inp_{name}"
        )
    else:
        val = st.number_input(
            f"**{name}**",
            min_value=int(mn), max_value=int(mx),
            value=int(prev), step=1,
            help=hint, key=f"inp_{name}"
        )
    return val


def page_detect():
    step = st.session_state.detect_step

    st.markdown("# 🔍 Network Intrusion Detection")
    st.markdown("Enter network connection features. The Random Forest model will classify the traffic.")
    st.markdown("---")

    render_step_header(step)

    # ── STEP 1 ─────────────────────────────────────────────────────────────────
    if step == 1:
        st.markdown("### 📡 Step 1 — Basic Connection Attributes")
        st.markdown("*Features 1–12: fundamental TCP/IP connection properties*")

        col1, col2 = st.columns(2)
        vals = {}
        for i, feat in enumerate(STEP1):
            with (col1 if i % 2 == 0 else col2):
                vals[feat[0]] = input_widget(feat)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next Step →", type="primary", use_container_width=False):
            st.session_state.input_data.update(vals)
            go('detect', 2)

    # ── STEP 2 ─────────────────────────────────────────────────────────────────
    elif step == 2:
        st.markdown("### 🖥 Step 2 — System Activity & Traffic Window")
        st.markdown("*Features 13–24: system-level events and 2-second connection window stats*")

        col1, col2 = st.columns(2)
        vals = {}
        for i, feat in enumerate(STEP2):
            with (col1 if i % 2 == 0 else col2):
                vals[feat[0]] = input_widget(feat)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("← Back", use_container_width=True):
                st.session_state.input_data.update(vals)
                go('detect', 1)
        with c2:
            if st.button("Next Step →", type="primary", use_container_width=False):
                st.session_state.input_data.update(vals)
                go('detect', 3)

    # ── STEP 3 ─────────────────────────────────────────────────────────────────
    elif step == 3:
        st.markdown("### 🌐 Step 3 — Destination Host Window Features")
        st.markdown("*Features 25–30: 100-connection destination host statistics*")

        col1, col2 = st.columns(2)
        vals = {}
        for i, feat in enumerate(STEP3):
            with (col1 if i % 2 == 0 else col2):
                vals[feat[0]] = input_widget(feat)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 5])

        with c1:
            if st.button("← Back", use_container_width=True):
                st.session_state.input_data.update(vals)
                go('detect', 2)

        with c2:
            detect_clicked = st.button("🔍 Detect Attack", type="primary", use_container_width=True)

        if detect_clicked:
            st.session_state.input_data.update(vals)

            # ── Run prediction ─────────────────────────────────────────────
            if model is None:
                st.error("Model not loaded. Upload model.h5 and restart.")
            else:
                # Build feature array in the exact training order
                arr = np.array([
                    float(st.session_state.input_data.get(n, 0))
                    for n in FEATURE_NAMES
                ], dtype=float).reshape(1, -1)

                prediction = model.predict(arr)[0]      # e.g. 'dos'
                proba      = model.predict_proba(arr)[0] # confidence per class
                confidence = max(proba) * 100

                # ── Show result ────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🎯 Detection Result")

                is_normal = (prediction == 'normal')
                color     = ATTACK_COLORS.get(prediction, '#888')
                icon      = '✅' if is_normal else '🚨'
                title     = 'Traffic is Normal' if is_normal else 'Attack Detected!'

                st.markdown(f"""
                <div class='{"result-normal" if is_normal else "result-attack"}'>
                  <div style='font-size:3rem;'>{icon}</div>
                  <div style='font-size:1.1rem; color:rgba(255,255,255,0.6);'>Detection Result</div>
                  <div style='font-size:2.5rem; font-weight:900; color:{color};'>
                    {prediction.upper()}
                  </div>
                  <div style='font-size:1rem; color:rgba(255,255,255,0.5); margin-top:8px;'>
                    {title} · Confidence: {confidence:.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar per class
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Confidence per Attack Class")
                classes = model.classes_
                conf_df = pd.DataFrame({
                    'Attack Type': classes,
                    'Confidence (%)': [p * 100 for p in proba]
                }).sort_values('Confidence (%)', ascending=False)

                conf_fig = px.bar(
                    conf_df, x='Attack Type', y='Confidence (%)',
                    color='Attack Type',
                    color_discrete_map={k: ATTACK_COLORS.get(k, '#888') for k in classes},
                    text=conf_df['Confidence (%)'].apply(lambda x: f"{x:.1f}%"),
                )
                conf_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='rgba(255,255,255,0.7)', showlegend=False,
                    xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    margin=dict(t=10, b=10), height=260,
                )
                st.plotly_chart(conf_fig, use_container_width=True)

                # Save to history
                st.session_state.history.append({
                    'time':       datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction': prediction,
                    'confidence': f"{confidence:.1f}%",
                    'inputs':     dict(st.session_state.input_data),
                })
                st.session_state.last_result = prediction

                # What does this attack mean?
                descriptions = {
                    'normal': "✅ **Normal Traffic** — No malicious activity detected. The network connection exhibits patterns consistent with legitimate use.",
                    'dos':    "💥 **Denial of Service (DoS)** — The traffic pattern suggests an attempt to overwhelm a server or service with excessive requests, making it unavailable to legitimate users.",
                    'probe':  "🔎 **Probe Attack** — Reconnaissance activity detected. An attacker may be scanning the network for vulnerabilities or open ports to exploit later.",
                    'r2l':    "🌐 **Remote to Local (R2L)** — An attacker from outside the network appears to be attempting unauthorised access to a local system, possibly via credential theft or exploitation.",
                    'u2r':    "👑 **User to Root (U2R)** — A local user appears to be attempting privilege escalation to gain root/administrator control over the system.",
                }
                st.info(descriptions.get(prediction, ''))

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Run Another Detection", use_container_width=False):
                    st.session_state.input_data  = {}
                    st.session_state.detect_step = 1
                    go('detect', 1)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("# 📋 Detection History")
    st.markdown("All detection runs from this session.")
    st.markdown("---")

    history = st.session_state.history

    if not history:
        st.info("No detections yet. Go to **Detection** to run your first analysis.")
        if st.button("🔍 Go to Detection"):
            go('detect', 1)
        return

    # Summary stats
    total = len(history)
    attack_counts = {}
    for h in history:
        p = h['prediction']
        attack_counts[p] = attack_counts.get(p, 0) + 1

    cols = st.columns(len(attack_counts) + 1)
    with cols[0]:
        st.metric("Total Runs", total)
    for col, (attack, cnt) in zip(cols[1:], attack_counts.items()):
        color = ATTACK_COLORS.get(attack, '#888')
        with col:
            st.markdown(f"""
            <div style='text-align:center; background:{color}11;
                        border:1px solid {color}55; border-radius:8px; padding:12px;'>
              <div style='color:{color}; font-weight:700;'>{attack.upper()}</div>
              <div style='color:#fff; font-size:1.5rem; font-weight:700;'>{cnt}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table of detections (newest first)
    for entry in reversed(history):
        color = ATTACK_COLORS.get(entry['prediction'], '#888')
        icon  = '✅' if entry['prediction'] == 'normal' else '🚨'
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); border-left:4px solid {color};
                    border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:8px;
                    display:flex; justify-content:space-between; align-items:center;'>
          <span style='color:rgba(255,255,255,0.5); font-size:0.85rem;'>{entry["time"]}</span>
          <span style='color:{color}; font-weight:700; font-size:1rem;'>
            {icon} {entry["prediction"].upper()}
          </span>
          <span style='color:rgba(255,255,255,0.4); font-size:0.8rem;'>
            Confidence: {entry["confidence"]}
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
render_sidebar()

if not st.session_state.logged_in:
    page_login()
else:
    page = st.session_state.page
    if page == 'dashboard': page_dashboard()
    elif page == 'detect':  page_detect()
    elif page == 'history': page_history()
    else:                   page_dashboard()

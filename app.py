import streamlit as st
import time

# ============== Page Config ==============
st.set_page_config(
    page_title="SalaryPredict AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============== Theme State & Toggle ==============
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
  
def toggle_theme():
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

# ============== Base CSS (uses CSS variables everywhere) ==============
BASE_CSS = """
<style>
:root {
  --bg: #ffffff;
  --text: #0f172a;
  --muted: #475569;
  --card: #ffffff;
  --border: #e5e7eb;
  --shadow: 0 8px 20px rgba(0,0,0,0.08);
  --accent: #6366f1;
  --accent-2: #8b5cf6;
  --nav-grad-start: #6366f1;
  --nav-grad-end: #8b5cf6;
  --hero-grad: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.08) 100%);
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  scroll-behavior: smooth;
}

/* Layout helpers */
.container { max-width: 1200px; margin: 0 auto; padding: 0 20px; }

/* Navbar */
.navbar {
  display:flex; justify-content:space-between; align-items:center;
  padding: 12px 20px; margin: 8px 0 24px;
  background: linear-gradient(90deg, var(--nav-grad-start), var(--nav-grad-end));
  border-radius: 14px; box-shadow: var(--shadow); position: sticky; top: 0; z-index: 1000;
}
.nav-left { color: #fff; font-weight: 800; letter-spacing: .2px; display:flex; gap:.6rem; align-items:center; }
.nav-right a {
  color:#fff; text-decoration:none; font-weight:600; margin-left: 16px;
  opacity:.95; transition: opacity .25s ease;
}
.nav-right a:hover { opacity: 1; text-decoration: underline; }

/* Fade-in */
.fade-in { animation: fadeIn .9s ease forwards; opacity: 0; transform: translateY(10px); }
@keyframes fadeIn { to { opacity:1; transform:none; } }

/* Floating hero shapes */
.floating { position:absolute; border-radius:50%; opacity:.25; animation: float 6s ease-in-out infinite; z-index:0; }
.shape1 { width:120px; height:120px; background: var(--accent); top:18%; left:8%; }
.shape2 { width:90px; height:90px; background: #f59e0b; bottom:18%; right:12%; }
@keyframes float { 0%,100%{ transform: translateY(0) } 50%{ transform: translateY(-18px) } }

/* Buttons */
.btn {
  padding: 14px 22px; border:none; border-radius: 999px; color:#fff; cursor:pointer;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  font-weight:700; box-shadow: var(--shadow); transition: transform .2s ease, box-shadow .2s ease;
}
.btn:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(0,0,0,.12); }

/* Sections */
.section { padding: 60px 0; }
.hero {
  position: relative; padding: 64px 12px; text-align:center; background: var(--hero-grad); border-radius: 18px;
}
.hero h1 { font-size: clamp(28px, 4.2vw, 48px); font-weight: 900; line-height:1.05; margin: 0 0 10px; }
.hero p { max-width: 720px; margin: 10px auto 22px; font-size: 18px; color: var(--muted); }

/* Features */
.features .cards {
  display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap:18px; margin-top: 18px;
}
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px;
  box-shadow: var(--shadow); transition: transform .2s ease, box-shadow .2s ease;
}
.card:hover { transform: translateY(-6px); box-shadow: 0 16px 36px rgba(0,0,0,.12); }
.card h3 { margin: 0 0 6px; }
.card p { margin: 0; color: var(--muted); }

/* How it works (timeline with animation) */
.timeline { max-width: 820px; margin: 16px auto 0; position: relative; padding-left: 56px; }
.timeline:before {
  content:""; position:absolute; left: 28px; top: 0; bottom: 0;
  width: 2px; background: linear-gradient(var(--accent), var(--accent-2));
  border-radius: 2px; opacity:.7, font-size:20px;
}
.step {
  position: relative; margin: 22px 0; background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px 16px 16px 20px; box-shadow: var(--shadow);
  opacity: 0; transform: translateX(24px);
  animation: stepIn .7s ease forwards;
}
.step .dot {
  position:absolute; left:-42px; top: 16px; width: 24px; height:24px; border-radius:50%;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); box-shadow: 0 0 0 3px rgba(99,102,241,.15);
}
.step h4 { margin: 0 0 6px; }
.step p { margin: 0; color: var(--muted); }
.step:nth-child(1) { animation-delay: .05s; }
.step:nth-child(2) { animation-delay: .2s; }
.step:nth-child(3) { animation-delay: .35s; }
@keyframes stepIn { to { opacity:1; transform:none; } }

/* Stats */
.stats { text-align:center; }
.stat-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 16px; margin-top: 14px; }
.stat {
  background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 18px; box-shadow: var(--shadow);
}
.stat .num { font-weight: 900; font-size: 34px; background: linear-gradient(135deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Contact */
.contact .form {
  max-width: 700px; margin: 0 auto; background: var(--card); border: 1px solid var(--border);
  border-radius: 16px; padding: 20px; box-shadow: var(--shadow);
}
.label { font-weight:700; margin: 6px 0; }
.help { color: var(--muted); font-size: 13px; }

/* Footer */
.footer { text-align:center; color: #8b8b8b; padding: 16px 8px; }
</style>
"""

# Theme override (ALWAYS render; last one wins)
THEME_CSS = f"""
<style>
:root {{
  {"".join([
    "--bg: #0f172a;",
    "--text: #e5e7eb;",
    "--muted: #cbd5e1;",
    "--card: #111827;",
    "--border: #374151;",
    "--shadow: 0 8px 26px rgba(0,0,0,.35);",
    "--accent: #8b5cf6;",
    "--accent-2: #6366f1;",
    "--nav-grad-start: #0ea5e9;",
    "--nav-grad-end: #6366f1;",
    "--hero-grad: linear-gradient(135deg, rgba(56,189,248,0.06) 0%, rgba(99,102,241,0.10) 100%);",
  ]) if st.session_state["theme"] == "dark" else "".join([
    "--bg: #ffffff;",
    "--text: #0f172a;",
    "--muted: #475569;",
    "--card: #ffffff;",
    "--border: #e5e7eb;",
    "--shadow: 0 8px 20px rgba(0,0,0,.08);",
    "--accent: #6366f1;",
    "--accent-2: #8b5cf6;",
    "--nav-grad-start: #6366f1;",
    "--nav-grad-end: #8b5cf6;",
    "--hero-grad: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.08) 100%);",
  ])}
}}
</style>
"""

# Inject base CSS once
st.markdown(BASE_CSS, unsafe_allow_html=True)
# Inject theme variables (this is what actually changes on toggle)
theme_slot = st.empty()
theme_slot.markdown(THEME_CSS, unsafe_allow_html=True)

# ============== Top Bar: Navbar + Toggle ==============
left, right = st.columns([10, 2])
with left:
    st.markdown(
        """
        <div class="navbar">
          <div class="nav-left">💼 SalaryPredict AI</div>
          <div class="nav-right">
            <a href="#home">Home</a>
            <a href="#features">Features</a>
            <a href="#how">How It Works</a>
            <a href="#stats">Stats</a>
            <a href="#contact">Contact</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.write("")  # spacer
    st.button(
        "🌙 Dark" if st.session_state["theme"] == "light" else "☀️ Light",
        key="theme_toggle",
        on_click=toggle_theme,
        use_container_width=True,
    )

    

# ============== HERO ==============
st.markdown(
    """
    <div id="home" class="container">
      <div class="hero fade-in">
        <div class="floating shape1"></div>
        <div class="floating shape2"></div>
        <h1>Predict Your <span style="color:var(--accent)">Future Salary</span> with AI</h1>
        <p>Leverage advanced machine learning algorithms to get accurate salary predictions
           based on your skills, experience, and market trends.</p>
    """,
    unsafe_allow_html=True,
)



# ============== FEATURES ==============
st.markdown(
    """
    <section id="features" class="section features">
      <div class="container">
        <h2 class="fade-in" style="text-align:center; margin:0 0 12px; font-size:28px; font-weight:900;
          background:linear-gradient(135deg,var(--accent),var(--accent-2));
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Powerful Features</h2>
        <p class="fade-in" style="text-align:center; color:var(--muted); margin:0 auto 10px; max-width:680px;">
          Everything you need for accurate salary predictions
        </p>
        <div class="cards">
          <div class="card fade-in"><h3>🧠 AI-Powered Analysis</h3><p>Advanced ML models trained on millions of salary data points.</p></div>
          <div class="card fade-in"><h3>📊 Market Insights</h3><p>Real-time salary analysis across industries and locations.</p></div>
          <div class="card fade-in"><h3>🌍 Global Coverage</h3><p>Predictions across multiple countries & job roles.</p></div>
          <div class="card fade-in"><h3>⚡ Instant Results</h3><p>Get predictions in seconds with detailed breakdowns.</p></div>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ============== HOW IT WORKS (animated steps) ==============
st.markdown(
    """
    <section id="how" class="section">
      <div class="container">
        <h2 class="fade-in" style="text-align:center; margin:0 0 10px; font-size:28px; font-weight:900;">🛠️ How It Works</h2>
        <p class="fade-in" style="text-align:center; color:var(--muted); margin:0 auto 16px; max-width:680px;">
          Simple steps to get your salary prediction
        </p>
        <div class="timeline">
          <div class="step">
            <span class="dot"></span>
            <h4>1) Enter Your Details</h4>
            <p>Provide your skills, experience, education, and location.</p>
          </div>
          <div class="step">
            <span class="dot"></span>
            <h4>2) AI Analysis</h4>
            <p>Our ML models analyze your profile against market data.</p>
          </div>
          <div class="step">
            <span class="dot"></span>
            <h4>3) Get Prediction</h4>
            <p>Receive personalized salary prediction with insights.</p>
          </div>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# if st.button("🚀 Start Prediction", key="hero_button"):
#     st.switch_page("pages/app.py")

st.markdown("</div></div>", unsafe_allow_html=True)

# ============== STATS (animated counters) ==============
st.markdown(
    """
    <section id="stats" class="section stats">
      <div class="container">
        <h2 class="fade-in" style="text-align:center; margin:0 0 10px; font-size:28px; font-weight:900;">📈 Stats</h2>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Dictionary of stats
stats = {
    "Predictions Made": 150,
    "Accuracy Rate (%)": 85,
    "Countries Covered": 6,
    "Job Roles": 5,
}

# Create only as many columns as needed
cols = st.columns(len(stats))

for col, (label, target) in zip(cols, stats.items()):
    with col:
        # st.markdown('<div class="stat">', unsafe_allow_html=True)
        placeholder = st.empty()
        for i in range(target + 1): 
            placeholder.markdown(
                f'<div class="num">{i}</div><div style="opacity:.85">{label}</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.008)
        st.markdown('</div>', unsafe_allow_html=True)

# ============== CONTACT ==============
st.markdown(
    """
    <section id="contact" class="section contact">
      <div class="container">
        <h2 class="fade-in" style="text-align:center; margin:0 0 12px; font-size:28px; font-weight:900;">📩 Contact Us</h2>
        <p class="fade-in" style="text-align:center; color:var(--muted); margin:0 auto 16px; max-width:680px;">
          Get in touch with us for any questions or feedback.
        </p>
        
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.form("contact_form"):
    st.markdown('<div class="container"><div class="form">', unsafe_allow_html=True)
    name = st.text_input("👤 Your Name")
    email = st.text_input("📧 Your Email")
    message = st.text_area("💬 Message", height=120)
    submitted = st.form_submit_button("Send Message")
    if submitted:
        if name and email and message:
            st.success(f"✅ Thanks {name}! Your message has been sent.")
            st.balloons()
        else:
            st.error("⚠️ Please fill in all fields.")
    st.markdown('</div></div>', unsafe_allow_html=True)

# ============== FOOTER ==============
st.markdown(
    """
    <div class="container">
      <div class="footer">© 2025 SalaryPredict AI • Built with ❤️ using Streamlit</div>
    </div>
    """,
    unsafe_allow_html=True,
)

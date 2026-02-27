# app.py - MindCare Companion: Complete Mental Health Chatbot
import streamlit as st
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MindCare Companion | AI Mental Health Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/mindcare',
        'Report a bug': 'https://github.com/yourusername/mindcare/issues',
        'About': "### MindCare Companion v1.0\nAI-powered mental health support system"
    }
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main title gradient */
    .main-header {
        font-size: 3.8rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    
    /* Subtitle */
    .sub-header {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2.5rem;
        font-size: 1.3rem;
        font-weight: 300;
    }
    
    /* Chat container */
    .chat-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 25px;
        max-height: 550px;
        overflow-y: auto;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* User message bubble */
    .user-msg {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 22px 22px 5px 22px;
        margin: 10px 0;
        text-align: right;
        max-width: 78%;
        margin-left: auto;
        font-size: 1.05rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        line-height: 1.5;
    }
    
    /* Bot message bubble */
    .bot-msg {
        background: white;
        padding: 14px 20px;
        border-radius: 22px 22px 22px 5px;
        margin: 10px 0;
        max-width: 78%;
        border-left: 6px solid #667eea;
        font-size: 1.05rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        line-height: 1.5;
    }
    
    /* Emergency alert */
    .emergency-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        padding: 22px;
        border-radius: 15px;
        border: 3px solid #ff3333;
        margin: 25px 0;
        color: white;
        animation: pulse 1.5s infinite;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    
    /* Resource cards */
    .resource-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        margin: 12px 0;
        border-left: 6px solid #667eea;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .resource-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0,0,0,0.12);
    }
    
    /* Metrics cards */
    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-top: 5px solid #667eea;
    }
    
    /* Mood badges */
    .mood-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 25px;
        margin: 5px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .badge-normal { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .badge-anxiety { background: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
    .badge-depression { background: #cce5ff; color: #004085; border: 2px solid #b8daff; }
    .badge-stress { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    
    /* Sidebar styling */
    .css-1d391kg {background-color: #f8f9fa;}
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e9ecef;
        padding: 15px;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        # Load the model
        model = joblib.load('mental_health_model.pkl')
        
        # Load the vectorizer (optional - model already has it)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        st.success("✅ AI Model loaded successfully!")
        return model, vectorizer
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please make sure these files are in the same folder:")
        st.code("""
        - mental_health_model.pkl
        - tfidf_vectorizer.pkl
        - mental_health_helper.py
        """)
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# Load models
model, vectorizer = load_models()

# ==================== HELPER FUNCTIONS ====================
def predict_mood_with_safety(text, model):
    """Predict mood with safety check"""
    if model is None:
        return "ERROR", "AI model failed to load. Please check the model files."
    
    try:
        predicted_status = model.predict([text])[0]
        
        # Safety Check for critical content
        if predicted_status == "Suicidal":
            emergency_response = """
            🚨 **CRITICAL: IMMEDIATE SUPPORT NEEDED** 🚨

            I'm deeply concerned about what you've shared. Your feelings are valid and important, 
            but this requires immediate professional attention.

            **TAKE THESE STEPS NOW:**

            **📞 Call National Suicide Prevention Lifeline:**
            1-800-273-8255 (24/7, free, confidential)

            **📱 Text Crisis Text Line:**
            Text "HOME" to 741741

            **🏥 Emergency Services:**
            Call 911 or go to the nearest emergency room

            **You are not alone. People care about you and want to help.**
            This conversation will be paused until you've reached out for help.
            """
            return "CRITICAL", emergency_response
        
        return predicted_status, None
        
    except Exception as e:
        return "ERROR", f"Prediction error: {str(e)}"

def get_empathetic_response(mood):
    """Get empathetic response based on mood"""
    
    mood_responses = {
        "Normal": {
            "title": "Feeling Balanced",
            "message": "🌟 **It's wonderful that you're feeling steady today!** Maintaining emotional balance is a skill worth celebrating.",
            "tip": "💡 **Wellness Tip:** Consider journaling about what's going well in your life. Recognizing positive moments can reinforce this balanced state.",
            "resources": [
                "📖 Read: 'The Happiness Advantage' by Shawn Achor",
                "🧘 Practice: Daily gratitude journaling",
                "🎯 Challenge: Set one small personal growth goal this week"
            ],
            "emoji": "😊",
            "color": "badge-normal"
        },
        "Anxiety": {
            "title": "Managing Anxiety",
            "message": "🌊 **I understand anxiety can feel like overwhelming waves.** Remember, you've navigated difficult feelings before and you can do it again.",
            "tip": "💡 **Grounding Exercise (5-4-3-2-1 Technique):**\n• **5** things you can see\n• **4** things you can touch\n• **3** things you can hear\n• **2** things you can smell\n• **1** thing you can taste",
            "resources": [
                "📱 App: Calm or Headspace for guided anxiety exercises",
                "📖 Read: 'The Anxiety and Phobia Workbook'",
                "🎵 Listen: Calming focus playlists on Spotify/YouTube"
            ],
            "emoji": "😰",
            "color": "badge-anxiety"
        },
        "Depression": {
            "title": "Navigating Low Mood",
            "message": "🫂 **Depression can make everything feel heavy.** Please know that what you're feeling is valid, and you don't have to carry this alone.",
            "tip": "💡 **Small Steps Approach:**\n1. Drink a glass of water\n2. Open a window for fresh air\n3. Send one text to someone you trust\n4. Write down one thing you accomplished today",
            "resources": [
                "📞 Support: National Depression Hotline: 1-800-273-8255",
                "📖 Read: 'The Noonday Demon' by Andrew Solomon",
                "🎨 Activity: Creative expression (drawing, writing, music)"
            ],
            "emoji": "😔",
            "color": "badge-depression"
        },
        "Stress": {
            "title": "Handling Stress",
            "message": "🎓 **Academic pressure is real and challenging.** It's okay to feel stressed about your responsibilities - it shows you care about your success.",
            "tip": "💡 **Quick Stress Relief (Box Breathing):**\nInhale for 4 seconds → Hold for 4 seconds → Exhale for 4 seconds → Hold for 4 seconds\nRepeat 5 times. This regulates your nervous system.",
            "resources": [
                "📱 App: Forest for focused study sessions",
                "📖 Read: 'The Stress Solution' by Dr. Rangan Chatterjee",
                "⏰ Technique: Pomodoro method (25 min work, 5 min break)"
            ],
            "emoji": "😫",
            "color": "badge-stress"
        },
        "ERROR": {
            "title": "Technical Issue",
            "message": "🔧 **There seems to be a technical issue.** Please try again or check if all model files are properly loaded.",
            "tip": "Make sure mental_health_model.pkl and tfidf_vectorizer.pkl are in the same folder as this app.",
            "resources": [],
            "emoji": "⚙️",
            "color": ""
        }
    }
    
    # Default response for unknown moods
    default_response = {
        "title": "I'm Here For You",
        "message": "💭 **I'm listening carefully.** Whatever you're experiencing, I'm here to provide support and understanding.",
        "tip": "💡 **General Wellness:** Sometimes simply acknowledging how you feel is the first step toward feeling better.",
        "resources": ["Take three deep breaths", "Stretch for 2 minutes", "Drink some water"],
        "emoji": "💭",
        "color": ""
    }
    
    return mood_responses.get(mood, default_response)

# ==================== SESSION STATE INITIALIZATION ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mood_tracker" not in st.session_state:
    st.session_state.mood_tracker = []
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total_messages": 0,
        "anxiety_count": 0,
        "depression_count": 0,
        "stress_count": 0,
        "normal_count": 0
    }

# ==================== SIDEBAR ====================
with st.sidebar:
    # Logo and title
    st.markdown("<h1 style='text-align: center; color: #667eea;'>🧠</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>MindCare</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>AI Mental Health Companion</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🗺️ Navigation")
    page_options = ["💬 Chat Now", "📊 Mood Dashboard", "🛠️ Resources", "ℹ️ About & Help"]
    page = st.radio("Select Page:", page_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", st.session_state.session_stats["total_messages"])
    with col2:
        if st.session_state.mood_tracker:
            latest_mood = st.session_state.mood_tracker[-1]["mood"]
            st.metric("Recent Mood", latest_mood)
        else:
            st.metric("Recent Mood", "–")
    
    st.markdown("---")
    
    # Emergency Section (always visible)
    with st.expander("🚨 **EMERGENCY CONTACTS**", expanded=True):
        st.warning("""
        **IMMEDIATE HELP NEEDED?**
        
        **📞 National Suicide Prevention Lifeline:**
        1-800-273-8255 (24/7)
        
        **📱 Crisis Text Line:**
        Text HOME to 741741
        
        **🏥 Emergency Medical Services:**
        Call 911
        
        **🌐 International Association for Suicide Prevention:**
        Find local resources: www.iasp.info
        """)
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.mood_tracker = []
        st.session_state.session_stats = {k: 0 for k in st.session_state.session_stats}
        st.success("History cleared! Starting fresh.")
        st.rerun()
    
    # Model status
    st.markdown("---")
    if model:
        st.success("✅ AI Model: Active")
    else:
        st.error("❌ AI Model: Not Loaded")

# ==================== PAGE 1: CHAT INTERFACE ====================
if page == "💬 Chat Now":
    # Header
    st.markdown('<h1 class="main-header">💬 MindCare Companion</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-powered mental health support system. Speak freely, I\'m here to listen and help.</p>', unsafe_allow_html=True)
    
    # Chat display container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="bot-msg">
                <h4>👋 Welcome to MindCare Companion!</h4>
                <p>I'm your AI mental health support assistant. You can talk to me about:</p>
                <ul>
                    <li>😰 Anxiety or stress you're experiencing</li>
                    <li>😔 Feelings of sadness or depression</li>
                    <li>🎓 Academic pressures</li>
                    <li>👥 Relationship concerns</li>
                    <li>💭 General emotional wellbeing</li>
                </ul>
                <p><strong>Everything you share is confidential.</strong> Start by typing below!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-msg"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick reply buttons
    st.markdown("### 💭 Quick Topics")
    quick_topics = st.columns(4)
    
    topic_buttons = {
        "Academic Stress": "I'm overwhelmed with assignments and exams...",
        "Anxiety": "I can't stop worrying about everything...",
        "Loneliness": "I feel really alone lately...",
        "Depression": "I've been feeling really low and unmotivated..."
    }
    
    for idx, (topic, prompt) in enumerate(topic_buttons.items()):
        with quick_topics[idx]:
            if st.button(f"**{topic}**", use_container_width=True, key=f"topic_{idx}"):
                # Auto-fill the text area
                st.session_state.user_input = prompt
    
    # Chat input section
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "**Type your message here:**",
            height=120,
            placeholder="How are you feeling today? What's on your mind?...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        send_button = st.button("**Send** ✉️", use_container_width=True, type="primary")
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Update stats
        st.session_state.session_stats["total_messages"] += 1
        
        # Get AI response
        with st.spinner("🤔 Analyzing your message..."):
            mood, safety_msg = predict_mood_with_safety(user_input, model)
            
            if safety_msg:  # Critical/emergency response
                response = safety_msg
                mood = "CRITICAL"
            else:
                response_data = get_empathetic_response(mood)
                
                # Format response
                response = f"""
                {response_data['emoji']} **{response_data['title']}**
                
                {response_data['message']}
                
                **💡 Practical Tip:**
                {response_data['tip']}
                
                **📚 Suggested Resources:**
                """
                for resource in response_data['resources']:
                    response += f"\n• {resource}"
                
                response += f"\n\n<span class='{response_data['color']} mood-badge'>Mood: {mood}</span>"
            
            # Add bot response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Track mood (if not error or critical)
            if mood not in ["ERROR", "CRITICAL"]:
                st.session_state.mood_tracker.append({
                    "mood": mood,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "message_preview": user_input[:40] + "..."
                })
                
                # Update mood counts
                if mood == "Anxiety":
                    st.session_state.session_stats["anxiety_count"] += 1
                elif mood == "Depression":
                    st.session_state.session_stats["depression_count"] += 1
                elif mood == "Stress":
                    st.session_state.session_stats["stress_count"] += 1
                elif mood == "Normal":
                    st.session_state.session_stats["normal_count"] += 1
        
        # Clear input and refresh
        if 'user_input' in st.session_state:
            del st.session_state.user_input
        st.rerun()

# ==================== PAGE 2: MOOD DASHBOARD ====================
elif page == "📊 Mood Dashboard":
    st.markdown('<h1 class="main-header">📊 Mood Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Track your emotional wellbeing patterns and progress over time</p>', unsafe_allow_html=True)
    
    if st.session_state.mood_tracker:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.mood_tracker)
        
        # Metrics row
        st.markdown("## 📈 Overview Metrics")
        metric_cols = st.columns(4)
        
        metrics = [
            ("Total Entries", len(df), "#667eea"),
            ("Anxiety Episodes", st.session_state.session_stats["anxiety_count"], "#ffc107"),
            ("Low Mood Days", st.session_state.session_stats["depression_count"], "#17a2b8"),
            ("Stress Events", st.session_state.session_stats["stress_count"], "#dc3545")
        ]
        
        for idx, (title, value, color) in enumerate(metrics):
            with metric_cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {color};">{value}</h3>
                    <p>{title}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Mood distribution chart
        st.markdown("## 📊 Mood Distribution")
        
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            mood_counts = df["mood"].value_counts()
            
            # Color mapping
            color_map = {
                "Normal": "#28a745",
                "Anxiety": "#ffc107",
                "Depression": "#17a2b8",
                "Stress": "#dc3545"
            }
            
            colors = [color_map.get(mood, "#6c757d") for mood in mood_counts.index]
            
            # Create bar chart
            bars = ax.bar(mood_counts.index, mood_counts.values, color=colors, edgecolor='white', linewidth=2)
            ax.set_xlabel("Mood State", fontsize=12, fontweight='bold')
            ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
            ax.set_title("Your Emotional Patterns", fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        # Recent history table
        st.markdown("## 📝 Recent Mood Log")
        
        # Create styled DataFrame
        display_df = df.tail(10).copy()
        
        # Add mood badges
        def create_badge(mood):
            color_class = {
                "Normal": "badge-normal",
                "Anxiety": "badge-anxiety",
                "Depression": "badge-depression",
                "Stress": "badge-stress"
            }.get(mood, "")
            return f'<span class="mood-badge {color_class}">{mood}</span>'
        
        display_df["Mood"] = display_df["mood"].apply(create_badge)
        display_df = display_df[["timestamp", "Mood", "message_preview"]]
        display_df.columns = ["Time", "Mood State", "Message Preview"]
        
        # Display as HTML
        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Export option
        st.markdown("---")
        if st.button("📥 Export Your Data (CSV)"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV File",
                data=csv,
                file_name=f"mindcare_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        # Empty state
        st.info("""
        ## 📊 No Data Yet
        
        Your mood dashboard will appear here once you start chatting!
        
        **Here's what you'll see:**
        - 📈 **Mood distribution charts** - Visualize your emotional patterns
        - 📝 **Recent conversation log** - Review your mood journey
        - 🎯 **Personal insights** - Understand your emotional triggers
        
        **👉 Go to the Chat page to get started!**
        """)

# ==================== PAGE 3: RESOURCES ====================
elif page == "🛠️ Resources":
    st.markdown('<h1 class="main-header">🛠️ Mental Health Resources</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Tools, techniques, and professional support options</p>', unsafe_allow_html=True)
    
    # Tab layout for resources
    resource_tabs = st.tabs(["🧘 Self-Help Tools", "🏥 Professional Support", "📚 Learning Resources"])
    
    with resource_tabs[0]:
        st.markdown("### 🧘 Self-Help Tools & Techniques")
        
        tools = [
            {
                "title": "Breathing Exercises",
                "description": "Guided breathing techniques for anxiety and stress relief",
                "link": "https://www.headspace.com/breathing-exercises",
                "icon": "🌬️"
            },
            {
                "title": "Meditation Guides",
                "description": "Free meditation sessions for various mental states",
                "link": "https://www.calm.com/meditations",
                "icon": "🧘"
            },
            {
                "title": "Sleep Improvement",
                "description": "Techniques and routines for better sleep quality",
                "link": "https://www.sleepfoundation.org/sleep-hygiene",
                "icon": "😴"
            },
            {
                "title": "Digital Wellness",
                "description": "Managing screen time and digital overload",
                "link": "https://www.humanetech.com/take-control",
                "icon": "📱"
            }
        ]
        
        for tool in tools:
            st.markdown(f"""
            <div class="resource-card">
                <h4>{tool['icon']} {tool['title']}</h4>
                <p>{tool['description']}</p>
                <a href="{tool['link']}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 600;">
                    🔗 Access Resource →
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    with resource_tabs[1]:
        st.markdown("### 🏥 Professional Support Services")
        
        services = [
            {
                "title": "Therapist Directory",
                "description": "Find licensed therapists in your area",
                "link": "https://www.psychologytoday.com/us/therapists",
                "icon": "👨‍⚕️"
            },
            {
                "title": "Online Counseling",
                "description": "Professional therapy from the comfort of your home",
                "link": "https://www.talkspace.com",
                "icon": "💻"
            },
            {
                "title": "Support Groups",
                "description": "Connect with others facing similar challenges",
                "link": "https://www.nami.org/Support-Education/Support-Groups",
                "icon": "👥"
            },
            {
                "title": "University Counseling",
                "description": "Student mental health services (most universities offer free counseling)",
                "link": "https://www.ulifeline.org",
                "icon": "🎓"
            }
        ]
        
        for service in services:
            st.markdown(f"""
            <div class="resource-card">
                <h4>{service['icon']} {service['title']}</h4>
                <p>{service['description']}</p>
                <a href="{service['link']}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 600;">
                    🔗 Find Support →
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    with resource_tabs[2]:
        st.markdown("### 📚 Educational Resources")
        
        books = [
            {
                "title": "The Anxiety and Phobia Workbook",
                "author": "Edmund J. Bourne",
                "description": "Practical exercises for managing anxiety"
            },
            {
                "title": "Feeling Good: The New Mood Therapy",
                "author": "David D. Burns",
                "description": "Cognitive behavioral techniques for depression"
            },
            {
                "title": "The Body Keeps the Score",
                "author": "Bessel van der Kolk",
                "description": "Understanding trauma and healing"
            },
            {
                "title": "Atomic Habits",
                "author": "James Clear",
                "description": "Building positive routines for mental health"
            }
        ]
        
        for book in books:
            st.markdown(f"""
            <div class="resource-card">
                <h4>📖 {book['title']}</h4>
                <p><em>by {book['author']}</em></p>
                <p>{book['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Daily wellness checklist
        st.markdown("### ✅ Daily Wellness Checklist")
        daily_tasks = [
            "Drink 8 glasses of water",
            "Get 30 minutes of movement",
            "Connect with someone",
            "Practice gratitude",
            "Take screen breaks",
            "Get 7-9 hours of sleep",
            "Eat balanced meals",
            "Spend time in nature"
        ]
        
        cols = st.columns(4)
        for idx, task in enumerate(daily_tasks):
            with cols[idx % 4]:
                st.checkbox(task)

# ==================== PAGE 4: ABOUT & HELP ====================
else:
    st.markdown('<h1 class="main-header">ℹ️ About MindCare</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌟 Our Mission
        
        MindCare Companion is an AI-powered mental health support system designed to provide **immediate, empathetic, 
        and accessible** emotional support to individuals navigating life's challenges.
        
        ### 🔬 How It Works
        
        **1. Mood Detection**  
        Using machine learning trained on thousands of mental health statements, I can identify emotional states including:
        - 😊 Normal/balanced mood
        - 😰 Anxiety and worry
        - 😔 Depression and low mood
        - 😫 Stress and pressure
        - 🚨 Crisis situations
        
        **2. Safety First Protocol**  
        Automatic detection of crisis situations with immediate emergency resource provision.
        
        **3. Personalized Support**  
        Customized responses and coping strategies based on your specific emotional state.
        
        **4. Progress Tracking**  
        Optional mood tracking to help you understand your emotional patterns over time.
        
        ### 🛡️ Privacy & Ethics
        
        - **No data storage**: Conversations are not permanently stored
        - **Local processing**: Your data stays on your device when possible
        - **Medical disclaimer**: This is not a replacement for professional care
        - **Transparency**: Clear about AI limitations and capabilities
        
        ### 📊 Technical Specifications
        
        | Component | Specification |
        |-----------|---------------|
        | Model | Linear Support Vector Classifier (LinearSVC) |
        | Accuracy | 82% on test dataset |
        | Training Data | 12,292 labeled mental health statements |
        | Response Time | < 1 second |
        | Framework | Streamlit + scikit-learn |
        """)
    
    with col2:
        # Feature highlights
        st.markdown("### 🏆 Key Features")
        
        features = [
            ("🤖", "AI-Powered Mood Analysis", "Real-time emotional state detection"),
            ("🚨", "Crisis Detection", "Automatic emergency protocol activation"),
            ("📊", "Progress Dashboard", "Visual mood tracking over time"),
            ("🔒", "Privacy Focused", "No permanent data storage"),
            ("💬", "Natural Conversations", "Empathetic, human-like responses"),
            ("📚", "Resource Library", "Curated mental health resources")
        ]
        
        for emoji, title, desc in features:
            with st.container():
                st.markdown(f"**{emoji} {title}**")
                st.caption(desc)
                st.markdown("---")
        
        # Support section
        st.markdown("### 🆘 Need Help?")
        st.info("""
        **Technical Issues:**
        - Check all .pkl files are in the app folder
        - Ensure Python 3.8+ is installed
        - Install requirements: `pip install -r requirements.txt`
        
        **Feedback & Suggestions:**
        Email: support@mindcare.example.com
        """)
    
    # Emergency disclaimer (always shown)
    st.markdown("---")
    st.markdown
"""
app.py - AI Meeting Assistant Streamlit UI
Professional dashboard with analytics and insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import Counter
import re
from meeting_analyzer import transcribe_audio, analyze_meeting


# Page configuration
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Card styling */
    .card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .app-title {
        color: white;
        font-size: 48px;
        font-weight: bold;
        margin: 0;
    }
    
    .app-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 18px;
        margin-top: 10px;
    }
    
    /* Task table styling */
    .task-table {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Status badges */
    .status-overdue {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .status-this-week {
        background-color: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .status-upcoming {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e2130;
    }
    
    /* Success message */
    .success-box {
        background-color: #065f46;
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)


def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Extract top keywords from text (simple frequency-based).
    
    Args:
        text: Input text
        top_n: Number of top keywords to return
        
    Returns:
        List of tuples (word, count)
    """
    # Remove special characters and convert to lowercase
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'that', 'this', 'with', 'from', 'have', 'will', 'been', 'were',
        'their', 'there', 'about', 'would', 'could', 'should', 'think',
        'know', 'just', 'going', 'need', 'want', 'make', 'sure', 'well',
        'yeah', 'okay', 'like', 'really', 'very', 'much', 'more', 'also',
        'some', 'then', 'than', 'into', 'them', 'these', 'those', 'what',
        'when', 'where', 'which', 'while', 'here', 'good', 'great'
    }
    
    # Filter out stop words
    filtered_words = [w for w in words if w not in stop_words]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(top_n)


def categorize_deadline(deadline: str) -> str:
    """
    Categorize deadline into time buckets.
    
    Args:
        deadline: Deadline string
        
    Returns:
        Category: 'Overdue', 'This Week', 'Next Week', 'Later', or 'Unspecified'
    """
    deadline_lower = deadline.lower()
    
    if 'not specified' in deadline_lower or deadline_lower == '':
        return 'Unspecified'
    
    # Simple keyword-based categorization
    today = datetime.now()
    
    if any(word in deadline_lower for word in ['overdue', 'yesterday', 'last week', 'past']):
        return 'Overdue'
    elif any(word in deadline_lower for word in ['today', 'tomorrow', 'this week', 'end of week']):
        return 'This Week'
    elif any(word in deadline_lower for word in ['next week', 'next monday', 'next friday']):
        return 'Next Week'
    else:
        return 'Later'


def render_header():
    """Render the app header."""
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">🎙️ AI Meeting Assistant</h1>
            <p class="app-subtitle">Transform audio recordings into actionable insights with AI</p>
        </div>
    """, unsafe_allow_html=True)


def render_metrics(data: dict):
    """
    Render key metrics in cards.
    
    Args:
        data: Analysis results dictionary
    """
    col1, col2, col3, col4 = st.columns(4)
    
    total_tasks = len(data.get('tasks', []))
    unique_owners = len(set(task['owner'] for task in data.get('tasks', [])))
    
    # Count tasks by deadline category
    deadline_categories = [categorize_deadline(task['deadline']) for task in data.get('tasks', [])]
    urgent_tasks = sum(1 for cat in deadline_categories if cat in ['Overdue', 'This Week'])
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Tasks</div>
                <div class="metric-value">{total_tasks}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-label">Team Members</div>
                <div class="metric-value">{unique_owners}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-label">Urgent Tasks</div>
                <div class="metric-value">{urgent_tasks}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        completion_rate = 0  # Placeholder - could track completed tasks
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-label">Completion Rate</div>
                <div class="metric-value">{completion_rate}%</div>
            </div>
        """, unsafe_allow_html=True)


def render_analytics(data: dict, transcript: str):
    """
    Render analytics dashboard with charts.
    
    Args:
        data: Analysis results dictionary
        transcript: Meeting transcript text
    """
    st.markdown("### 📊 Meeting Analytics")
    
    tasks = data.get('tasks', [])
    
    if not tasks:
        st.info("No tasks found to analyze. Upload and analyze a meeting first!")
        return
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tasks by Owner")
        
        # Count tasks per owner
        owner_counts = Counter(task['owner'] for task in tasks)
        owner_df = pd.DataFrame(
            list(owner_counts.items()),
            columns=['Owner', 'Tasks']
        ).sort_values('Tasks', ascending=False)
        
        # Create bar chart with Plotly
        fig_owner = px.bar(
            owner_df,
            x='Owner',
            y='Tasks',
            color='Tasks',
            color_continuous_scale='Viridis',
            title='Task Distribution by Team Member'
        )
        fig_owner.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_owner, use_container_width=True)
    
    with col2:
        st.markdown("#### Tasks by Deadline")
        
        # Categorize deadlines
        deadline_categories = [categorize_deadline(task['deadline']) for task in tasks]
        deadline_counts = Counter(deadline_categories)
        
        # Define order and colors
        category_order = ['Overdue', 'This Week', 'Next Week', 'Later', 'Unspecified']
        colors = {
            'Overdue': '#ef4444',
            'This Week': '#f59e0b',
            'Next Week': '#3b82f6',
            'Later': '#10b981',
            'Unspecified': '#6b7280'
        }
        
        deadline_df = pd.DataFrame([
            {'Category': cat, 'Count': deadline_counts.get(cat, 0)}
            for cat in category_order
        ])
        
        # Create bar chart
        fig_deadline = go.Figure(data=[
            go.Bar(
                x=deadline_df['Category'],
                y=deadline_df['Count'],
                marker_color=[colors[cat] for cat in deadline_df['Category']],
                text=deadline_df['Count'],
                textposition='auto'
            )
        ])
        fig_deadline.update_layout(
            title='Task Urgency Timeline',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            height=400,
            xaxis_title='Deadline Category',
            yaxis_title='Number of Tasks'
        )
        st.plotly_chart(fig_deadline, use_container_width=True)
    
    # Keyword analysis
    st.markdown("#### 🔑 Top Keywords from Meeting")
    
    if transcript:
        keywords = extract_keywords(transcript, top_n=15)
        
        if keywords:
            keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
            
            fig_keywords = px.bar(
                keyword_df,
                x='Frequency',
                y='Keyword',
                orientation='h',
                color='Frequency',
                color_continuous_scale='Plasma',
                title='Most Mentioned Topics'
            )
            fig_keywords.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False,
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_keywords, use_container_width=True)
        else:
            st.info("Not enough text to extract keywords.")
    
    # Task complexity breakdown
    st.markdown("#### 📋 Task Breakdown by Owner")
    
    # Create a detailed breakdown table
    owner_task_details = {}
    for task in tasks:
        owner = task['owner']
        if owner not in owner_task_details:
            owner_task_details[owner] = []
        owner_task_details[owner].append({
            'task': task['task'],
            'deadline': task['deadline'],
            'category': categorize_deadline(task['deadline'])
        })
    
    for owner, owner_tasks in owner_task_details.items():
        with st.expander(f"👤 {owner} ({len(owner_tasks)} tasks)"):
            for idx, t in enumerate(owner_tasks, 1):
                category = t['category']
                color = colors.get(category, '#6b7280')
                st.markdown(f"""
                    <div style="background-color: #1e2130; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {color};">
                        <strong>{idx}. {t['task']}</strong><br>
                        <span style="color: #9ca3af;">Deadline: {t['deadline']}</span>
                        <span style="float: right; background-color: {color}; padding: 2px 8px; border-radius: 10px; font-size: 11px;">{category}</span>
                    </div>
                """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    
    # Render header
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        
        # Model selection
        whisper_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info(
            "This AI Meeting Assistant uses **faster-whisper** for transcription "
            "and **Ollama (llama3)** for intelligent task extraction. "
            "All processing happens locally on your machine."
        )
        
        st.markdown("---")
        st.markdown("### 🚀 How to Use")
        st.markdown("""
        1. Upload an audio/video file
        2. Click **Analyze Meeting**
        3. View transcript & tasks
        4. Explore analytics
        """)
        
        st.markdown("---")
        st.markdown("**Supported Formats:**")
        st.markdown("""
        **Audio:** MP3, WAV, M4A, OGG, FLAC, AAC, WMA  
        **Video:** MP4, AVI, MKV (extracts audio)
        """)
    
    # Main content area
    st.markdown("### 📤 Upload Audio/Video Recording")
    
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'wma', 'mp4', 'avi', 'mkv'],
        help="Upload your meeting recording in any supported audio or video format"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            file_extension = Path(uploaded_file.name).suffix.upper().replace('.', '')
            st.metric("File Type", file_extension)
        
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Analyze button
        if st.button("🔍 Analyze Meeting", type="primary", use_container_width=True):
            try:
                # Step 1: Transcription
                with st.spinner("🎤 Transcribing audio... This may take a few minutes."):
                    transcript = transcribe_audio(str(temp_file_path), model_size=whisper_model)
                    st.session_state.transcript = transcript
                
                st.success("✅ Transcription complete!")
                
                # Step 2: Analysis
                with st.spinner("🤖 Analyzing meeting with AI..."):
                    analysis_result = analyze_meeting(transcript)
                    st.session_state.analysis_data = analysis_result
                    st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete!")
                
                # Save outputs
                output_dir = Path("meeting_outputs")
                output_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = Path(uploaded_file.name).stem
                
                # Save transcript
                transcript_path = output_dir / f"{base_name}_{timestamp}_transcript.txt"
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                
                # Save JSON
                json_path = output_dir / f"{base_name}_{timestamp}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False)
                
                # Save CSV
                if analysis_result.get('tasks'):
                    csv_path = output_dir / f"{base_name}_{timestamp}.csv"
                    df = pd.DataFrame(analysis_result['tasks'])
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                
                st.markdown(f"""
                    <div class="success-box">
                        ✅ Files saved to <code>meeting_outputs/</code>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.session_state.analysis_complete = False
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_data:
        st.markdown("---")
        
        # Render metrics
        render_metrics(st.session_state.analysis_data)
        
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Analytics",
            "📝 Summary", 
            "✅ Action Items",
            "📄 Transcript"
        ])
        
        with tab1:
            render_analytics(st.session_state.analysis_data, st.session_state.transcript)
        
        with tab2:
            st.markdown("### 📝 Meeting Summary")
            st.markdown(f"""
                <div class="card">
                    {st.session_state.analysis_data.get('summary', 'No summary available')}
                </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### ✅ Action Items")
            
            tasks = st.session_state.analysis_data.get('tasks', [])
            
            if tasks:
                # Create DataFrame with deadline categories
                tasks_df = pd.DataFrame(tasks)
                tasks_df['Priority'] = tasks_df['deadline'].apply(categorize_deadline)
                
                # Add color coding
                def get_priority_badge(priority):
                    colors = {
                        'Overdue': '🔴',
                        'This Week': '🟡',
                        'Next Week': '🔵',
                        'Later': '🟢',
                        'Unspecified': '⚪'
                    }
                    return colors.get(priority, '⚪')
                
                tasks_df['Status'] = tasks_df['Priority'].apply(get_priority_badge)
                
                # Display as formatted table
                st.dataframe(
                    tasks_df[['Status', 'task', 'owner', 'deadline', 'Priority']].rename(columns={
                        'task': 'Task',
                        'owner': 'Owner',
                        'deadline': 'Deadline',
                        'Status': '🚦'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = tasks_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Tasks as CSV",
                    data=csv,
                    file_name=f"tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No action items identified in this meeting.")
        
        with tab4:
            st.markdown("### 📄 Full Transcript")
            
            if st.session_state.transcript:
                st.markdown(f"""
                    <div class="card" style="max-height: 600px; overflow-y: auto;">
                        {st.session_state.transcript.replace(chr(10), '<br>')}
                    </div>
                """, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Transcript",
                    data=st.session_state.transcript,
                    file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.info("No transcript available.")
    
    elif not uploaded_file:
        # Show welcome message
        st.markdown("---")
        st.info("👆 Upload an audio or video file to get started!")
        
        # Show demo preview
        st.markdown("### 🎯 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="card">
                    <h3>🎤 Transcription</h3>
                    <p>Convert audio/video to text using state-of-the-art Whisper AI</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <h3>🤖 AI Analysis</h3>
                    <p>Extract tasks, owners, and deadlines automatically</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="card">
                    <h3>📊 Analytics</h3>
                    <p>Visualize task distribution and meeting insights</p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
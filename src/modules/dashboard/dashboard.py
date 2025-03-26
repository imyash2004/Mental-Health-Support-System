import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go

class DashboardModule:
    """
    Dashboard Module for providing a centralized view of user interactions and resources.
    Displays assessment results, chat history, and other user data.
    """
    
    def __init__(self, db):
        """
        Initialize the Dashboard Module.
        
        Args:
            db: Database instance for retrieving user data
        """
        self.db = db
    
    def render(self):
        """Render the dashboard interface."""
        st.markdown("<div class='main-header'>Your Dashboard</div>", unsafe_allow_html=True)
        
        # Check if user is logged in
        if not st.session_state.user_name:
            st.warning("Please enter your name in the sidebar to view your personalized dashboard.")
            return
        
        # Get user ID
        user_id = self.db.get_or_create_user(st.session_state.user_name)
        
        # Display welcome message
        st.markdown(
            f"""
            <div class='info-box'>
            Welcome to your personal dashboard, {st.session_state.user_name}! 
            Here you can track your mental health journey, view assessment results, and access resources.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Create tabs for different dashboard sections
        tab1, tab2, tab3 = st.tabs(["Assessment Results", "Activity Summary", "Resources"])
        
        # Tab 1: Assessment Results
        with tab1:
            st.markdown("<div class='sub-header'>Your Assessment Results</div>", unsafe_allow_html=True)
            
            # Get assessment results from database
            assessment_results = self.db.get_assessment_results(user_id)
            
            if not assessment_results:
                st.info("You haven't taken any assessments yet. Take an assessment to see your results here.")
                if st.button("Take Self-Assessment", key="dashboard_to_assessment"):
                    st.session_state.current_page = 'Self Assessment'
                    st.rerun()
            else:
                # Display the most recent assessment result
                latest_result = assessment_results[0]
                
                st.markdown(f"**Latest Assessment:** {latest_result['assessment_type'].title()} Assessment")
                st.markdown(f"**Date:** {datetime.fromisoformat(latest_result['created_at']).strftime('%B %d, %Y at %I:%M %p')}")
                
                # Display score and risk level
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Score:** {latest_result['score']:.1f}/10")
                    
                    # Create a gauge chart for the score
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = latest_result['score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Mental Health Score"},
                        gauge = {
                            'axis': {'range': [0, 10]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 3], 'color': "green"},
                                {'range': [3, 7], 'color': "yellow"},
                                {'range': [7, 10], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': latest_result['score']
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig)
                
                with col2:
                    # Determine risk level based on score
                    risk_level = "Low"
                    if latest_result['score'] >= 7:
                        risk_level = "High"
                    elif latest_result['score'] >= 3:
                        risk_level = "Moderate"
                    
                    st.markdown(f"**Risk Level:** {risk_level}")
                    
                    # Display risk level explanation
                    if risk_level == "Low":
                        st.markdown(
                            """
                            <div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>
                            <strong>Low Risk:</strong> Your responses suggest you're currently experiencing good mental health. 
                            Continue your self-care practices and monitor your well-being.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    elif risk_level == "Moderate":
                        st.markdown(
                            """
                            <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px;'>
                            <strong>Moderate Risk:</strong> Your responses indicate some challenges with your mental health. 
                            Consider implementing the recommendations below and reaching out for support if needed.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    elif risk_level == "High":
                        st.markdown(
                            """
                            <div style='background-color: #f8d7da; padding: 10px; border-radius: 5px;'>
                            <strong>High Risk:</strong> Your responses suggest significant challenges with your mental health. 
                            We strongly recommend consulting with a mental health professional for support.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                # Display recommendations
                st.markdown("<div class='sub-header'>Recommendations</div>", unsafe_allow_html=True)
                
                recommendations = json.loads(latest_result['recommendations']) if isinstance(latest_result['recommendations'], str) else latest_result['recommendations']
                
                for recommendation in recommendations:
                    st.markdown(
                        f"""
                        <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <strong>{recommendation['category']}:</strong> {recommendation['text']}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Display assessment history if there are multiple assessments
                if len(assessment_results) > 1:
                    st.markdown("<div class='sub-header'>Assessment History</div>", unsafe_allow_html=True)
                    
                    # Prepare data for the chart
                    history_data = []
                    for result in assessment_results:
                        history_data.append({
                            'date': datetime.fromisoformat(result['created_at']),
                            'score': result['score'],
                            'type': result['assessment_type'].title()
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    
                    # Create line chart of assessment scores over time
                    fig = px.line(
                        history_df, 
                        x='date', 
                        y='score', 
                        color='type',
                        title='Assessment Scores Over Time',
                        labels={'date': 'Date', 'score': 'Score', 'type': 'Assessment Type'},
                        markers=True
                    )
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Score",
                        yaxis=dict(range=[0, 10])
                    )
                    
                    st.plotly_chart(fig)
                
                # Button to take another assessment
                if st.button("Take Another Assessment", key="dashboard_to_assessment"):
                    st.session_state.current_page = 'Self Assessment'
                    st.rerun()
        
        # Tab 2: Activity Summary
        with tab2:
            st.markdown("<div class='sub-header'>Your Activity Summary</div>", unsafe_allow_html=True)
            
            # Get chat history from database
            chat_history = self.db.get_chat_history(user_id)
            
            # Get assessment results from database (already fetched above)
            if 'assessment_results' not in locals():
                assessment_results = self.db.get_assessment_results(user_id)
            
            # Display activity stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Assessments Taken",
                    value=len(assessment_results)
                )
            
            with col2:
                st.metric(
                    label="Chat Interactions",
                    value=len(chat_history) // 2  # Divide by 2 because each interaction has user and bot messages
                )
            
            with col3:
                # Calculate days active
                if assessment_results or chat_history:
                    dates = []
                    for result in assessment_results:
                        dates.append(datetime.fromisoformat(result['created_at']).date())
                    
                    for message in chat_history:
                        dates.append(datetime.fromisoformat(message['created_at']).date())
                    
                    unique_dates = len(set(dates))
                    st.metric(
                        label="Days Active",
                        value=unique_dates
                    )
                else:
                    st.metric(
                        label="Days Active",
                        value=0
                    )
            
            # Display recent activity
            st.markdown("<div class='sub-header'>Recent Activity</div>", unsafe_allow_html=True)
            
            if not assessment_results and not chat_history:
                st.info("No activity recorded yet. Take an assessment or chat with the support bot to get started.")
            else:
                # Combine assessment results and chat history into a single activity timeline
                activities = []
                
                for result in assessment_results:
                    activities.append({
                        'type': 'assessment',
                        'subtype': result['assessment_type'],
                        'timestamp': datetime.fromisoformat(result['created_at']),
                        'details': f"Completed {result['assessment_type'].title()} Assessment with score {result['score']:.1f}/10"
                    })
                
                for i in range(0, len(chat_history), 2):
                    if i + 1 < len(chat_history):
                        activities.append({
                            'type': 'chat',
                            'subtype': 'conversation',
                            'timestamp': datetime.fromisoformat(chat_history[i]['created_at']),
                            'details': f"Chat conversation: '{chat_history[i]['user_message'][:50]}...' → '{chat_history[i+1]['bot_message'][:50]}...'"
                        })
                
                # Sort activities by timestamp (most recent first)
                activities.sort(key=lambda x: x['timestamp'], reverse=True)
                
                # Display activities
                for activity in activities[:10]:  # Show only the 10 most recent activities
                    activity_date = activity['timestamp'].strftime('%B %d, %Y at %I:%M %p')
                    
                    if activity['type'] == 'assessment':
                        st.markdown(
                            f"""
                            <div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{activity_date}</strong><br>
                            📋 {activity['details']}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    elif activity['type'] == 'chat':
                        st.markdown(
                            f"""
                            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{activity_date}</strong><br>
                            💬 {activity['details']}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
        
        # Tab 3: Resources
        with tab3:
            st.markdown("<div class='sub-header'>Recommended Resources</div>", unsafe_allow_html=True)
            
            # Display resources based on assessment results
            if assessment_results:
                latest_result = assessment_results[0]
                assessment_type = latest_result['assessment_type']
                
                # Determine risk level based on score
                risk_level = "Low"
                if latest_result['score'] >= 7:
                    risk_level = "High"
                elif latest_result['score'] >= 3:
                    risk_level = "Moderate"
                
                # Display resources based on assessment type and risk level
                if assessment_type == "general":
                    st.markdown("### General Mental Health Resources")
                    
                    resources = [
                        {
                            "title": "Mental Health Foundation",
                            "description": "Information and resources for better mental health.",
                            "url": "https://www.mentalhealth.org.uk/"
                        },
                        {
                            "title": "National Alliance on Mental Illness (NAMI)",
                            "description": "Mental health education, advocacy, and support.",
                            "url": "https://www.nami.org/"
                        },
                        {
                            "title": "Mental Health America",
                            "description": "Tools, resources, and community for mental health.",
                            "url": "https://www.mhanational.org/"
                        }
                    ]
                    
                    for resource in resources:
                        st.markdown(
                            f"""
                            <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{resource['title']}</strong><br>
                            {resource['description']}<br>
                            <a href="{resource['url']}" target="_blank">{resource['url']}</a>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                elif assessment_type == "anxiety":
                    st.markdown("### Anxiety Resources")
                    
                    resources = [
                        {
                            "title": "Anxiety and Depression Association of America",
                            "description": "Information, resources, and support for anxiety disorders.",
                            "url": "https://adaa.org/"
                        },
                        {
                            "title": "Calm App",
                            "description": "Meditation and relaxation app for reducing anxiety.",
                            "url": "https://www.calm.com/"
                        },
                        {
                            "title": "Anxiety Canada",
                            "description": "Evidence-based resources for managing anxiety.",
                            "url": "https://www.anxietycanada.com/"
                        }
                    ]
                    
                    for resource in resources:
                        st.markdown(
                            f"""
                            <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{resource['title']}</strong><br>
                            {resource['description']}<br>
                            <a href="{resource['url']}" target="_blank">{resource['url']}</a>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                elif assessment_type == "depression":
                    st.markdown("### Depression Resources")
                    
                    resources = [
                        {
                            "title": "Depression and Bipolar Support Alliance",
                            "description": "Support, education, and resources for depression.",
                            "url": "https://www.dbsalliance.org/"
                        },
                        {
                            "title": "Headspace",
                            "description": "Meditation and mindfulness app with specific programs for depression.",
                            "url": "https://www.headspace.com/"
                        },
                        {
                            "title": "National Institute of Mental Health - Depression",
                            "description": "Information and resources about depression from NIMH.",
                            "url": "https://www.nimh.nih.gov/health/topics/depression"
                        }
                    ]
                    
                    for resource in resources:
                        st.markdown(
                            f"""
                            <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{resource['title']}</strong><br>
                            {resource['description']}<br>
                            <a href="{resource['url']}" target="_blank">{resource['url']}</a>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                # Add crisis resources if risk level is high
                if risk_level == "High":
                    st.markdown("### Crisis Resources")
                    
                    resources = [
                        {
                            "title": "National Suicide Prevention Lifeline",
                            "description": "24/7 support for people in distress.",
                            "url": "https://suicidepreventionlifeline.org/",
                            "phone": "1-800-273-8255"
                        },
                        {
                            "title": "Crisis Text Line",
                            "description": "Text HOME to 741741 to connect with a Crisis Counselor.",
                            "url": "https://www.crisistextline.org/",
                            "phone": "Text HOME to 741741"
                        },
                        {
                            "title": "SAMHSA's National Helpline",
                            "description": "Treatment referral and information service for individuals facing mental health challenges.",
                            "url": "https://www.samhsa.gov/find-help/national-helpline",
                            "phone": "1-800-662-4357"
                        }
                    ]
                    
                    for resource in resources:
                        st.markdown(
                            f"""
                            <div style='background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{resource['title']}</strong><br>
                            {resource['description']}<br>
                            <strong>Phone: {resource['phone']}</strong><br>
                            <a href="{resource['url']}" target="_blank">{resource['url']}</a>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.info("Take an assessment to receive personalized resource recommendations.")
                if st.button("Take Self-Assessment", key="resources_to_assessment"):
                    st.session_state.current_page = 'Self Assessment'
                    st.rerun()
            
            # General resources section
            st.markdown("### General Resources")
            
            resources = [
                {
                    "title": "Mental Health First Aid",
                    "description": "Learn how to help someone who is developing a mental health problem or experiencing a mental health crisis.",
                    "url": "https://www.mentalhealthfirstaid.org/"
                },
                {
                    "title": "Psychology Today - Find a Therapist",
                    "description": "Directory to find therapists, psychiatrists, treatment centers, and support groups.",
                    "url": "https://www.psychologytoday.com/us/therapists"
                },
                {
                    "title": "MindTools - Stress Management Techniques",
                    "description": "Practical techniques for managing stress and improving well-being.",
                    "url": "https://www.mindtools.com/pages/article/managing-stress.htm"
                }
            ]
            
            for resource in resources:
                st.markdown(
                    f"""
                    <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <strong>{resource['title']}</strong><br>
                    {resource['description']}<br>
                    <a href="{resource['url']}" target="_blank">{resource['url']}</a>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

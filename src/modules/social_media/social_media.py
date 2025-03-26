
import streamlit as st
import pandas as pd
import plotly.express as px
from .analyzer import SocialMediaAnalyzer

class SocialMediaModule:
    """
    Social Media Module for analyzing social media posts for mental health concerns.
    """
    
    def __init__(self, db=None):
        """
        Initialize the Social Media Module.
        
        Args:
            db: Database instance (optional)
        """
        self.db = db
        self.analyzer = SocialMediaAnalyzer()
    
    def render(self):
        """Render the social media analysis interface."""
        st.markdown("<div class='main-header'>Social Media Analysis</div>", unsafe_allow_html=True)
        
        # Display information about the module
        st.markdown(
            """
            <div class='info-box'>
            This module analyzes social media posts for potential mental health concerns. 
            It can help identify early warning signs and provide appropriate recommendations.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Initialize the active tab in session state if not already there
        if 'social_media_active_tab' not in st.session_state:
            st.session_state.social_media_active_tab = "Social Media Analysis"  # Default to analysis tab
        
        # Create radio buttons for tab selection (more reliable than st.tabs for maintaining state)
        tab_options = ["Social Media Analysis", "Sample Analysis", "About"]
        selected_tab = st.radio("Select Tab", tab_options, 
                               index=tab_options.index(st.session_state.social_media_active_tab),
                               key="social_media_tab_selector", 
                               horizontal=True)
        
        # Update the session state based on the selected tab
        st.session_state.social_media_active_tab = selected_tab
        
        # Create a horizontal line for visual separation
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Social Media Analysis Tab
        if selected_tab == "Social Media Analysis":
            st.markdown("<div class='sub-header'>Analyze Your Social Media Content</div>", unsafe_allow_html=True)
            
            # Create subtabs for different social media platforms
            platform_tabs = st.tabs(["Direct Input", "Twitter/X", "Facebook", "Instagram", "Reddit"])
            
            # Direct Input Tab
            with platform_tabs[0]:
                st.markdown(
                    """
                    <div class='info-box'>
                    Enter any social media posts, messages, or personal thoughts below to analyze them for potential mental health concerns.
                    This analysis is performed locally and your data is not stored or shared.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Text input
                custom_text = st.text_area("Enter text to analyze:", height=150, 
                                          placeholder="Example: I've been feeling really down lately and can't seem to find joy in things I used to love. Sometimes I wonder if life is worth living.")
                
                # Analysis button
                analyze_button = st.button("Analyze Text", key="analyze_direct_input")
                
                if analyze_button and custom_text.strip():
                    with st.spinner("Analyzing your content..."):
                        # Analyze the text
                        concern_scores = self.analyzer.analyze_post(custom_text)
                        risk_level = self.analyzer.get_risk_level(concern_scores)
                        
                        # Get top concerns
                        top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
                        top_concerns = [concern for concern, score in top_concerns if score > 0]
                        
                        # Display analysis results in a nice container
                        st.markdown("<div class='sub-header'>Analysis Results</div>", unsafe_allow_html=True)
                        
                        # Display explanation if available
                        if 'explanation' in concern_scores:
                            st.markdown(
                                f"""
                                <div style='background-color: rgba(58, 58, 78, 0.8); padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                <strong style='color: #ffffff;'>Analysis:</strong> <span style='color: white;'>{concern_scores['explanation']}</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        # Display risk level with appropriate color
                        risk_color = "#4cd97b"  # Bright green for low risk
                        risk_bg_color = "rgba(76, 217, 123, 0.15)"
                        if risk_level == "Moderate":
                            risk_color = "#ffcc5c"  # Bright yellow for moderate risk
                            risk_bg_color = "rgba(255, 204, 92, 0.15)"
                        elif risk_level == "High":
                            risk_color = "#ff5c5c"  # Bright red for high risk
                            risk_bg_color = "rgba(255, 92, 92, 0.15)"
                        
                        st.markdown(
                            f"""
                            <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Display top concerns
                        st.markdown("**Top Concerns:**")
                        if top_concerns:
                            for concern in top_concerns[:3]:  # Top 3 concerns
                                if concern != 'explanation':  # Skip the explanation field
                                    st.markdown(f"- {concern.replace('_', ' ').title()}")
                        else:
                            st.markdown("No significant concerns detected.")
                        
                        # Display concern scores as a bar chart
                        # Filter out non-numeric values like 'explanation'
                        numeric_scores = {k: v for k, v in concern_scores.items() if isinstance(v, (int, float))}
                        
                        if any(numeric_scores.values()):
                            # Convert to DataFrame for plotting
                            df = pd.DataFrame({
                                'Concern': [k.replace('_', ' ').title() for k in numeric_scores.keys()],
                                'Score': list(numeric_scores.values())
                            })
                            
                            # Sort by score
                            df = df.sort_values('Score', ascending=False)
                            
                            # Create bar chart
                            fig = px.bar(
                                df, 
                                x='Concern', 
                                y='Score',
                                title='Concern Scores',
                                color='Score',
                                color_continuous_scale=['green', 'yellow', 'red'],
                                range_color=[0, 1]
                            )
                            
                            # Update layout for dark theme
                            fig.update_layout(
                                plot_bgcolor='rgba(42, 42, 58, 0.8)',
                                paper_bgcolor='rgba(42, 42, 58, 0.8)',
                                font=dict(color='white')
                            )
                            
                            st.plotly_chart(fig)
                        
                        # Display recommendations
                        st.markdown("**Recommendations:**")
                        recommendations = self.analyzer.get_recommendations(risk_level, [c for c in top_concerns[:3] if c != 'explanation'])
                        
                        for recommendation in recommendations:
                            st.markdown(
                                f"""
                                <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        # Save to database if user is logged in
                        if 'user_name' in st.session_state and st.session_state.user_name and self.db:
                            try:
                                user_id = self.db.get_or_create_user(st.session_state.user_name)
                                # Save analysis result (implementation would depend on your database structure)
                                st.success("Analysis saved to your profile.")
                            except Exception as e:
                                st.error(f"Could not save analysis: {e}")
                elif analyze_button:
                    st.warning("Please enter some text to analyze.")
            
            # Twitter/X Tab
            with platform_tabs[1]:
                st.markdown(
                    """
                    <div class='info-box'>
                    Enter your Twitter/X username or paste specific tweets to analyze for mental health concerns.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                twitter_username = st.text_input("Twitter/X Username (without @):", placeholder="username")
                twitter_posts = st.text_area("Or paste specific tweets (one per line):", height=150)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    num_tweets = st.number_input("Number of recent tweets to analyze:", min_value=1, max_value=50, value=10)
                
                analyze_twitter = st.button("Analyze Twitter Content", key="analyze_twitter")
                
                if analyze_twitter:
                    if twitter_username.strip() or twitter_posts.strip():
                        with st.spinner("This feature requires Twitter API access. For now, we'll analyze the text you provided directly."):
                            # For now, just analyze the pasted tweets
                            if twitter_posts.strip():
                                # Split by lines and analyze each tweet
                                tweets = twitter_posts.strip().split('\n')
                                all_text = ' '.join(tweets)
                                
                                # Analyze the combined text
                                concern_scores = self.analyzer.analyze_post(all_text)
                                risk_level = self.analyzer.get_risk_level(concern_scores)
                                
                                # Get top concerns
                                top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
                                top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                                
                                # Display risk level with appropriate color
                                risk_color = "#4cd97b"  # Bright green for low risk
                                risk_bg_color = "rgba(76, 217, 123, 0.15)"
                                if risk_level == "Moderate":
                                    risk_color = "#ffcc5c"  # Bright yellow for moderate risk
                                    risk_bg_color = "rgba(255, 204, 92, 0.15)"
                                elif risk_level == "High":
                                    risk_color = "#ff5c5c"  # Bright red for high risk
                                    risk_bg_color = "rgba(255, 92, 92, 0.15)"
                                
                                st.markdown(
                                    f"""
                                    <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                    <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                                
                                # Display top concerns
                                st.markdown("**Top Concerns:**")
                                if top_concerns:
                                    for concern in top_concerns[:3]:  # Top 3 concerns
                                        st.markdown(f"- {concern.replace('_', ' ').title()}")
                                else:
                                    st.markdown("No significant concerns detected.")
                                
                                # Display recommendations
                                st.markdown("**Recommendations:**")
                                recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                                
                                for recommendation in recommendations:
                                    st.markdown(
                                        f"""
                                        <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                        <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("Twitter API integration is coming soon. Please paste specific tweets for now.")
                    else:
                        st.warning("Please enter a Twitter username or paste some tweets.")
            
            # Facebook Tab
            with platform_tabs[2]:
                st.markdown(
                    """
                    <div class='info-box'>
                    Paste your Facebook posts or status updates to analyze for mental health concerns.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                facebook_posts = st.text_area("Paste Facebook posts (one per line):", height=150)
                analyze_facebook = st.button("Analyze Facebook Content", key="analyze_facebook")
                
                if analyze_facebook:
                    if facebook_posts.strip():
                        with st.spinner("Analyzing your Facebook content..."):
                            # Analyze the text
                            concern_scores = self.analyzer.analyze_post(facebook_posts)
                            risk_level = self.analyzer.get_risk_level(concern_scores)
                            
                            # Get top concerns
                            top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
                            top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                            
                            # Display risk level with appropriate color
                            risk_color = "#4cd97b"  # Bright green for low risk
                            risk_bg_color = "rgba(76, 217, 123, 0.15)"
                            if risk_level == "Moderate":
                                risk_color = "#ffcc5c"  # Bright yellow for moderate risk
                                risk_bg_color = "rgba(255, 204, 92, 0.15)"
                            elif risk_level == "High":
                                risk_color = "#ff5c5c"  # Bright red for high risk
                                risk_bg_color = "rgba(255, 92, 92, 0.15)"
                            
                            st.markdown(
                                f"""
                                <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                            
                            # Display top concerns
                            st.markdown("**Top Concerns:**")
                            if top_concerns:
                                for concern in top_concerns[:3]:  # Top 3 concerns
                                    st.markdown(f"- {concern.replace('_', ' ').title()}")
                            else:
                                st.markdown("No significant concerns detected.")
                            
                            # Display recommendations
                            st.markdown("**Recommendations:**")
                            recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                            
                            for recommendation in recommendations:
                                st.markdown(
                                    f"""
                                    <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                    <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.warning("Please paste some Facebook posts to analyze.")
            
            # Instagram Tab
            with platform_tabs[3]:
                st.markdown(
                    """
                    <div class='info-box'>
                    Paste your Instagram captions or comments to analyze for mental health concerns.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                instagram_posts = st.text_area("Paste Instagram captions (one per line):", height=150)
                analyze_instagram = st.button("Analyze Instagram Content", key="analyze_instagram")
                
                if analyze_instagram:
                    if instagram_posts.strip():
                        with st.spinner("Analyzing your Instagram content..."):
                            # Analyze the text
                            concern_scores = self.analyzer.analyze_post(instagram_posts)
                            risk_level = self.analyzer.get_risk_level(concern_scores)
                            
                            # Get top concerns
                            top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
                            top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                            
                            # Display risk level with appropriate color
                            risk_color = "#4cd97b"  # Bright green for low risk
                            risk_bg_color = "rgba(76, 217, 123, 0.15)"
                            if risk_level == "Moderate":
                                risk_color = "#ffcc5c"  # Bright yellow for moderate risk
                                risk_bg_color = "rgba(255, 204, 92, 0.15)"
                            elif risk_level == "High":
                                risk_color = "#ff5c5c"  # Bright red for high risk
                                risk_bg_color = "rgba(255, 92, 92, 0.15)"
                            
                            st.markdown(
                                f"""
                                <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                            
                            # Display top concerns
                            st.markdown("**Top Concerns:**")
                            if top_concerns:
                                for concern in top_concerns[:3]:  # Top 3 concerns
                                    st.markdown(f"- {concern.replace('_', ' ').title()}")
                            else:
                                st.markdown("No significant concerns detected.")
                            
                            # Display concern scores as a bar chart
                            # Filter out non-numeric values like 'explanation'
                            numeric_scores = {k: v for k, v in concern_scores.items() if isinstance(v, (int, float))}
                            
                            if any(numeric_scores.values()):
                                # Convert to DataFrame for plotting
                                df = pd.DataFrame({
                                    'Concern': [k.replace('_', ' ').title() for k in numeric_scores.keys()],
                                    'Score': list(numeric_scores.values())
                                })
                                
                                # Sort by score
                                df = df.sort_values('Score', ascending=False)
                                
                                # Create bar chart
                                fig = px.bar(
                                    df, 
                                    x='Concern', 
                                    y='Score',
                                    title='Concern Scores',
                                    color='Score',
                                    color_continuous_scale=['green', 'yellow', 'red'],
                                    range_color=[0, 1]
                                )
                                
                                # Update layout for dark theme
                                fig.update_layout(
                                    plot_bgcolor='rgba(42, 42, 58, 0.8)',
                                    paper_bgcolor='rgba(42, 42, 58, 0.8)',
                                    font=dict(color='white')
                                )
                                
                                st.plotly_chart(fig)
                            
                            # Display recommendations
                            st.markdown("**Recommendations:**")
                            recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                            
                            for recommendation in recommendations:
                                st.markdown(
                                    f"""
                                    <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                    <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.warning("Please paste some Instagram captions to analyze.")
            
            # Reddit Tab
            with platform_tabs[4]:
                st.markdown(
                    """
                    <div class='info-box'>
                    Enter your Reddit username or paste specific posts/comments to analyze for mental health concerns.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                reddit_username = st.text_input("Reddit Username (without u/):", placeholder="username")
                reddit_posts = st.text_area("Or paste specific posts/comments (one per line):", height=150)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    num_posts = st.number_input("Number of recent posts to analyze:", min_value=1, max_value=50, value=10)
                
                analyze_reddit = st.button("Analyze Reddit Content", key="analyze_reddit")
                
                if analyze_reddit:
                    if reddit_username.strip() or reddit_posts.strip():
                        with st.spinner("This feature requires Reddit API access. For now, we'll analyze the text you provided directly."):
                            # For now, just analyze the pasted posts
                            if reddit_posts.strip():
                                # Analyze the text
                                concern_scores = self.analyzer.analyze_post(reddit_posts)
                                risk_level = self.analyzer.get_risk_level(concern_scores)
                                
                                # Get top concerns
                                top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
                                top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                                
                                # Display risk level with appropriate color
                                risk_color = "#4cd97b"  # Bright green for low risk
                                risk_bg_color = "rgba(76, 217, 123, 0.15)"
                                if risk_level == "Moderate":
                                    risk_color = "#ffcc5c"  # Bright yellow for moderate risk
                                    risk_bg_color = "rgba(255, 204, 92, 0.15)"
                                elif risk_level == "High":
                                    risk_color = "#ff5c5c"  # Bright red for high risk
                                    risk_bg_color = "rgba(255, 92, 92, 0.15)"
                                
                                st.markdown(
                                    f"""
                                    <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                    <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                                
                                # Display top concerns
                                st.markdown("**Top Concerns:**")
                                if top_concerns:
                                    for concern in top_concerns[:3]:  # Top 3 concerns
                                        st.markdown(f"- {concern.replace('_', ' ').title()}")
                                else:
                                    st.markdown("No significant concerns detected.")
                                
                                # Display concern scores as a bar chart
                                # Filter out non-numeric values like 'explanation'
                                numeric_scores = {k: v for k, v in concern_scores.items() if isinstance(v, (int, float))}
                                
                                if any(numeric_scores.values()):
                                    # Convert to DataFrame for plotting
                                    df = pd.DataFrame({
                                        'Concern': [k.replace('_', ' ').title() for k in numeric_scores.keys()],
                                        'Score': list(numeric_scores.values())
                                    })
                                    
                                    # Sort by score
                                    df = df.sort_values('Score', ascending=False)
                                    
                                    # Create bar chart
                                    fig = px.bar(
                                        df, 
                                        x='Concern', 
                                        y='Score',
                                        title='Concern Scores',
                                        color='Score',
                                        color_continuous_scale=['green', 'yellow', 'red'],
                                        range_color=[0, 1]
                                    )
                                    
                                    # Update layout for dark theme
                                    fig.update_layout(
                                        plot_bgcolor='rgba(42, 42, 58, 0.8)',
                                        paper_bgcolor='rgba(42, 42, 58, 0.8)',
                                        font=dict(color='white')
                                    )
                                    
                                    st.plotly_chart(fig)
                                
                                # Display recommendations
                                st.markdown("**Recommendations:**")
                                recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                                
                                for recommendation in recommendations:
                                    st.markdown(
                                        f"""
                                        <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                        <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("Reddit API integration is coming soon. Please paste specific posts for now.")
                    else:
                        st.warning("Please enter a Reddit username or paste some posts.")
        if selected_tab == "About":
            st.title("About Mental Health Disorders")
    
            st.markdown("""
            Mental health disorders, including depression and ADHD, are common conditions that can affect anyone. 
            Understanding these disorders is crucial for seeking help and support.
            """)

            st.subheader("Depression")
            st.write("Depression is characterized by persistent feelings of sadness, hopelessness, and a lack of interest or pleasure in activities.")

            st.markdown("#### Common Symptoms:")
            st.markdown("- Persistent sadness or low mood\n- Loss of interest in activities once enjoyed\n- Changes in appetite or weight\n- Sleep disturbances\n- Fatigue or low energy\n- Difficulty concentrating or making decisions")

            st.markdown("#### Keywords:")
            st.markdown("**Sadness, Hopelessness, Anxiety, Isolation, Fatigue**")

            st.markdown("#### How to Overcome Depression:")
            st.markdown("- Seek professional help\n- Talk to someone\n- Practice self-care\n- Stay connected\n- Set realistic goals")

            st.subheader("ADHD (Attention-Deficit/Hyperactivity Disorder)")
            st.write("ADHD is a neurodevelopmental disorder characterized by patterns of inattention, hyperactivity, and impulsivity.")

            st.markdown("#### Common Symptoms:")
            st.markdown("- Difficulty sustaining attention\n- Impulsivity\n- Hyperactivity\n- Disorganization")

            st.markdown("#### Keywords:")
            st.markdown("**Inattention, Hyperactivity, Impulsivity, Disorganization**")

            st.markdown("#### How to Overcome ADHD:")
            st.markdown("- Seek professional evaluation and treatment\n- Implement organizational strategies\n- Use reminders and tools to stay focused\n- Consider behavioral therapy")

            st.subheader("Other Common Mental Health Disorders")
            st.markdown("- **Anxiety Disorders**: Characterized by excessive fear or worry.")
            st.markdown("- **Bipolar Disorder**: Involves mood swings ranging from depressive lows to manic highs.")

            st.markdown("#### General Strategies for Overcoming Mental Health Disorders:")
            st.markdown("- **Seek Professional Help**: Consult a mental health professional for therapy or counseling.")
            st.markdown("- **Talk to Someone**: Share your feelings with friends or family members who can provide support.")
            st.markdown("- **Practice Self-Care**: Engage in activities that promote well-being, such as exercise, healthy eating, and mindfulness.")
            st.markdown("- **Stay Connected**: Maintain social connections and avoid isolation.")
            st.markdown("- **Set Realistic Goals**: Break tasks into smaller, manageable steps to avoid feeling overwhelmed.")




            
        

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from .analyzer import SocialMediaAnalyzer

# class SocialMediaModule:
#     """
#     Social Media Module for analyzing social media posts for mental health concerns.
#     """
    
#     def __init__(self, db=None):
#         """
#         Initialize the Social Media Module.
        
#         Args:
#             db: Database instance (optional)
#         """
#         self.db = db
#         self.analyzer = SocialMediaAnalyzer()
    
#     def render(self):
#         """Render the social media analysis interface."""
#         st.markdown("<div class='main-header'>Social Media Analysis</div>", unsafe_allow_html=True)
        
#         # Display information about the module
#         st.markdown(
#             """
#             <div class='info-box'>
#             This module analyzes social media posts for potential mental health concerns. 
#             It can help identify early warning signs and provide appropriate recommendations.
#             </div>
#             """
#         )
        
#         # Initialize the active tab in session state if not already there
#         if 'social_media_active_tab' not in st.session_state:
#             st.session_state.social_media_active_tab = "Social Media Analysis"  # Default to analysis tab
        
#         # Create radio buttons for tab selection (more reliable than st.tabs for maintaining state)
#         tab_options = ["Social Media Analysis", "About"]
#         selected_tab = st.radio("Select Tab", tab_options, 
#                                index=tab_options.index(st.session_state.social_media_active_tab),
#                                key="social_media_tab_selector", 
#                                horizontal=True)
        
#         # Update the session state based on the selected tab
#         st.session_state.social_media_active_tab = selected_tab
        
#         # Create a horizontal line for visual separation
#         st.markdown("<hr>", unsafe_allow_html=True)
        
#         # Social Media Analysis Tab
#         if selected_tab == "Social Media Analysis":
#             st.markdown("<div class='sub-header'>Analyze Your Social Media Content</div>", unsafe_allow_html=True)
#             # (Existing code for analysis tab)
#             # ...
        
#         # Text input
#         custom_text = st.text_area("Enter text to analyze:", height=150, 
#                                           placeholder="Example: I've been feeling really down lately and can't seem to find joy in things I used to love. Sometimes I wonder if life is worth living.")
                
#         # Analysis button
#         analyze_button = st.button("Analyze Text", key="analyze_direct_input")
                
#         if analyze_button and custom_text.strip():
#             with st.spinner("Analyzing your content..."):
#                 # Analyze the text
#                 concern_scores = self.analyzer.analyze_post(custom_text)
#                 risk_level = self.analyzer.get_risk_level(concern_scores)
                
#                 # Get top concerns
#                 top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
#                 top_concerns = [concern for concern, score in top_concerns if score > 0]
                
#                 # Display analysis results in a nice container
#                 st.markdown("<div class='sub-header'>Analysis Results</div>", unsafe_allow_html=True)
                
#                 # Display explanation if available
#                 if 'explanation' in concern_scores:
#                     st.markdown(
#                         f"""
#                         <div style='background-color: rgba(58, 58, 78, 0.8); padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                         <strong style='color: #ffffff;'>Analysis:</strong> <span style='color: white;'>{concern_scores['explanation']}</span>
#                         </div>
#                         """, 
#                         unsafe_allow_html=True
#                     )
                
#                 # Display risk level with appropriate color
#                 risk_color = "#4cd97b"  # Bright green for low risk
#                 risk_bg_color = "rgba(76, 217, 123, 0.15)"
#                 if risk_level == "Moderate":
#                     risk_color = "#ffcc5c"  # Bright yellow for moderate risk
#                     risk_bg_color = "rgba(255, 204, 92, 0.15)"
#                 elif risk_level == "High":
#                     risk_color = "#ff5c5c"  # Bright red for high risk
#                     risk_bg_color = "rgba(255, 92, 92, 0.15)"
                
#                 st.markdown(
#                     f"""
#                     <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                     <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )
                
#                 # Display top concerns
#                 st.markdown("**Top Concerns:**")
#                 if top_concerns:
#                     for concern in top_concerns[:3]:  # Top 3 concerns
#                         if concern != 'explanation':  # Skip the explanation field
#                             st.markdown(f"- {concern.replace('_', ' ').title()}")
#                 else:
#                     st.markdown("No significant concerns detected.")
                
#                 # Display concern scores as a bar chart
#                 # Filter out non-numeric values like 'explanation'
#                 numeric_scores = {k: v for k, v in concern_scores.items() if isinstance(v, (int, float))}
                
#                 if any(numeric_scores.values()):
#                     # Convert to DataFrame for plotting
#                     df = pd.DataFrame({
#                         'Concern': [k.replace('_', ' ').title() for k in numeric_scores.keys()],
#                         'Score': list(numeric_scores.values())
#                     })
                    
#                     # Sort by score
#                     df = df.sort_values('Score', ascending=False)
                    
#                     # Create bar chart
#                     fig = px.bar(
#                         df, 
#                         x='Concern', 
#                         y='Score',
#                         title='Concern Scores',
#                         color='Score',
#                         color_continuous_scale=['green', 'yellow', 'red'],
#                         range_color=[0, 1]
#                     )
                    
#                     # Update layout for dark theme
#                     fig.update_layout(
#                         plot_bgcolor='rgba(42, 42, 58, 0.8)',
#                         paper_bgcolor='rgba(42, 42, 58, 0.8)',
#                         font=dict(color='white')
#                     )
                    
#                     st.plotly_chart(fig)
                
#                 # Display recommendations
#                 st.markdown("**Recommendations:**")
#                 recommendations = self.analyzer.get_recommendations(risk_level, [c for c in top_concerns[:3] if c != 'explanation'])
                
#                 for recommendation in recommendations:
#                     st.markdown(
#                         f"""
#                         <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                         <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
#                         </div>
#                         """, 
#                         unsafe_allow_html=True
#                     )
                
#                 # Save to database if user is logged in
#                 if 'user_name' in st.session_state and st.session_state.user_name and self.db:
#                     try:
#                         user_id = self.db.get_or_create_user(st.session_state.user_name)
#                         # Save analysis result (implementation would depend on your database structure)
#                         self.db.save_analysis(user_id, custom_text, concern_scores)
#                         st.success("Analysis saved to your profile.")
#                     except Exception as e:
#                         st.error(f"Could not save analysis: {e}")
#         elif analyze_button:
#             st.warning("Please enter some text to analyze.")
            
#         # Twitter/X Tab
#         with platform_tabs[1]:
#             st.markdown(
#                 """
#                 <div class='info-box'>
#                 Enter your Twitter/X username or paste specific tweets to analyze for mental health concerns.
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
            
#             twitter_username = st.text_input("Twitter/X Username (without @):", placeholder="username")
#             twitter_posts = st.text_area("Or paste specific tweets (one per line):", height=150)
            
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 num_tweets = st.number_input("Number of recent tweets to analyze:", min_value=1, max_value=50, value=10)
            
#             analyze_twitter = st.button("Analyze Twitter Content", key="analyze_twitter")
            
#             if analyze_twitter:
#                 if twitter_username.strip() or twitter_posts.strip():
#                     with st.spinner("This feature requires Twitter API access. For now, we'll analyze the text you provided directly."):
#                         # For now, just analyze the pasted tweets
#                         if twitter_posts.strip():
#                             # Split by lines and analyze each tweet
#                             tweets = twitter_posts.strip().split('\n')
#                             all_text = ' '.join(tweets)
                            
#                             # Analyze the combined text
#                             concern_scores = self.analyzer.analyze_post(all_text)
#                             risk_level = self.analyzer.get_risk_level(concern_scores)
                            
#                             # Get top concerns
#                             top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
#                             top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                            
#                             # Display risk level with appropriate color
#                             risk_color = "#4cd97b"  # Bright green for low risk
#                             risk_bg_color = "rgba(76, 217, 123, 0.15)"
#                             if risk_level == "Moderate":
#                                 risk_color = "#ffcc5c"  # Bright yellow for moderate risk
#                                 risk_bg_color = "rgba(255, 204, 92, 0.15)"
#                             elif risk_level == "High":
#                                 risk_color = "#ff5c5c"  # Bright red for high risk
#                                 risk_bg_color = "rgba(255, 92, 92, 0.15)"
                            
#                             st.markdown(
#                                 f"""
#                                 <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                 <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
#                                 </div>
#                                 """, 
#                                 unsafe_allow_html=True
#                             )
                            
#                             # Display top concerns
#                             st.markdown("**Top Concerns:**")
#                             if top_concerns:
#                                 for concern in top_concerns[:3]:  # Top 3 concerns
#                                     st.markdown(f"- {concern.replace('_', ' ').title()}")
#                             else:
#                                 st.markdown("No significant concerns detected.")
                            
#                             # Display recommendations
#                             st.markdown("**Recommendations:**")
#                             recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                            
#                             for recommendation in recommendations:
#                                 st.markdown(
#                                     f"""
#                                     <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                     <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
#                                     </div>
#                                     """, 
#                                     unsafe_allow_html=True
#                                 )
#                         else:
#                             st.info("Twitter API integration is coming soon. Please paste specific tweets for now.")
#                 else:
#                     st.warning("Please enter a Twitter username or paste some tweets.")
            
#             # Facebook Tab
#             with platform_tabs[2]:
#                 st.markdown(
#                     """
#                     <div class='info-box'>
#                     Paste your Facebook posts or status updates to analyze for mental health concerns.
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )
                
#                 facebook_posts = st.text_area("Paste Facebook posts (one per line):", height=150)
#                 analyze_facebook = st.button("Analyze Facebook Content", key="analyze_facebook")
                
#                 if analyze_facebook:
#                     if facebook_posts.strip():
#                         with st.spinner("Analyzing your Facebook content..."):
#                             # Analyze the text
#                             concern_scores = self.analyzer.analyze_post(facebook_posts)
#                             risk_level = self.analyzer.get_risk_level(concern_scores)
                            
#                             # Get top concerns
#                             top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
#                             top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                            
#                             # Display risk level with appropriate color
#                             risk_color = "#4cd97b"  # Bright green for low risk
#                             risk_bg_color = "rgba(76, 217, 123, 0.15)"
#                             if risk_level == "Moderate":
#                                 risk_color = "#ffcc5c"  # Bright yellow for moderate risk
#                                 risk_bg_color = "rgba(255, 204, 92, 0.15)"
#                             elif risk_level == "High":
#                                 risk_color = "#ff5c5c"  # Bright red for high risk
#                                 risk_bg_color = "rgba(255, 92, 92, 0.15)"
                            
#                             st.markdown(
#                                 f"""
#                                 <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                 <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
#                                 </div>
#                                 """, 
#                                 unsafe_allow_html=True
#                             )
                            
#                             # Display top concerns
#                             st.markdown("**Top Concerns:**")
#                             if top_concerns:
#                                 for concern in top_concerns[:3]:  # Top 3 concerns
#                                     st.markdown(f"- {concern.replace('_', ' ').title()}")
#                             else:
#                                 st.markdown("No significant concerns detected.")
                            
#                             # Display recommendations
#                             st.markdown("**Recommendations:**")
#                             recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                            
#                             for recommendation in recommendations:
#                                 st.markdown(
#                                     f"""
#                                     <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                     <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
#                                     </div>
#                                     """, 
#                                     unsafe_allow_html=True
#                                 )
#                     else:
#                         st.warning("Please paste some Facebook posts to analyze.")
            
#             # Instagram Tab
#             with platform_tabs[3]:
#                 st.markdown(
#                     """
#                     <div class='info-box'>
#                     Paste your Instagram captions or comments to analyze for mental health concerns.
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )
                
#                 instagram_posts = st.text_area("Paste Instagram captions (one per line):", height=150)
#                 analyze_instagram = st.button("Analyze Instagram Content", key="analyze_instagram")
                
#                 if analyze_instagram:
#                     if instagram_posts.strip():
#                         with st.spinner("Analyzing your Instagram content..."):
#                             # Analyze the text
#                             concern_scores = self.analyzer.analyze_post(instagram_posts)
#                             risk_level = self.analyzer.get_risk_level(concern_scores)
                            
#                             # Get top concerns
#                             top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
#                             top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                            
#                             # Display risk level with appropriate color
#                             risk_color = "#4cd97b"  # Bright green for low risk
#                             risk_bg_color = "rgba(76, 217, 123, 0.15)"
#                             if risk_level == "Moderate":
#                                 risk_color = "#ffcc5c"  # Bright yellow for moderate risk
#                                 risk_bg_color = "rgba(255, 204, 92, 0.15)"
#                             elif risk_level == "High":
#                                 risk_color = "#ff5c5c"  # Bright red for high risk
#                                 risk_bg_color = "rgba(255, 92, 92, 0.15)"
                            
#                             st.markdown(
#                                 f"""
#                                 <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                 <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
#                                 </div>
#                                 """, 
#                                 unsafe_allow_html=True
#                             )
                            
#                             # Display top concerns
#                             st.markdown("**Top Concerns:**")
#                             if top_concerns:
#                                 for concern in top_concerns[:3]:  # Top 3 concerns
#                                     st.markdown(f"- {concern.replace('_', ' ').title()}")
#                             else:
#                                 st.markdown("No significant concerns detected.")
                            
#                             # Display concern scores as a bar chart
#                             # Filter out non-numeric values like 'explanation'
#                             numeric_scores = {k: v for k, v in concern_scores.items() if isinstance(v, (int, float))}
                            
#                             if any(numeric_scores.values()):
#                                 # Convert to DataFrame for plotting
#                                 df = pd.DataFrame({
#                                     'Concern': [k.replace('_', ' ').title() for k in numeric_scores.keys()],
#                                     'Score': list(numeric_scores.values())
#                                 })
                                
#                                 # Sort by score
#                                 df = df.sort_values('Score', ascending=False)
                                
#                                 # Create bar chart
#                                 fig = px.bar(
#                                     df, 
#                                     x='Concern', 
#                                     y='Score',
#                                     title='Concern Scores',
#                                     color='Score',
#                                     color_continuous_scale=['green', 'yellow', 'red'],
#                                     range_color=[0, 1]
#                                 )
                                
#                                 # Update layout for dark theme
#                                 fig.update_layout(
#                                     plot_bgcolor='rgba(42, 42, 58, 0.8)',
#                                     paper_bgcolor='rgba(42, 42, 58, 0.8)',
#                                     font=dict(color='white')
#                                 )
                                
#                                 st.plotly_chart(fig)
                            
#                             # Display recommendations
#                             st.markdown("**Recommendations:**")
#                             recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                            
#                             for recommendation in recommendations:
#                                 st.markdown(
#                                     f"""
#                                     <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                     <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
#                                     </div>
#                                     """, 
#                                     unsafe_allow_html=True
#                                 )
#                     else:
#                         st.warning("Please paste some Instagram captions to analyze.")
            
#             # Reddit Tab
#             with platform_tabs[4]:
#                 st.markdown(
#                     """
#                     <div class='info-box'>
#                     Enter your Reddit username or paste specific posts/comments to analyze for mental health concerns.
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )
                
#                 reddit_username = st.text_input("Reddit Username (without u/):", placeholder="username")
#                 reddit_posts = st.text_area("Or paste specific posts/comments (one per line):", height=150)
                
#                 col1, col2 = st.columns([1, 3])
#                 with col1:
#                     num_posts = st.number_input("Number of recent posts to analyze:", min_value=1, max_value=50, value=10)
                
#                 analyze_reddit = st.button("Analyze Reddit Content", key="analyze_reddit")
                
#                 if analyze_reddit:
#                     if reddit_username.strip() or reddit_posts.strip():
#                         with st.spinner("This feature requires Reddit API access. For now, we'll analyze the text you provided directly."):
#                             # For now, just analyze the pasted posts
#                             if reddit_posts.strip():
#                                 # Analyze the text
#                                 concern_scores = self.analyzer.analyze_post(reddit_posts)
#                                 risk_level = self.analyzer.get_risk_level(concern_scores)
                                
#                                 # Get top concerns
#                                 top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
#                                 top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                                
#                                 # Display risk level with appropriate color
#                                 risk_color = "#4cd97b"  # Bright green for low risk
#                                 risk_bg_color = "rgba(76, 217, 123, 0.15)"
#                                 if risk_level == "Moderate":
#                                     risk_color = "#ffcc5c"  # Bright yellow for moderate risk
#                                     risk_bg_color = "rgba(255, 204, 92, 0.15)"
#                                 elif risk_level == "High":
#                                     risk_color = "#ff5c5c"  # Bright red for high risk
#                                     risk_bg_color = "rgba(255, 92, 92, 0.15)"
                                
#                                 st.markdown(
#                                     f"""
#                                     <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                     <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
#                                     </div>
#                                     """, 
#                                     unsafe_allow_html=True
#                                 )
                                
#                                 # Display top concerns
#                                 st.markdown("**Top Concerns:**")
#                                 if top_concerns:
#                                     for concern in top_concerns[:3]:  # Top 3 concerns
#                                         st.markdown(f"- {concern.replace('_', ' ').title()}")
#                                 else:
#                                     st.markdown("No significant concerns detected.")
                                
#                                 # Display concern scores as a bar chart
#                                 # Filter out non-numeric values like 'explanation'
#                                 numeric_scores = {k: v for k, v in concern_scores.items() if isinstance(v, (int, float))}
                                
#                                 if any(numeric_scores.values()):
#                                     # Convert to DataFrame for plotting
#                                     df = pd.DataFrame({
#                                         'Concern': [k.replace('_', ' ').title() for k in numeric_scores.keys()],
#                                         'Score': list(numeric_scores.values())
#                                     })
                                    
#                                     # Sort by score
#                                     df = df.sort_values('Score', ascending=False)
                                    
#                                     # Create bar chart
#                                     fig = px.bar(
#                                         df, 
#                                         x='Concern', 
#                                         y='Score',
#                                         title='Concern Scores',
#                                         color='Score',
#                                         color_continuous_scale=['green', 'yellow', 'red'],
#                                         range_color=[0, 1]
#                                     )
                                    
#                                     # Update layout for dark theme
#                                     fig.update_layout(
#                                         plot_bgcolor='rgba(42, 42, 58, 0.8)',
#                                         paper_bgcolor='rgba(42, 42, 58, 0.8)',
#                                         font=dict(color='white')
#                                     )
                                    
#                                     st.plotly_chart(fig)
                                
#                                 # Display recommendations
#                                 st.markdown("**Recommendations:**")
#                                 recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                                
#                                 for recommendation in recommendations:
#                                     st.markdown(
#                                         f"""
#                                         <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                         <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
#                                         </div>
#                                         """, 
#                                         unsafe_allow_html=True
#                                     )
#                             else:
#                                 st.info("Reddit API integration is coming soon. Please paste specific posts for now.")
                    
                    
                    
                    # else:
#                     #     st.warning("Please enter a Reddit username or paste some posts.")
# ```

# Now, I will proceed to save the updated file. 

# <create_file>
# <path>src/modules/social_media/social_media.py</path>
# <content>
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from .analyzer import SocialMediaAnalyzer

# class SocialMediaModule:
#     """
#     Social Media Module for analyzing social media posts for mental health concerns.
#     """
    
#     def __init__(self, db=None):
#         """
#         Initialize the Social Media Module.
        
#         Args:
#             db: Database instance (optional)
#         """
#         self.db = db
#         self.analyzer = SocialMediaAnalyzer()
    
#     def render(self):
#         """Render the social media analysis interface."""
#         st.markdown("<div class='main-header'>Social Media Analysis</div>", unsafe_allow_html=True)
        
#         # Display information about the module
#         st.markdown(
#             """
#             <div class='info-box'>
#             This module analyzes social media posts for potential mental health concerns. 
#             It can help identify early warning signs and provide appropriate recommendations.
#             </div>
#             """
#         )
        
#         # Initialize the active tab in session state if not already there
#         if 'social_media_active_tab' not in st.session_state:
#             st.session_state.social_media_active_tab = "Social Media Analysis"  # Default to analysis tab
        
#         # Create radio buttons for tab selection (more reliable than st.tabs for maintaining state)
#         tab_options = ["Social Media Analysis", "About"]
#         selected_tab = st.radio("Select Tab", tab_options, 
#                                index=tab_options.index(st.session_state.social_media_active_tab),
#                                key="social_media_tab_selector", 
#                                horizontal=True)
        
#         # Update the session state based on the selected tab
#         st.session_state.social_media_active_tab = selected_tab
        
#         # Create a horizontal line for visual separation
#         st.markdown("<hr>", unsafe_allow_html=True)
        
#         # Social Media Analysis Tab
#         if selected_tab == "Social Media Analysis":
#             st.markdown("<div class='sub-header'>Analyze Your Social Media Content</div>", unsafe_allow_html=True)
#             # (Existing code for analysis tab)
#             # ...
        
#         # Text input
#         custom_text = st.text_area("Enter text to analyze:", height=150, 
#                                           placeholder="Example: I've been feeling really down lately and can't seem to find joy in things I used to love. Sometimes I wonder if life is worth living.")
                
#         # Analysis button
#         analyze_button = st.button("Analyze Text", key="analyze_direct_input")
                
#         if analyze_button and custom_text.strip():
#             with st.spinner("Analyzing your content..."):
#                 # Analyze the text
#                 concern_scores = self.analyzer.analyze_post(custom_text)
#                 risk_level = self.analyzer.get_risk_level(concern_scores)
                
#                 # Get top concerns
#                 top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
#                 top_concerns = [concern for concern, score in top_concerns if score > 0]
                
#                 # Display analysis results in a nice container
#                 st.markdown("<div class='sub-header'>Analysis Results</div>", unsafe_allow_html=True)
                
#                 # Display explanation if available
#                 if 'explanation' in concern_scores:
#                     st.markdown(
#                         f"""
#                         <div style='background-color: rgba(58, 58, 78, 0.8); padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                         <strong style='color: #ffffff;'>Analysis:</strong> <span style='color: white;'>{concern_scores['explanation']}</span>
#                         </div>
#                         """, 
#                         unsafe_allow_html=True
#                     )
                
#                 # Display risk level with appropriate color
#                 risk_color = "#4cd97b"  # Bright green for low risk
#                 risk_bg_color = "rgba(76, 217, 123, 0.15)"
#                 if risk_level == "Moderate":
#                     risk_color = "#ffcc5c"  # Bright yellow for moderate risk
#                     risk_bg_color = "rgba(255, 204, 92, 0.15)"
#                 elif risk_level == "High":
#                     risk_color = "#ff5c5c"  # Bright red for high risk
#                     risk_bg_color = "rgba(255, 92, 92, 0.15)"
                
#                 st.markdown(
#                     f"""
#                     <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                     <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )
                
#                 # Display top concerns
#                 st.markdown("**Top Concerns:**")
#                 if top_concerns:
#                     for concern in top_concerns[:3]:  # Top 3 concerns
#                         if concern != 'explanation':  # Skip the explanation field
#                             st.markdown(f"- {concern.replace('_', ' ').title()}")
#                 else:
#                     st.markdown("No significant concerns detected.")
                
#                 # Display concern scores as a bar chart
#                 # Filter out non-numeric values like 'explanation'
#                 numeric_scores = {k: v for k, v in concern_scores.items() if isinstance(v, (int, float))}
                
#                 if any(numeric_scores.values()):
#                     # Convert to DataFrame for plotting
#                     df = pd.DataFrame({
#                         'Concern': [k.replace('_', ' ').title() for k in numeric_scores.keys()],
#                         'Score': list(numeric_scores.values())
#                     })
                    
#                     # Sort by score
#                     df = df.sort_values('Score', ascending=False)
                    
#                     # Create bar chart
#                     fig = px.bar(
#                         df, 
#                         x='Concern', 
#                         y='Score',
#                         title='Concern Scores',
#                         color='Score',
#                         color_continuous_scale=['green', 'yellow', 'red'],
#                         range_color=[0, 1]
#                     )
                    
#                     # Update layout for dark theme
#                     fig.update_layout(
#                         plot_bgcolor='rgba(42, 42, 58, 0.8)',
#                         paper_bgcolor='rgba(42, 42, 58, 0.8)',
#                         font=dict(color='white')
#                     )
                    
#                     st.plotly_chart(fig)
                
#                 # Display recommendations
#                 st.markdown("**Recommendations:**")
#                 recommendations = self.analyzer.get_recommendations(risk_level, [c for c in top_concerns[:3] if c != 'explanation'])
                
#                 for recommendation in recommendations:
#                     st.markdown(
#                         f"""
#                         <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                         <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
#                         </div>
#                         """, 
#                         unsafe_allow_html=True
#                     )
                
#                 # Save to database if user is logged in
#                 if 'user_name' in st.session_state and st.session_state.user_name and self.db:
#                     try:
#                         user_id = self.db.get_or_create_user(st.session_state.user_name)
#                         # Save analysis result (implementation would depend on your database structure)
#                         self.db.save_analysis(user_id, custom_text, concern_scores)
#                         st.success("Analysis saved to your profile.")
#                     except Exception as e:
#                         st.error(f"Could not save analysis: {e}")
#         elif analyze_button:
#             st.warning("Please enter some text to analyze.")
            
#         # Twitter/X Tab
#         with platform_tabs[1]:
#             st.markdown(
#                 """
#                 <div class='info-box'>
#                 Enter your Twitter/X username or paste specific tweets to analyze for mental health concerns.
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
            
#             twitter_username = st.text_input("Twitter/X Username (without @):", placeholder="username")
#             twitter_posts = st.text_area("Or paste specific tweets (one per line):", height=150)
            
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 num_tweets = st.number_input("Number of recent tweets to analyze:", min_value=1, max_value=50, value=10)
            
#             analyze_twitter = st.button("Analyze Twitter Content", key="analyze_twitter")
            
#             if analyze_twitter:
#                 if twitter_username.strip() or twitter_posts.strip():
#                     with st.spinner("This feature requires Twitter API access. For now, we'll analyze the text you provided directly."):
#                         # For now, just analyze the pasted tweets
#                         if twitter_posts.strip():
#                             # Split by lines and analyze each tweet
#                             tweets = twitter_posts.strip().split('\n')
#                             all_text = ' '.join(tweets)
                            
#                             # Analyze the combined text
#                             concern_scores = self.analyzer.analyze_post(all_text)
#                             risk_level = self.analyzer.get_risk_level(concern_scores)
                            
#                             # Get top concerns
#                             top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
#                             top_concerns = [concern for concern, score in top_concerns if score > 0 and concern != 'explanation']
                            
#                             # Display risk level with appropriate color
#                             risk_color = "#4cd97b"  # Bright green for low risk
#                             risk_bg_color = "rgba(76, 217, 123, 0.15)"
#                             if risk_level == "Moderate":
#                                 risk_color = "#ffcc5c"  # Bright yellow for moderate risk
#                                 risk_bg_color = "rgba(255, 204, 92, 0.15)"
#                             elif risk_level == "High":
#                                 risk_color = "#ff5c5c"  # Bright red for high risk
#                                 risk_bg_color = "rgba(255, 92, 92, 0.15)"
                            
#                             st.markdown(
#                                 f"""
#                                 <div style='background-color: {risk_bg_color}; border-left: 4px solid {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                 <strong style='color: {risk_color};'>Risk Level:</strong> <span style='color: white;'>{risk_level}</span>
#                                 </div>
#                                 """, 
#                                 unsafe_allow_html=True
#                             )
                            
#                             # Display top concerns
#                             st.markdown("**Top Concerns:**")
#                             if top_concerns:
#                                 for concern in top_concerns[:3]:  # Top 3 concerns
#                                     st.markdown(f"- {concern.replace('_', ' ').title()}")
#                             else:
#                                 st.markdown("No significant concerns detected.")
                            
#                             # Display recommendations
#                             st.markdown("**Recommendations:**")
#                             recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                            
#                             for recommendation in recommendations:
#                                 st.markdown(
#                                     f"""
#                                     <div style='background-color: rgba(58, 58, 78, 0.8); border-left: 4px solid #5ce1ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                                     <strong style='color: #5ce1ff;'>{recommendation['category']}:</strong> <span style='color: white;'>{recommendation['text']}</span>
#                                     </div>
#                                     """, 
#                                     unsafe_allow_html=True
#                                 )
#                         else:
#                             st.info("Twitter API integration is coming soon. Please paste specific tweets for now.")
#                 else:
#                     st.warning("Please enter a Twitter username or paste some tweets.")
            
#             # Facebook Tab
#             with platform_tabs[2]:
#                 st.markdown(
#                     """
#                     <div class='info-box'>
#                     Paste your Facebook posts or status updates to analyze for mental health concerns.
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )
                
#                 facebook_posts = st.text_area("Paste Facebook posts (one per line):", height=150)
#                 analyze_facebook = st.button("Analyze Facebook Content", key="analyze_facebook")
                
#                 if analyze_facebook:

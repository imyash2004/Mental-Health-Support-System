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
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Sample Analysis", "Social Media Analysis", "About"])
        
        # Tab 1: Sample Analysis
        with tab1:
            st.markdown("<div class='sub-header'>Sample Social Media Posts Analysis</div>", unsafe_allow_html=True)
            
            st.markdown(
                """
                Below are sample social media posts with mental health concern analysis.
                These posts are from the provided dataset and demonstrate how the system
                can identify potential mental health concerns in social media content.
                """
            )
            
            # Get sample posts with analysis
            sample_posts = self.analyzer.get_sample_posts(5)
            
            # Display each post with analysis
            for i, post in enumerate(sample_posts):
                with st.expander(f"Post {i+1} (User ID: {post['user_id']}) - Risk Level: {post['risk_level']}", expanded=i==0):
                    st.markdown(f"**Post Content:**")
                    st.markdown(f"<div class='info-box'>{post['post']}</div>", unsafe_allow_html=True)
                    
                    # Display risk level with appropriate color
                    risk_color = "var(--success-color)"  # Green for low risk
                    if post['risk_level'] == "Moderate":
                        risk_color = "var(--warning-color)"  # Yellow for moderate risk
                    elif post['risk_level'] == "High":
                        risk_color = "var(--danger-color)"  # Red for high risk
                    
                    st.markdown(
                        f"""
                        <div class='info-box' style='border-left: 4px solid {risk_color};'>
                        <strong>Risk Level:</strong> {post['risk_level']}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Display top concerns
                    st.markdown("**Top Concerns:**")
                    if post['top_concerns']:
                        for concern in post['top_concerns']:
                            st.markdown(f"- {concern.replace('_', ' ').title()}")
                    else:
                        st.markdown("No significant concerns detected.")
                    
                    # Display concern scores as a bar chart
                    concern_scores = post['concern_scores']
                    if any(concern_scores.values()):
                        # Convert to DataFrame for plotting
                        df = pd.DataFrame({
                            'Concern': [k.replace('_', ' ').title() for k in concern_scores.keys()],
                            'Score': list(concern_scores.values())
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
                    recommendations = self.analyzer.get_recommendations(post['risk_level'], post['top_concerns'])
                    
                    for recommendation in recommendations:
                        st.markdown(
                            f"""
                            <div class='info-box'>
                            <strong>{recommendation['category']}:</strong> {recommendation['text']}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
        
        # Tab 2: Social Media Analysis
        with tab2:
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
                        if any(concern_scores.values()):
                            # Convert to DataFrame for plotting
                            df = pd.DataFrame({
                                'Concern': [k.replace('_', ' ').title() for k in concern_scores.keys()],
                                'Score': list(concern_scores.values())
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
                        
                        # Save to database if user is logged in
                        if st.session_state.user_name and self.db:
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
                                top_concerns = [concern for concern, score in top_concerns if score > 0]
                                
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
                            top_concerns = [concern for concern, score in top_concerns if score > 0]
                            
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
                            top_concerns = [concern for concern, score in top_concerns if score > 0]
                            
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
                            if any(concern_scores.values()):
                                # Convert to DataFrame for plotting
                                df = pd.DataFrame({
                                    'Concern': [k.replace('_', ' ').title() for k in concern_scores.keys()],
                                    'Score': list(concern_scores.values())
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
                                top_concerns = [concern for concern, score in top_concerns if score > 0]
                                
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
                                if any(concern_scores.values()):
                                    # Convert to DataFrame for plotting
                                    df = pd.DataFrame({
                                        'Concern': [k.replace('_', ' ').title() for k in concern_scores.keys()],
                                        'Score': list(concern_scores.values())
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
        
        # Tab 3: About
        with tab3:
            st.markdown("<div class='sub-header'>About Social Media Analysis</div>", unsafe_allow_html=True)
            
            st.markdown(
                """
                ### How It Works
                
                The Social Media Analysis module uses natural language processing techniques to identify potential mental health concerns in text content. The current implementation uses a simple keyword-based approach for demonstration purposes, but a production system would use more sophisticated machine learning models.
                
                ### Key Features
                
                - **Mental Health Concern Detection**: Identifies potential mental health concerns such as depression, anxiety, suicidal ideation, and more.
                - **Risk Level Assessment**: Categorizes posts as Low, Moderate, or High risk based on the detected concerns.
                - **Personalized Recommendations**: Provides tailored recommendations based on the specific concerns identified.
                - **Multi-Platform Support**: Analyze content from various social media platforms including Twitter, Facebook, Instagram, and Reddit.
                
                ### Limitations
                
                - The current implementation uses a simple keyword-based approach, which may not capture the nuances of mental health language.
                - False positives and false negatives are possible, especially with complex or ambiguous language.
                - This tool is not a replacement for professional mental health assessment and should be used as a screening tool only.
                
                ### Privacy Considerations
                
                - All analysis is performed locally and no data is sent to external servers.
                - User data is not stored unless explicitly saved by the user.
                - For a production system, proper consent and privacy measures would be implemented.
                """
            )

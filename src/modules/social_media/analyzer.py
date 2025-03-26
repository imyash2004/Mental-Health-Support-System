import os
import csv
import re

class SocialMediaAnalyzer:
    """
    Social Media Analysis Module for detecting mental health concerns in social media posts.
    Uses a simple keyword-based approach for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the Social Media Analyzer."""
        # Define concern keywords for simple detection
        self.concern_keywords = {
            'depression': ['depressed', 'depression', 'sad', 'hopeless', 'worthless', 'empty', 'numb'],
            'anxiety': ['anxious', 'anxiety', 'worried', 'nervous', 'panic', 'fear', 'stress'],
            'suicidal': ['suicide', 'kill myself', 'end my life', 'die', 'death', 'no reason to live'],
            'self_harm': ['cut', 'harm', 'hurt myself', 'pain', 'injury', 'blood', 'scars'],
            'insomnia': ['sleep', 'insomnia', 'awake', 'tired', 'exhausted', 'rest', 'fatigue'],
            'substance_abuse': ['alcohol', 'drugs', 'substance', 'addiction', 'dependent', 'withdrawal']
        }
        
        # Load sample data
        self.train_data = self._load_sample_data('Full_Train_Data.tsv', 5)
        self.test_data = self._load_sample_data('Full_Test_Data.tsv', 5)
    
    def _load_sample_data(self, filename, num_samples=5):
        """
        Load a sample of data from the TSV file.
        
        Args:
            filename (str): Path to the TSV file
            num_samples (int): Number of samples to load
            
        Returns:
            list: List of dictionaries containing the data
        """
        data = []
        try:
            file_path = os.path.join(os.getcwd(), filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')
                header = next(reader)  # Skip header
                
                for i, row in enumerate(reader):
                    if i >= num_samples:
                        break
                    
                    if len(row) >= 2:
                        post_data = {
                            'user_id': row[0],
                            'post': row[1]
                        }
                        
                        # Add label if available (training data)
                        if len(row) >= 3:
                            post_data['label'] = row[2]
                        
                        data.append(post_data)
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
        
        return data
    
    def analyze_post(self, post_text):
        """
        Analyze a social media post for mental health concerns.
        
        Args:
            post_text (str): Text of the post
            
        Returns:
            dict: Dictionary of concerns and their confidence scores
        """
        if not post_text or not isinstance(post_text, str):
            return {}
        
        # Preprocess text
        post_text = post_text.lower()
        
        # Count concern keywords with word boundary checks
        concern_scores = {concern: 0 for concern in self.concern_keywords}
        
        for concern, keywords in self.concern_keywords.items():
            for keyword in keywords:
                # Use word boundary checks to avoid partial matches
                # For example, "sad" should not match "saddle"
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, post_text)
                
                # Add weight based on the number of occurrences
                if matches:
                    # Give more weight to exact matches of critical keywords
                    if concern in ['suicidal', 'self_harm'] and keyword in ['suicide', 'kill myself', 'end my life', 'cut', 'harm', 'hurt myself']:
                        concern_scores[concern] += len(matches) * 1.5
                    else:
                        concern_scores[concern] += len(matches)
        
        # Check for phrases that indicate severity
        severity_phrases = [
            "can't take it anymore", "no reason to live", "better off dead", 
            "want to die", "don't want to live", "hate myself",
            "always anxious", "constant worry", "panic attack",
            "completely hopeless", "extremely depressed", "severely depressed"
        ]
        
        for phrase in severity_phrases:
            if phrase in post_text:
                # Identify which concern this phrase relates to
                if phrase in ["can't take it anymore", "no reason to live", "better off dead", "want to die", "don't want to live"]:
                    concern_scores['suicidal'] += 2
                elif phrase in ["hate myself"]:
                    concern_scores['depression'] += 1.5
                elif phrase in ["always anxious", "constant worry", "panic attack"]:
                    concern_scores['anxiety'] += 1.5
                elif phrase in ["completely hopeless", "extremely depressed", "severely depressed"]:
                    concern_scores['depression'] += 2
        
        # Check for context indicators
        context_indicators = {
            'depression': ["for weeks", "for months", "every day", "all the time", "can't feel", "no joy"],
            'anxiety': ["constantly", "all the time", "can't stop", "overwhelming", "terrified"],
            'insomnia': ["can't sleep", "awake all night", "haven't slept", "no sleep"],
            'substance_abuse': ["need it", "can't stop", "withdrawal", "addicted", "dependency"]
        }
        
        for concern, indicators in context_indicators.items():
            for indicator in indicators:
                if indicator in post_text:
                    concern_scores[concern] += 1
        
        # Normalize scores
        max_score = max(concern_scores.values()) if concern_scores.values() else 0
        if max_score > 0:
            for concern in concern_scores:
                concern_scores[concern] /= max_score
                # Cap at 1.0
                concern_scores[concern] = min(concern_scores[concern], 1.0)
        
        return concern_scores
    
    def get_risk_level(self, concern_scores):
        """
        Determine the risk level based on concern scores.
        
        Args:
            concern_scores (dict): Dictionary of concerns and their scores
            
        Returns:
            str: Risk level (Low, Moderate, High)
        """
        # Check for suicidal or self-harm concerns first (highest priority)
        suicidal_score = concern_scores.get('suicidal', 0)
        self_harm_score = concern_scores.get('self_harm', 0)
        
        # Immediate high risk if suicidal or self-harm scores are significant
        if suicidal_score > 0.4 or self_harm_score > 0.4:
            return "High"
        
        # Check for substance abuse as a secondary concern
        substance_score = concern_scores.get('substance_abuse', 0)
        
        # Check for depression and anxiety scores
        depression_score = concern_scores.get('depression', 0)
        anxiety_score = concern_scores.get('anxiety', 0)
        
        # Calculate weighted average score with more weight on critical concerns
        weighted_scores = [
            suicidal_score * 3,  # Highest weight
            self_harm_score * 2.5,
            substance_score * 1.5,
            depression_score * 1.2,
            anxiety_score * 1.2
        ]
        
        weights = [3, 2.5, 1.5, 1.2, 1.2]
        valid_weights = sum(weights) if any(weighted_scores) else 1
        weighted_avg = sum(weighted_scores) / valid_weights
        
        # Calculate regular average as a backup
        avg_score = sum(concern_scores.values()) / len(concern_scores) if concern_scores else 0
        
        # Use the higher of the two averages
        final_score = max(weighted_avg, avg_score)
        
        # Determine risk level based on final score
        if final_score > 0.5:
            return "High"
        elif final_score > 0.2:
            return "Moderate"
        else:
            return "Low"
    
    def get_sample_posts(self, num_posts=5):
        """
        Get a sample of posts from the loaded data.
        
        Args:
            num_posts (int): Number of posts to return
            
        Returns:
            list: List of post dictionaries with analysis results
        """
        result = []
        
        # Combine train and test data
        all_data = self.train_data + self.test_data
        
        # Get unique posts
        unique_posts = []
        seen_ids = set()
        
        for post in all_data:
            if post['user_id'] not in seen_ids:
                unique_posts.append(post)
                seen_ids.add(post['user_id'])
        
        # Analyze posts
        for post in unique_posts[:num_posts]:
            concern_scores = self.analyze_post(post['post'])
            risk_level = self.get_risk_level(concern_scores)
            
            # Get top concerns
            top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
            top_concerns = [concern for concern, score in top_concerns if score > 0]
            
            result.append({
                'user_id': post['user_id'],
                'post': post['post'][:200] + '...' if len(post['post']) > 200 else post['post'],
                'risk_level': risk_level,
                'top_concerns': top_concerns[:3],  # Top 3 concerns
                'concern_scores': concern_scores
            })
        
        return result
    
    def get_recommendations(self, risk_level, top_concerns):
        """
        Get recommendations based on risk level and top concerns.
        
        Args:
            risk_level (str): Risk level (Low, Moderate, High)
            top_concerns (list): List of top concerns
            
        Returns:
            list: List of recommendation dictionaries
        """
        recommendations = []
        
        # General recommendation for all risk levels
        recommendations.append({
            "category": "General",
            "text": "Practice self-care activities like regular exercise, healthy eating, and adequate sleep."
        })
        
        # Add recommendations based on risk level
        if risk_level == "Low":
            recommendations.append({
                "category": "Maintenance",
                "text": "Continue monitoring your mental health and practice mindfulness techniques."
            })
        
        elif risk_level == "Moderate":
            recommendations.append({
                "category": "Support",
                "text": "Consider talking to a trusted friend, family member, or mental health professional about your feelings."
            })
            recommendations.append({
                "category": "Self-Help",
                "text": "Explore stress reduction techniques such as deep breathing exercises, progressive muscle relaxation, or guided imagery."
            })
        
        elif risk_level == "High":
            recommendations.append({
                "category": "Professional Help",
                "text": "We strongly recommend consulting with a mental health professional for further evaluation and support."
            })
            recommendations.append({
                "category": "Immediate Support",
                "text": "If you're experiencing a crisis, please contact a mental health helpline or emergency services."
            })
        
        # Add concern-specific recommendations
        if 'depression' in top_concerns:
            recommendations.append({
                "category": "Depression",
                "text": "Try to engage in activities you used to enjoy, even if you don't feel like it at first. Start small and be gentle with yourself."
            })
        
        if 'anxiety' in top_concerns:
            recommendations.append({
                "category": "Anxiety",
                "text": "Practice grounding techniques when feeling anxious: focus on 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste."
            })
        
        if 'suicidal' in top_concerns:
            recommendations.append({
                "category": "Crisis Support",
                "text": "Please reach out to a crisis helpline immediately. The National Suicide Prevention Lifeline is available 24/7 at 1-800-273-8255."
            })
        
        if 'self_harm' in top_concerns:
            recommendations.append({
                "category": "Self-Harm",
                "text": "Try alternative coping strategies like holding ice, snapping a rubber band on your wrist, or intense exercise when you feel the urge to harm yourself."
            })
        
        if 'insomnia' in top_concerns:
            recommendations.append({
                "category": "Sleep",
                "text": "Establish a regular sleep schedule, create a relaxing bedtime routine, and limit screen time before bed to improve sleep quality."
            })
        
        if 'substance_abuse' in top_concerns:
            recommendations.append({
                "category": "Substance Use",
                "text": "Consider seeking support from a substance abuse counselor or support group like AA or NA."
            })
        
        return recommendations

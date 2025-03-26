import os
import csv
import re
import json
import openai

# Set OpenAI API key - using the same key as in the chatbot module

class SocialMediaAnalyzer:
    """
    Social Media Analysis Module for detecting mental health concerns in social media posts.
    Uses a simple keyword-based approach for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the Social Media Analyzer."""
        # Define concern keywords for simple detection
        self.concern_keywords = {
    'depression': [
        'depressed', 'depression', 'sad', 'hopeless', 'worthless', 'empty', 'numb',
        'lonely', 'down', 'melancholy', 'miserable', 'despair', 'grief', 'tearful',
        'low mood', 'drained', 'exhausted', 'losing interest', 'apathetic', 'dead inside',
        'void', 'crying', 'can’t feel anything', 'hollow', 'isolated', 'lifeless',
        'nothing matters', 'no energy', 'dull', 'tired of everything', 'burned out',
        'dark thoughts', 'no motivation', 'can’t get out of bed', 'lost hope', 'wishing I wasn’t here',
        'feeling blue', 'emotionally drained', 'done with life', 'empty inside', 'lifeless'
    ],
    'anxiety': [
        'anxious', 'anxiety', 'worried', 'nervous', 'panic', 'fear', 'stress',
        'overthinking', 'restless', 'uneasy', 'tension', 'dread', 'overwhelmed',
        'heart racing', 'shaky', 'sweaty palms', 'butterflies in stomach',
        'can’t breathe', 'racing thoughts', 'paranoid', 'jittery', 'on edge',
        'panic attack', 'nauseous from stress', 'headache from anxiety', 'tight chest',
        'impending doom', 'mind won’t stop', 'stomach in knots', 'hyperventilating',
        'can’t focus', 'always worrying', 'constant fear', 'dizzy from panic',
        'fearful', 'spiraling thoughts', 'afraid of everything', 'too much pressure'
    ],
    'suicidal': [
        'suicide', 'kill myself', 'end my life', 'die', 'death', 'no reason to live',
        'give up', 'hopeless', 'not worth it', 'better off dead', 'goodbye forever',
        'nothing matters', 'self-destruction', 'want to disappear', 'final goodbye',
        'life is pointless', 'can’t take it anymore', 'tired of being alive',
        'wouldn’t mind dying', 'life is too painful', 'ready to go', 'no escape',
        'never wake up', 'want it all to end', 'losing the will to live',
        'thinking about ending it', 'feeling like a burden', 'nobody would care',
        'I don’t belong here', 'want to vanish', 'just want peace', 'exit plan',
        'wish I wasn’t here', 'darkest thoughts', 'last resort', 'giving up on life'
    ],
    'self_harm': [
        'cut', 'harm', 'hurt myself', 'pain', 'injury', 'blood', 'scars',
        'burn myself', 'bruise', 'self-inflicted', 'scratch', 'wound', 'hidden scars',
        'numb the pain', 'punish myself', 'release the pain', 'bleeding',
        'marks on my skin', 'pain is the only thing I feel', 'relief through pain',
        'carving into skin', 'hidden wounds', 'self-inflicted wounds', 'need to feel something',
        'hurting makes it better', 'deserve the pain', 'feels like I need it',
        'body is a canvas of pain', 'I like seeing the blood', 'pain addiction'
    ],
    'insomnia': [
        'sleep', 'insomnia', 'awake', 'tired', 'exhausted', 'rest', 'fatigue',
        'can’t sleep', 'no sleep', 'sleepless', 'restless nights', 'tossing and turning',
        'waking up at night', 'nightmares', 'disturbed sleep', 'early waking',
        'brain won’t shut off', 'stuck in my thoughts', 'midnight thoughts',
        'sleep deprived', 'hours without sleep', 'exhaustion hitting hard',
        'eyes heavy but can’t sleep', 'clock watching', 'counting sheep doesn’t work',
        'sleep is impossible', 'wish I could sleep', 'mind racing at night',
        'no rest for the weary', 'day and night are the same', 'bedtime struggle',
        'permanently tired', 'zombie mode', 'can’t function without sleep'
    ],
    'substance_abuse': [
        'alcohol', 'drugs', 'substance', 'addiction', 'dependent', 'withdrawal',
        'drunk', 'high', 'overdose', 'rehab', 'relapse', 'binge drinking',
        'substance craving', 'can’t stop', 'drug abuse', 'blackout', 'hangover',
        'opioids', 'stimulants', 'cocaine', 'heroin', 'meth', 'pills', 'painkillers',
        'need a drink', 'can’t get through the day without it', 'getting wasted',
        'substance use problem', 'drowning my sorrows', 'popping pills',
        'chasing a high', 'need a fix', 'withdrawal symptoms', 'can’t function sober',
        'losing control', 'drugs are my escape', 'addicted to the feeling',
        'hooked on it', 'can’t quit', 'always looking for my next hit'
    ]
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
    
    def analyze_post_with_openai(self, post_text):
        """
        Analyze a social media post for mental health concerns using OpenAI.
        
        Args:
            post_text (str): Text of the post
            
        Returns:
            dict: Dictionary of concerns and their confidence scores
        """
        if not post_text or not isinstance(post_text, str):
            return {}
        
        try:
            # Create a system message that provides context about the analysis task
            system_message = """
            You are an AI mental health analyzer. Your task is to analyze the provided text for signs of mental health concerns.
            Specifically, evaluate the text for the following concerns:
            
            1. Depression
            2. Anxiety
            3. Suicidal thoughts
            4. Self-harm
            5. Insomnia
            6. Substance abuse
            
            For each concern, provide a score between 0 and 1, where:
            - 0 means no indication of this concern
            - 0.3 means mild indication
            - 0.6 means moderate indication
            - 0.9-1.0 means strong indication
            
            Pay special attention to suicidal thoughts and self-harm, as these are critical concerns.
            
            Return your analysis in the following JSON format:
            {
                "depression": score (0-1),
                "anxiety": score (0-1),
                "suicidal": score (0-1),
                "self_harm": score (0-1),
                "insomnia": score (0-1),
                "substance_abuse": score (0-1),
                "explanation": "Brief explanation of your analysis"
            }
            
            Only return the JSON object, nothing else.
            """
            
            # Create the conversation with the system message and user message
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Analyze this text for mental health concerns: \"{post_text}\""}
                ],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more consistent results
                response_format={"type": "json_object"}  # Request JSON response
            )
            
            # Extract the response text
            result_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            try:
                result = json.loads(result_text)
                
                # Ensure all required keys are present
                concern_scores = {
                    'depression': float(result.get('depression', 0)),
                    'anxiety': float(result.get('anxiety', 0)),
                    'suicidal': float(result.get('suicidal', 0)),
                    'self_harm': float(result.get('self_harm', 0)),
                    'insomnia': float(result.get('insomnia', 0)),
                    'substance_abuse': float(result.get('substance_abuse', 0))
                }
                
                # Store explanation if available
                if 'explanation' in result:
                    concern_scores['explanation'] = result['explanation']
                
                return concern_scores
                
            except json.JSONDecodeError:
                print(f"Error parsing OpenAI response: {result_text}")
                # Fall back to keyword-based analysis
                return self.analyze_post_with_keywords(post_text)
                
        except Exception as e:
            print(f"Error using OpenAI for analysis: {e}")
            # Fall back to keyword-based analysis
            return self.analyze_post_with_keywords(post_text)
    
    def analyze_post_with_keywords(self, post_text):
        """
        Analyze a social media post for mental health concerns using keyword matching.
        This is a fallback method when OpenAI analysis fails.
        
        Args:
            post_text (str): Text of the post
            
        Returns:
            dict: Dictionary of concerns and their confidence scores
        """
        if not post_text or not isinstance(post_text, str):
            return {}
        
        # Preprocess text
        post_text = post_text.lower().strip()
        
        # Direct detection of critical phrases - these should immediately trigger high scores
        critical_phrases = {
            'suicidal': [
                'suicidal thoughts', 'thinking about suicide', 'want to kill myself', 
                'planning to end my life', 'considering suicide', 'suicidal ideation',
                'suicidal', 'kill myself', 'end my life', 'end it all', 'take my own life',
                'don\'t want to live anymore', 'want to die', 'ready to die',
                'thinking of ending it', 'no reason to live', 'life is not worth living'
            ],
            'self_harm': [
                'cutting myself', 'hurting myself', 'self harm', 'self-harm', 
                'harming myself', 'injuring myself', 'burning myself',
                'want to hurt myself', 'thinking about hurting myself'
            ]
        }
        
        # Initialize concern scores
        concern_scores = {concern: 0 for concern in self.concern_keywords}
        
        # Check for direct critical phrases first
        for concern, phrases in critical_phrases.items():
            for phrase in phrases:
                if phrase in post_text:
                    # Direct matches to critical phrases get a very high score
                    concern_scores[concern] = 0.9  # Almost maximum score
                    # If it's suicidal thoughts, this is extremely serious
                    if concern == 'suicidal' and ('suicidal thoughts' in post_text or 'thinking about suicide' in post_text):
                        concern_scores[concern] = 1.0  # Maximum score
        
        # Count concern keywords with word boundary checks
        for concern, keywords in self.concern_keywords.items():
            for keyword in keywords:
                # Use word boundary checks to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, post_text)
                
                # Add weight based on the number of occurrences
                if matches:
                    # Give more weight to exact matches of critical keywords
                    if concern in ['suicidal', 'self_harm'] and keyword in ['suicide', 'kill myself', 'end my life', 'cut', 'harm', 'hurt myself']:
                        concern_scores[concern] += len(matches) * 2.0  # Increased weight
                    else:
                        concern_scores[concern] += len(matches)
        
        # Check for phrases that indicate severity
        severity_phrases = [
            "can't take it anymore", "no reason to live", "better off dead", 
            "want to die", "don't want to live", "hate myself",
            "always anxious", "constant worry", "panic attack",
            "completely hopeless", "extremely depressed", "severely depressed",
            "life is pointless", "no hope", "giving up", "lost all hope",
            "can't go on", "too much to bear", "unbearable pain"
        ]
        
        for phrase in severity_phrases:
            if phrase in post_text:
                # Identify which concern this phrase relates to
                if phrase in ["can't take it anymore", "no reason to live", "better off dead", "want to die", "don't want to live", "life is pointless", "no hope", "giving up", "lost all hope", "can't go on"]:
                    concern_scores['suicidal'] += 2.5  # Increased weight
                elif phrase in ["hate myself"]:
                    concern_scores['depression'] += 2.0
                elif phrase in ["always anxious", "constant worry", "panic attack"]:
                    concern_scores['anxiety'] += 2.0
                elif phrase in ["completely hopeless", "extremely depressed", "severely depressed", "too much to bear", "unbearable pain"]:
                    concern_scores['depression'] += 2.5
        
        # Check for context indicators
        context_indicators = {
            'depression': ["for weeks", "for months", "every day", "all the time", "can't feel", "no joy", "feeling empty", "nothing matters"],
            'anxiety': ["constantly", "all the time", "can't stop", "overwhelming", "terrified", "panic", "dread"],
            'insomnia': ["can't sleep", "awake all night", "haven't slept", "no sleep", "insomnia"],
            'substance_abuse': ["need it", "can't stop", "withdrawal", "addicted", "dependency", "relying on"]
        }
        
        for concern, indicators in context_indicators.items():
            for indicator in indicators:
                if indicator in post_text:
                    concern_scores[concern] += 1.5  # Increased weight
        
        # Special case for direct mention of suicidal thoughts
        if "suicidal thought" in post_text or "suicidal ideation" in post_text:
            concern_scores['suicidal'] = 1.0  # Maximum score
        
        # Normalize scores but ensure critical concerns maintain high values
        max_score = max(concern_scores.values()) if concern_scores.values() else 0
        if max_score > 0:
            for concern in concern_scores:
                # For suicidal and self-harm, keep scores high
                if concern in ['suicidal', 'self_harm'] and concern_scores[concern] > 0.4:
                    # Keep it high, minimum 0.7
                    concern_scores[concern] = max(concern_scores[concern] / max_score, 0.7)
                else:
                    concern_scores[concern] /= max_score
                
                # Cap at 1.0
                concern_scores[concern] = min(concern_scores[concern], 1.0)
        
        return concern_scores
    
    def analyze_post(self, post_text):
        """
        Analyze a social media post for mental health concerns.
        Tries to use OpenAI first, falls back to keyword-based analysis if that fails.
        
        Args:
            post_text (str): Text of the post
            
        Returns:
            dict: Dictionary of concerns and their confidence scores
        """
        # Try OpenAI analysis first
        return self.analyze_post_with_openai(post_text)
    
    def get_risk_level(self, concern_scores):
        """
        Determine the risk level based on concern scores.
        
        Args:
            concern_scores (dict): Dictionary of concerns and their scores
            
        Returns:
            str: Risk level (Low, Moderate, High)
        """
        # Direct check for suicidal or self-harm mentions - these are always high risk
        suicidal_score = concern_scores.get('suicidal', 0)
        self_harm_score = concern_scores.get('self_harm', 0)
        
        # Any significant mention of suicide or self-harm is high risk
        # Lowered threshold to catch more potential cases
        if suicidal_score > 0.2 or self_harm_score > 0.3:
            return "High"
        
        # Check for substance abuse as a secondary concern
        substance_score = concern_scores.get('substance_abuse', 0)
        
        # Check for depression and anxiety scores
        depression_score = concern_scores.get('depression', 0)
        anxiety_score = concern_scores.get('anxiety', 0)
        
        # Calculate weighted average score with more weight on critical concerns
        weighted_scores = [
            suicidal_score * 5,  # Increased weight
            self_harm_score * 4,  # Increased weight
            substance_score * 2,
            depression_score * 1.5,
            anxiety_score * 1.5
        ]
        
        weights = [5, 4, 2, 1.5, 1.5]  # Updated weights
        valid_weights = sum(weights) if any(weighted_scores) else 1
        weighted_avg = sum(weighted_scores) / valid_weights
        
        # Calculate regular average as a backup
        avg_score = sum(concern_scores.values()) / len(concern_scores) if concern_scores else 0
        
        # Use the higher of the two averages
        final_score = max(weighted_avg, avg_score)
        
        # Adjusted thresholds to be more sensitive
        if final_score > 0.35:  # Lowered from 0.5
            return "High"
        elif final_score > 0.15:  # Lowered from 0.2
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

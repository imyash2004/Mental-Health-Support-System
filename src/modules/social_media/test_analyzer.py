import os
from analyzer import SocialMediaAnalyzer

def test_analyze_post():
    analyzer = SocialMediaAnalyzer()
    sample_post = "I feel so hopeless and depressed. I don't want to live anymore."
    results = analyzer.analyze_post(sample_post)
    print("Analysis Results:", results)

if __name__ == "__main__":
    test_analyze_post()

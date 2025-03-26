# AI-Powered Mental Health Support System

An interactive platform that provides mental health support through AI-powered tools and resources.

## Overview

This system aims to address the growing mental health crisis by providing accessible, AI-powered support to individuals who may not have immediate access to professional mental health services. It serves as both an early detection system and a supportive resource for those experiencing mental health challenges.

## Features

### Self-Assessment Module
- Interactive questionnaires for general mental health, anxiety, and depression
- NLP-based analysis of user responses
- Scoring and personalized recommendations based on assessment results

### Chatbot Module
- AI-powered conversational support
- Intent recognition and sentiment analysis
- Empathetic responses based on user's emotional state
- Emergency detection and crisis resources

### User Dashboard
- Visualization of assessment results over time
- Activity tracking and progress monitoring
- Personalized resource recommendations

### Social Media Analysis Module
- Analysis of social media posts for mental health concerns
- Risk level assessment (Low, Moderate, High)
- Personalized recommendations based on detected concerns
- Sample and custom text analysis capabilities

### Resource Library
- Curated mental health resources
- Articles, videos, and tools
- Crisis support information

## Technical Details

### Technologies Used
- **Frontend**: Streamlit
- **Backend**: Python
- **NLP & ML**: Transformers (Hugging Face), NLTK, spaCy
- **Data Storage**: SQLite
- **Visualization**: Plotly, Matplotlib

### Architecture
The system follows a modular architecture with clear separation of concerns:
- Each module (self-assessment, chatbot, dashboard, social media) is implemented as a separate component
- Database layer handles data storage and retrieval
- Utility functions provide common NLP operations

## Getting Started

### Prerequisites
- Python 3.6+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/mental-health-support-system.git
cd mental-health-support-system
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Initialize the database with sample data (optional)
```
python run.py --init-db
```

4. Run the application
```
python run.py
```

### Running in Simple Mode

If you don't have all the NLP dependencies installed, you can run the application in simple mode:

```
python run.py --simple
```

This will run the application without the NLP features, using mock data instead.

## Usage

1. **Self-Assessment**: Take a questionnaire to evaluate your mental health status
   - Choose from general, anxiety, or depression assessments
   - Receive a score, risk level, and personalized recommendations

2. **Chat Support**: Interact with the AI chatbot for guidance and support
   - Discuss your feelings and concerns
   - Receive empathetic responses and helpful resources
   - Emergency detection for crisis situations

3. **Dashboard**: View your assessment results and track your progress
   - See your assessment history over time
   - Track your activity and engagement
   - Access personalized resource recommendations

4. **Social Media Analysis**: Analyze social media posts for mental health concerns
   - View sample analyses from the provided dataset
   - Analyze your own text for potential concerns
   - Receive risk assessment and recommendations

5. **Resources**: Access curated mental health resources and educational materials

## Project Structure

```
mental-health-support-system/
├── data/                      # Database and data files
├── src/                       # Source code
│   ├── app.py                 # Main application file
│   ├── init_db.py             # Database initialization script
│   ├── modules/               # Application modules
│   │   ├── chatbot/           # Chatbot module
│   │   ├── dashboard/         # Dashboard module
│   │   ├── database/          # Database module
│   │   ├── self_assessment/   # Self-assessment module
│   │   ├── social_media/      # Social media analysis module
│   │   └── utils/             # Utility functions
├── requirements.txt           # Project dependencies
└── run.py                     # Application runner script
```

## Ethical Considerations

This system is designed with the following ethical principles in mind:
- **Privacy**: User data is stored securely and anonymized
- **Transparency**: Clear explanations of how AI recommendations are generated
- **Support, Not Replace**: The system is intended to supplement, not replace, professional mental health care
- **Crisis Response**: Clear protocols for directing users to emergency services when needed

## Limitations

- The AI models used in this system have limitations and may not accurately assess all mental health conditions
- The system is not a replacement for professional mental health care
- The effectiveness of the system depends on the accuracy and honesty of user inputs
- The social media analysis uses a simple keyword-based approach for demonstration purposes
- When deep learning frameworks (PyTorch, TensorFlow, or Flax) are not available, the system falls back to rule-based approaches for NLP operations

## Running Without Deep Learning Frameworks

The system is designed to work even when deep learning frameworks are not available:

- **Simple Mode**: Run with `--simple` flag to use rule-based approaches instead of deep learning models
- **Fallback Mechanisms**: Automatic fallback to rule-based sentiment analysis when transformers library or deep learning frameworks are not available
- **Graceful Degradation**: Core functionality remains available even without advanced NLP capabilities

## Future Enhancements

- Integration with social media APIs for real-time analysis
- Appointment scheduling with mental health professionals
- Mobile application for improved accessibility
- Multi-language support
- Advanced NLP models fine-tuned for mental health language
- User feedback mechanism for continuous improvement

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mental health resources and assessment tools adapted from established mental health organizations
- NLP models provided by Hugging Face Transformers
- Streamlit for the interactive web interface

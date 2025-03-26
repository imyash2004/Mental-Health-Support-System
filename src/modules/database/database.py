import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

class Database:
    """
    Database class for handling data storage and retrieval.
    Initially using SQLite for simplicity, can be extended to use Firebase or other databases.
    """
    
    def __init__(self, db_path="data/mental_health.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Initialize database tables
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        
        # Users table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Assessment results table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            assessment_type TEXT NOT NULL,
            score REAL,
            responses TEXT,
            recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Chat history table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_message TEXT,
            bot_message TEXT,
            sentiment_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Resources table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS resources (
            resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            category TEXT,
            url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Commit the changes
        self.conn.commit()
    
    def get_or_create_user(self, username):
        """
        Get a user by username or create if not exists.
        
        Args:
            username (str): Username to get or create
            
        Returns:
            int: User ID
        """
        self.cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def save_assessment_result(self, user_id, assessment_type, score, responses, recommendations):
        """
        Save assessment result to the database.
        
        Args:
            user_id (int): User ID
            assessment_type (str): Type of assessment
            score (float): Assessment score
            responses (dict): User responses
            recommendations (list): Recommendations based on assessment
            
        Returns:
            int: Result ID
        """
        responses_json = json.dumps(responses)
        recommendations_json = json.dumps(recommendations)
        
        self.cursor.execute(
            "INSERT INTO assessment_results (user_id, assessment_type, score, responses, recommendations) VALUES (?, ?, ?, ?, ?)",
            (user_id, assessment_type, score, responses_json, recommendations_json)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def get_assessment_results(self, user_id, limit=5):
        """
        Get assessment results for a user.
        
        Args:
            user_id (int): User ID
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of assessment results
        """
        self.cursor.execute(
            "SELECT result_id, assessment_type, score, responses, recommendations, created_at FROM assessment_results WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        )
        
        results = []
        for row in self.cursor.fetchall():
            result = {
                'result_id': row[0],
                'assessment_type': row[1],
                'score': row[2],
                'responses': json.loads(row[3]),
                'recommendations': json.loads(row[4]),
                'created_at': row[5]
            }
            results.append(result)
        
        return results
    
    def save_chat_message(self, user_id, user_message, bot_message, sentiment_score=0.0):
        """
        Save chat message to the database.
        
        Args:
            user_id (int): User ID
            user_message (str): User's message
            bot_message (str): Bot's response
            sentiment_score (float): Sentiment score of the user's message
            
        Returns:
            int: Chat ID
        """
        self.cursor.execute(
            "INSERT INTO chat_history (user_id, user_message, bot_message, sentiment_score) VALUES (?, ?, ?, ?)",
            (user_id, user_message, bot_message, sentiment_score)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def get_chat_history(self, user_id, limit=20):
        """
        Get chat history for a user.
        
        Args:
            user_id (int): User ID
            limit (int): Maximum number of messages to return
            
        Returns:
            list: List of chat messages
        """
        self.cursor.execute(
            "SELECT chat_id, user_message, bot_message, sentiment_score, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at ASC LIMIT ?",
            (user_id, limit)
        )
        
        messages = []
        for row in self.cursor.fetchall():
            message = {
                'chat_id': row[0],
                'user_message': row[1],
                'bot_message': row[2],
                'sentiment_score': row[3],
                'created_at': row[4]
            }
            messages.append(message)
        
        return messages
    
    def add_resource(self, title, description, category, url):
        """
        Add a resource to the database.
        
        Args:
            title (str): Resource title
            description (str): Resource description
            category (str): Resource category
            url (str): Resource URL
            
        Returns:
            int: Resource ID
        """
        self.cursor.execute(
            "INSERT INTO resources (title, description, category, url) VALUES (?, ?, ?, ?)",
            (title, description, category, url)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def get_resources(self, category=None, limit=20):
        """
        Get resources from the database.
        
        Args:
            category (str, optional): Filter by category
            limit (int): Maximum number of resources to return
            
        Returns:
            list: List of resources
        """
        if category:
            self.cursor.execute(
                "SELECT resource_id, title, description, category, url FROM resources WHERE category = ? LIMIT ?",
                (category, limit)
            )
        else:
            self.cursor.execute(
                "SELECT resource_id, title, description, category, url FROM resources LIMIT ?",
                (limit,)
            )
        
        resources = []
        for row in self.cursor.fetchall():
            resource = {
                'resource_id': row[0],
                'title': row[1],
                'description': row[2],
                'category': row[3],
                'url': row[4]
            }
            resources.append(resource)
        
        return resources
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Destructor to ensure the database connection is closed."""
        self.close()

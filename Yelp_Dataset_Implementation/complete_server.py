"""
Complete Restaurant Recommendation System Server (Updated for Real Yelp Dataset)
Combines frontend serving and API endpoints in one Flask app
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class RestaurantRecommendationSystem:
    def __init__(self, business_file='business.json', review_file='review.json', 
                 checkin_file='checkin.json', filter_city=None, max_businesses=500):
        self.businesses_df = None
        self.reviews_df = None
        self.checkins_df = None
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.filter_city = filter_city
        self.max_businesses = max_businesses
        
        self.load_yelp_data(business_file, review_file, checkin_file)
        self.preprocess_data()
        self.build_models()
    
    def safe_parse_dict(self, dict_string):
        """Safely parse dictionary-like strings from Yelp data"""
        if pd.isna(dict_string) or dict_string == 'None' or dict_string is None:
            return {}
        
        try:
            cleaned = str(dict_string).replace("u'", "'").replace('u"', '"')
            cleaned = cleaned.replace("'", '"')
            return ast.literal_eval(cleaned.replace('"', "'"))
        except:
            return {}
    
    def parse_business_attributes(self, attrs_string):
        """Parse business attributes from string to dict"""
        attrs = self.safe_parse_dict(attrs_string)
        parsed = {}
        
        key_attrs = {
            'RestaurantsGoodForGroups': 'GoodForGroups',
            'GoodForKids': 'GoodForKids',
            'OutdoorSeating': 'OutdoorSeating',
            'RestaurantsTakeOut': 'TakeOut',
            'RestaurantsDelivery': 'Delivery',
            'RestaurantsPriceRange2': 'price_range',
            'Alcohol': 'Alcohol',
            'WiFi': 'WiFi',
            'HasTV': 'HasTV'
        }
        
        for yelp_key, our_key in key_attrs.items():
            if yelp_key in attrs:
                val = attrs[yelp_key]
                if val in ['True', 'true', True]:
                    parsed[our_key] = True
                elif val in ['False', 'false', False]:
                    parsed[our_key] = False
                elif yelp_key == 'RestaurantsPriceRange2':
                    try:
                        parsed[our_key] = int(val)
                    except:
                        parsed[our_key] = 2
                else:
                    parsed[our_key] = str(val) if val not in ['None', None] else None
            else:
                if our_key == 'price_range':
                    parsed[our_key] = 2
                else:
                    parsed[our_key] = False
        
        return parsed
    
    def parse_hours(self, hours_dict):
        """Parse business hours from Yelp format"""
        if pd.isna(hours_dict) or hours_dict == 'None' or hours_dict is None:
            return None
        
        try:
            if isinstance(hours_dict, str):
                hours_dict = self.safe_parse_dict(hours_dict)
            
            parsed_hours = {}
            for day, time_range in hours_dict.items():
                if time_range and ':' in str(time_range):
                    parts = str(time_range).split('-')
                    if len(parts) == 2:
                        open_time = parts[0].replace(':', ':').zfill(5)
                        close_time = parts[1].replace(':', ':').zfill(5)
                        parsed_hours[day] = f"{open_time}-{close_time}"
            
            return parsed_hours if parsed_hours else None
        except:
            return None
    
    def load_yelp_data(self, business_file, review_file, checkin_file):
        """Load real Yelp dataset from JSON files"""
        print("Loading Yelp dataset...")
        
        # Load businesses
        print(f"Loading businesses from {business_file}...")
        businesses = []
        with open(business_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    business = json.loads(line.strip())
                    if business.get('categories') and 'Restaurant' in business.get('categories', ''):
                        businesses.append(business)
                        if len(businesses) >= self.max_businesses:
                            break
                except:
                    continue
        
        self.businesses_df = pd.DataFrame(businesses)
        print(f"Loaded {len(self.businesses_df)} restaurants")
        
        if self.filter_city and 'city' in self.businesses_df.columns:
            self.businesses_df = self.businesses_df[
                self.businesses_df['city'].str.lower() == self.filter_city.lower()
            ]
            print(f"Filtered to {len(self.businesses_df)} restaurants in {self.filter_city}")
        
        if len(self.businesses_df) == 0:
            raise ValueError("No restaurants found in dataset!")
        
        self.businesses_df['categories_list'] = self.businesses_df['categories'].apply(
            lambda x: [cat.strip() for cat in str(x).split(',') if cat.strip()] if x else []
        )
        
        print("Parsing business attributes...")
        if 'attributes' in self.businesses_df.columns:
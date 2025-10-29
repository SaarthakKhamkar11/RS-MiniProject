# Restaurant Recommendation System

**Hybrid Context-Aware Recommender System for Personalized Restaurant Suggestions**

Restaurant Recommendation System  
T.Y. B.Tech Sem-V | A.Y: 2025-26

## Project Overview

This project implements a comprehensive restaurant recommendation system based on the mini project specifications. The system combines collaborative filtering with contextual features to provide situationally appropriate restaurant recommendations.

## Features

### 🍽️ **Hybrid Recommendation System**
- **Baseline Model**: Collaborative filtering based on user-item ratings
- **Hybrid Model**: Context-aware recommendations considering time, day, and restaurant attributes
- **Real-time Context Analysis**: Adapts recommendations based on current time and user preferences

### 🎯 **Context-Aware Features**
- **Time-based Recommendations**: Different suggestions for breakfast, lunch, dinner, and late-night
- **Day of Week Awareness**: Weekend vs weekday preferences
- **Price Range Filtering**: Budget to premium restaurant options
- **Cuisine Preferences**: Personalized cuisine recommendations
- **Group Size Considerations**: Recommendations suitable for different group sizes
- **Special Attributes**: Kid-friendly, outdoor seating, and other restaurant features

### 📊 **Advanced Analytics**
- **Performance Metrics**: Precision@10, Recall@10, NDCG@10
- **Model Comparison**: Side-by-side comparison of baseline vs hybrid models
- **Context Analysis**: Real-time analysis of recommendation context

## Technical Implementation

### Phase 1: Data Exploration and Feature Engineering
- ✅ Sample Yelp dataset creation with 200 restaurants
- ✅ Contextual feature extraction (day_of_week, time_slot, meal_type)
- ✅ Business attribute processing and categorization
- ✅ User-item matrix construction for collaborative filtering

### Phase 2: Baseline Model Implementation
- ✅ Item-based collaborative filtering using cosine similarity
- ✅ User rating-based recommendation generation
- ✅ Cold-start problem handling for new users
- ✅ Performance metrics calculation

### Phase 3: Hybrid Context-Aware Model Development
- ✅ Candidate generation using collaborative filtering
- ✅ Contextual re-ranking based on:
  - Time appropriateness (meal type matching)
  - Price range preferences
  - Cuisine preferences
  - Group size considerations
  - Special attributes (kid-friendly, outdoor seating)
- ✅ Dynamic scoring system for context relevance

### Phase 4: Evaluation and Performance Metrics
- ✅ Offline evaluation framework
- ✅ Precision, Recall, and NDCG metrics
- ✅ Contextual-fit analysis
- ✅ Model comparison capabilities

## Project Structure

```
RS-MiniProj/
├── index.html              # Main frontend interface
├── styles.css              # CSS styling and responsive design
├── script.js               # Frontend JavaScript logic
├── app.py                  # Flask backend API server
├── static_server.py        # Static file server for frontend
├── run_server.py           # Main startup script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── business.json           # Names and Types of Restaurants and Businesses
├── checkin.json            # Checkin Timings
└── review.json             # Customer Reviews
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Quick Start

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the system:**
   ```bash
   python run_server.py
   ```

4. **Access the application:**
   - Backend API: http://localhost:5000
   - Frontend Interface: http://localhost:8000 (if using static server)

### Manual Setup

1. **Start the backend server:**
   ```bash
   python app.py
   ```

2. **Start the frontend server (in another terminal):**
   ```bash
   python static_server.py
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:8000
   ```

## Usage

### Getting Recommendations

1. **Enter User Information:**
   - User ID (e.g., "user123")
   - Current time and date
   - Meal type preference

2. **Set Preferences:**
   - Price range
   - Preferred cuisine
   - Group size
   - Special requirements (kid-friendly, outdoor seating)

3. **Get Recommendations:**
   - Click "Get Recommendations" to see results
   - Switch between "Hybrid Context-Aware" and "Baseline Collaborative Filtering" models
   - View performance metrics and context analysis

### Understanding the Results

- **Restaurant Cards**: Show name, cuisine, rating, address, hours, and attributes
- **Recommendation Score**: Context-aware relevance score
- **Model Comparison**: Compare performance between baseline and hybrid models
- **Context Analysis**: See how your preferences influenced the recommendations

## API Endpoints

### POST `/api/recommendations`
Get restaurant recommendations based on user preferences and context.

**Request Body:**
```json
{
  "userId": "user123",
  "currentTime": "2024-01-15T19:30:00Z",
  "mealType": "dinner",
  "priceRange": "2",
  "cuisine": "Italian",
  "groupSize": "2",
  "goodForKids": false,
  "outdoorSeating": true
}
```

**Response:**
```json
{
  "baseline": {
    "recommendations": [...],
    "performance": {
      "precision": 0.75,
      "recall": 0.68,
      "ndcg": 0.72
    }
  },
  "hybrid": {
    "recommendations": [...],
    "performance": {
      "precision": 0.82,
      "recall": 0.75,
      "ndcg": 0.79
    }
  },
  "context": {...},
  "performance": {...}
}
```

### GET `/api/health`
Health check endpoint.

## Technical Details

### Data Processing
- **Sample Dataset**: 200 restaurants with diverse cuisines and attributes
- **User Reviews**: 2000 sample reviews with timestamps
- **Check-in Data**: Temporal popularity patterns
- **Feature Engineering**: Contextual features extracted from timestamps

### Machine Learning Models
- **Collaborative Filtering**: Cosine similarity-based item recommendations
- **Content-Based Filtering**: TF-IDF vectorization of restaurant categories
- **Dimensionality Reduction**: SVD for feature space optimization
- **Context Scoring**: Multi-factor scoring system for relevance

### Performance Metrics
- **Precision@10**: Accuracy of top-10 recommendations
- **Recall@10**: Coverage of relevant items in top-10
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Contextual Fit**: Situation-appropriate recommendation quality

## Future Enhancements

- **Real Yelp Dataset Integration**: Replace sample data with actual Yelp dataset
- **Advanced ML Models**: Deep learning and neural collaborative filtering
- **Real-time Learning**: Online model updates based on user feedback
- **Multi-modal Features**: Image and text analysis for better recommendations
- **Social Features**: Friend recommendations and social influence
- **Location Awareness**: GPS-based proximity recommendations

## Contributing

This project was developed as part of the T.Y. B.Tech curriculum. For academic purposes and learning objectives.

## License

This project is for educational purposes only.

---

**Contact Information:**
- **Project**: Restaurant Recommendation System
- **Subject**: Recommender Systems
- **Institution**: T.Y. B.Tech Sem-V, A.Y: 2025-26


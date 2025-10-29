// Restaurant Recommendation System - Frontend JavaScript


class RestaurantRecommendationSystem {
    constructor() {
        this.currentModel = 'hybrid';
        this.currentRecommendations = null;
        this.initializeEventListeners();
        this.setDefaultTime();
    }

    initializeEventListeners() {
        // Form submission
        document.getElementById('preferenceForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.getRecommendations();
        });

        // Model tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchModel(e.target.dataset.model);
            });
        });
    }

    setDefaultTime() {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        
        const datetimeString = `${year}-${month}-${day}T${hours}:${minutes}`;
        document.getElementById('currentTime').value = datetimeString;
    }

    async getRecommendations() {
        const formData = this.collectFormData();
        
        if (!this.validateFormData(formData)) {
            return;
        }

        this.showLoading(true);
        
        try {
            const response = await fetch('http://localhost:5000/api/recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.currentRecommendations = data;
            this.displayRecommendations(data);
            this.showResults();
            
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            this.showError('Failed to get recommendations. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    collectFormData() {
        const currentTime = new Date(document.getElementById('currentTime').value);
        
        return {
            userId: document.getElementById('userId').value,
            currentTime: currentTime.toISOString(),
            mealType: document.getElementById('mealType').value,
            priceRange: document.getElementById('priceRange').value,
            cuisine: document.getElementById('cuisine').value,
            groupSize: document.getElementById('groupSize').value,
            goodForKids: document.getElementById('goodForKids').checked,
            outdoorSeating: document.getElementById('outdoorSeating').checked,
            context: {
                dayOfWeek: currentTime.getDay(),
                hour: currentTime.getHours(),
                isWeekend: currentTime.getDay() === 0 || currentTime.getDay() === 6
            }
        };
    }

    validateFormData(data) {
        if (!data.userId.trim()) {
            this.showError('Please enter a User ID');
            return false;
        }
        
        if (!data.mealType) {
            this.showError('Please select a meal type');
            return false;
        }
        
        return true;
    }

    displayRecommendations(data) {
        const container = document.getElementById('resultsContainer');
        const modelData = data[this.currentModel];
        
        if (!modelData || !modelData.recommendations) {
            container.innerHTML = '<p>No recommendations available for the selected model.</p>';
            return;
        }

        const recommendations = modelData.recommendations;
        
        container.innerHTML = recommendations.map((restaurant, index) => `
            <div class="restaurant-card fade-in" style="animation-delay: ${index * 0.1}s">
                <div class="restaurant-header">
                    <div>
                        <div class="restaurant-name">${restaurant.name}</div>
                        <div class="restaurant-cuisine">${restaurant.categories.join(', ')}</div>
                    </div>
                    <div class="restaurant-rating">
                        <i class="fas fa-star"></i>
                        <span>${restaurant.stars.toFixed(1)}</span>
                    </div>
                </div>
                
                <div class="restaurant-attributes">
                    ${restaurant.attributes.map(attr => 
                        `<span class="attribute-tag">${attr}</span>`
                    ).join('')}
                </div>
                
                <div class="restaurant-address">
                    <i class="fas fa-map-marker-alt"></i> ${restaurant.address}
                </div>
                
                <div class="restaurant-hours">
                    <i class="fas fa-clock"></i> ${restaurant.hours}
                </div>
                
            </div>
        `).join('');
    }


    switchModel(model) {
        this.currentModel = model;
        
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-model="${model}"]`).classList.add('active');
        
        // Update model description
        const description = document.getElementById('modelDescription');
        if (model === 'hybrid') {
            description.innerHTML = `
                <strong>Hybrid Context-Aware Model:</strong> Combines collaborative filtering with contextual features like time, day, and restaurant attributes to provide situationally appropriate recommendations.
            `;
        } else {
            description.innerHTML = `
                <strong>Baseline Collaborative Filtering:</strong> Uses user-item rating similarity to generate recommendations without considering contextual factors.
            `;
        }
        
        // Update recommendations display
        if (this.currentRecommendations) {
            this.displayRecommendations(this.currentRecommendations);
        }
    }

    showResults() {
        document.getElementById('resultsSection').style.display = 'block';
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth' 
        });
    }

    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'block' : 'none';
    }

    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #e74c3c;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i> ${message}
        `;
        
        document.body.appendChild(errorDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RestaurantRecommendationSystem();
});

// Add CSS for error notification animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .error-notification {
        font-weight: 600;
    }
    
    .error-notification i {
        margin-right: 10px;
    }
`;
document.head.appendChild(style);

# app.py - Main Flask API file
# Save this as 'app.py' in your sms-api folder

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
from datetime import datetime

print("🚀 Starting SMS Categorization API...")

app = Flask(__name__)
CORS(app)  # Allow Flutter app to access this API

# Load your trained model and vectorizer
try:
    print("📁 Loading trained model...")
    model = joblib.load('sms_categorization_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

def preprocess_text(text):
    """Clean SMS text same way as training"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_amount_from_sms(sms_text):
    """Extract money amount from SMS"""
    patterns = [
        r'(?:rs\.?|inr|₹)\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:rs\.?|inr|₹)',
        r'amount\s*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'debited\s*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'spent\s*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sms_text.lower())
        if match:
            amount = match.group(1).replace(',', '')
            return float(amount)
    return 0.0

def extract_merchant_from_sms(sms_text):
    """Extract store/merchant name from SMS"""
    # Common merchant patterns
    merchants = ['swiggy', 'zomato', 'amazon', 'flipkart', 'netflix', 'uber', 'ola', 
                'pizza hut', 'dominos', 'mcdonald', 'starbucks', 'big bazaar', 'dmart']
    
    for merchant in merchants:
        if merchant in sms_text.lower():
            return merchant.title()
    
    return "Unknown Merchant"

@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        'message': '🤖 SMS Categorization API is running!',
        'status': 'healthy',
        'model_loaded': model is not None,
        'categories': ['Essentials', 'Emergency', 'Impulse'],
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/categorize', methods=['POST'])
def categorize_sms():
    """Main endpoint - categorize SMS"""
    try:
        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded properly',
                'success': False
            }), 500
        
        # Get SMS text from request
        data = request.get_json()
        
        if not data or 'sms_text' not in data:
            return jsonify({
                'error': 'Please provide sms_text in JSON',
                'success': False,
                'example': {'sms_text': 'Rs 200 spent on Swiggy order'}
            }), 400
        
        sms_text = data['sms_text']
        print(f"📱 Processing SMS: {sms_text[:50]}...")
        
        # Preprocess the text
        processed_text = preprocess_text(sms_text)
        
        # Transform to numbers using vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Predict category
        prediction = model.predict(text_vector)[0]
        
        # Get confidence score
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities)
        
        # Extract amount and merchant
        amount = extract_amount_from_sms(sms_text)
        merchant = extract_merchant_from_sms(sms_text)
        
        # Prepare response
        result = {
            'success': True,
            'category': prediction,
            'confidence': round(confidence, 2),
            'amount': amount,
            'merchant': merchant,
            'original_text': sms_text,
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Result: {prediction} (confidence: {confidence:.2f})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/test', methods=['GET'])
def test_api():
    """Test endpoint with sample data"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    test_sms = [
        "Rs 200 spent on Pizza Hut order",
        "Rs 50 paid for bus fare", 
        "Rs 1500 emergency doctor fees"
    ]
    
    results = []
    for sms in test_sms:
        try:
            processed = preprocess_text(sms)
            vector = vectorizer.transform([processed])
            prediction = model.predict(vector)[0]
            confidence = max(model.predict_proba(vector)[0])
            
            results.append({
                'sms': sms,
                'category': prediction,
                'confidence': round(confidence, 2)
            })
        except Exception as e:
            results.append({
                'sms': sms,
                'error': str(e)
            })
    
    return jsonify({
        'test_results': results,
        'model_status': 'working'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Starting API on port {port}")
    print("📱 Ready to categorize SMS messages!")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=True)

import joblib
import pickle

# Load the model and vectorizer
model = joblib.load('mental_health_model.pkl')

def predict_mood_with_safety(text):
    """Predict mood with safety check"""
    predicted_status = model.predict([text])[0]
    
    # Safety Check
    if predicted_status == "Suicidal":
        return "CRITICAL", "I'm concerned about what you're sharing. Please know that you're not alone. You can reach out to a professional or a helpline for immediate support. I am here to listen, but a professional can help you best right now."
    
    return predicted_status, None

def get_empathetic_response(mood):
    """Get empathetic response based on mood"""
    responses = {
        "Normal": {
            "message": "It's great to hear you're doing okay! Consistency is key to maintaining your well-being.",
            "tip": "Try journaling for 5 minutes today to keep track of your positive thoughts."
        },
        "Anxiety": {
            "message": "I can feel that things might be a bit overwhelming right now. You are doing your best, and that is enough.",
            "tip": "Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you can touch, 3 you hear, 2 you can smell, and 1 you can taste."
        },
        "Stress": {
            "message": "Student life can be a lot to handle. Remember to take it one step at a time.",
            "tip": "Try a 'Box Breathing' exercise: Inhale for 4 seconds, hold for 4, exhale for 4, and hold for 4. Repeat 3 times."
        },
        "Depression": {
            "message": "I'm sorry things feel so heavy right now. Please be gentle with yourself today.",
            "tip": "Try to step outside for just 5 minutes of fresh air, or reach out to one trusted friend just to say hello."
        },
        "Suicidal": {
            "message": "I'm really concerned about what you're sharing. Your feelings are valid and important.",
            "tip": "Please reach out to a crisis helpline or speak to a trusted adult immediately. You're not alone."
        }
    }
    
    # Default response
    default = {
        "message": "I am here to listen and support you through whatever you are going through.",
        "tip": "Sometimes a short walk or a glass of water can help reset your focus."
    }
    
    return responses.get(mood, default)

def chat_with_companion(user_input):
    """Main chat function"""
    mood, safety_msg = predict_mood_with_safety(user_input)
    
    if safety_msg:
        return safety_msg
    
    response_data = get_empathetic_response(mood)
    return f"{response_data['message']}\n\nTip: {response_data['tip']}"

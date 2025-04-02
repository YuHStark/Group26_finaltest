from flask import Flask, request, jsonify
import os
import json
import traceback
from recommendation_engine import RecommendationEngine

app = Flask(__name__)

# Initialize the recommendation engine with data path and TF-IDF paths
processed_books_path = os.environ.get('PROCESSED_BOOKS_PATH', 'processed_data/processed_books.csv')
tfidf_matrix_path = os.environ.get('TFIDF_MATRIX_PATH', 'processed_data/tfidf_matrix.npz')
tfidf_vectorizer_path = os.environ.get('TFIDF_VECTORIZER_PATH', 'processed_data/tfidf_vectorizer.pkl')

# Initialize engine with appropriate paths
engine = RecommendationEngine(
    processed_data_path=processed_books_path,
    tfidf_matrix_path=tfidf_matrix_path,
    vectorizer_path=tfidf_vectorizer_path
)

# Mapping functions for converting natural language to parameters
def map_rating_level(rating_level):
    """Map rating level descriptions to numerical values."""
    if not rating_level:
        return 0  # No preference

    if isinstance(rating_level, str):
        rating_level = rating_level.lower()
        
    if rating_level in ["high rating", "good rating", "excellent", "high", "good"]:
        return 4.0
    elif rating_level in ["average", "medium", "decent"]:
        return 3.0
    
    return 0  # Default: no rating filter

def map_length_level(length_level):
    """Map length level descriptions to page count values."""
    if not length_level:
        return None  # No preference
    
    if isinstance(length_level, str):
        length_level = length_level.lower()
    
    if length_level in ["short", "brief", "quick read"]:
        return 300  # Maximum pages for short books
    elif length_level in ["medium", "average length", "moderate"]:
        return 500  # Maximum pages for medium books
    elif length_level in ["long", "lengthy", "epic"]:
        return None  # No maximum for long books (minimum would be 500)
    
    return None  # Default: no page limit

def get_book_from_context(contexts):
    """Extract book title from conversation context."""
    if not contexts:
        return None
        
    for context in contexts:
        if 'parameters' in context and 'book_title' in context['parameters']:
            return context['parameters']['book_title']
    return None

@app.route('/', methods=['GET'])
def index():
    """Basic endpoint to verify the API is running."""
    return jsonify({
        'status': 'online',
        'message': 'Book Recommendation API is running',
        'version': '1.0.0'
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle webhook requests from Dialogflow."""
    req = request.get_json(silent=True, force=True)
    
    # Log the incoming request for debugging (optional)
    print("Request from Dialogflow:")
    print(json.dumps(req, indent=2))
    
    try:
        # Extract the intent and parameters
        query_result = req.get('queryResult', {})
        intent = query_result.get('intent', {}).get('displayName', '')
        parameters = query_result.get('parameters', {})
        
        print(f"Processing intent: {intent}")
        
        # Route to appropriate handler based on intent
        if intent == 'Collect_Book_Genre':
            return handle_collect_genre(parameters)
        elif intent == 'Collect_Book_Style':
            return handle_collect_style(parameters)
        elif intent == 'Collect_Rating_Preference':
            return handle_collect_rating(parameters)
        elif intent == 'Collect_Book_Length':
            return handle_collect_length(parameters)
        elif intent == 'Request_Book_Details':
            return handle_book_details(parameters)
        elif intent == 'Request_Similar_Books':
            return handle_similar_books(query_result)
        elif intent == 'Request_New_Conversation':
            return handle_new_conversation()
        elif intent == 'Goodbye':
            return handle_goodbye()
        else:
            # Default response for unknown intents
            return jsonify({
                'fulfillmentText': 'I\'m not sure how to help with that. Would you like a book recommendation?'
            })
    
    except Exception as e:
        print(f"Error processing request: {e}")
        # Print full stack trace for debugging
        traceback.print_exc()
        return jsonify({
            'fulfillmentText': 'Sorry, I encountered an error processing your request. Please try again.'
        })

def handle_collect_genre(parameters):
    """Handle the first step of conversation - collecting genre preference."""
    genre = parameters.get('genre', '')
    
    if not genre:
        return jsonify({
            'fulfillmentText': 'What genre of books are you interested in?'
        })
    
    # Acknowledge genre and ask for style preference
    return jsonify({
        'fulfillmentText': f'Great! I\'ll look for {genre} books. What kind of style do you prefer? For example: humorous, adventure, serious, etc.'
    })

def handle_collect_style(parameters):
    """Handle the second step - collecting style preference."""
    style = parameters.get('style', '')
    
    if not style:
        return jsonify({
            'fulfillmentText': 'What style of books do you prefer? For example: humorous, adventure, serious, etc.'
        })
    
    # Acknowledge style and ask for rating preference
    return jsonify({
        'fulfillmentText': f'I\'ll look for books with a {style} style. Do you have a preference for ratings? For example, would you like books rated 4.0 and above?'
    })

def handle_collect_rating(parameters):
    """Handle the third step - collecting rating preference."""
    rating_level = parameters.get('rating_level', '')
    
    if not rating_level:
        return jsonify({
            'fulfillmentText': 'Do you have a preference for book ratings? For instance, would you like high-rated books (4.0+), average-rated, or no preference?'
        })
    
    # Map rating text to actual value for confirmation
    rating_value = map_rating_level(rating_level)
    rating_text = "highly-rated" if rating_value >= 4.0 else "moderately-rated" if rating_value >= 3.0 else "any rating"
    
    # Acknowledge rating and ask for length preference
    return jsonify({
        'fulfillmentText': f'I\'ll look for {rating_text} books. Do you have a preference for book length? For example: short (under 300 pages), medium (300-500 pages), or long (over 500 pages)?'
    })

def handle_collect_length(parameters):
    """Handle the fourth step - collecting length preference and providing recommendations."""
    # Extract all parameters from the request
    genre = parameters.get('genre', '')
    style = parameters.get('style', '')
    rating_level = parameters.get('rating_level', '')
    length_level = parameters.get('length_level', '')
    
    if not length_level:
        return jsonify({
            'fulfillmentText': 'What length of book do you prefer? Short (under 300 pages), medium (300-500 pages), long (over 500 pages), or no preference?'
        })
    
    # Map parameters to values
    min_rating = map_rating_level(rating_level)
    max_pages = map_length_level(length_level)
    
    # Get recommendations using ensemble method
    recommendations = engine.ensemble_recommendations(
        genre=genre, 
        style=style, 
        min_rating=min_rating, 
        max_pages=max_pages, 
        top_n=3
    )
    
    if len(recommendations) == 0:
        return jsonify({
            'fulfillmentText': 'I couldn\'t find any books matching your preferences. Would you like to try with different criteria?'
        })
    
    # Format response
    response_text = f"Based on your preferences for {genre} books with {style} style, here are my recommendations:\n\n"
    
    for _, book in recommendations.iterrows():
        response_text += f"• {book['book_title']} – by {book['book_authors']}, "
        response_text += f"rating: {book['book_rating']:.1f}"
        
        # Add page count if available
        if 'book_pages' in book and book['book_pages'] > 0:
            response_text += f", {book['book_pages']} pages"
        
        response_text += "\n"
    
    response_text += "\nWould you like to know more details about any of these books?"
    
    return jsonify({
        'fulfillmentText': response_text
    })

def handle_book_details(parameters):
    """Handle request for detailed information about a specific book."""
    book_title = parameters.get('book_title', '')
    
    if not book_title:
        return jsonify({
            'fulfillmentText': 'Which book would you like to know more about?'
        })
    
    book_details = engine.get_book_details(book_title)
    
    if not book_details:
        return jsonify({
            'fulfillmentText': f'I couldn\'t find details for a book titled "{book_title}". Could you check the spelling or try another book?'
        })
    
    # Format detailed response
    response_text = f"Here are the details about '{book_details['title']}':\n\n"
    
    # Add description if available
    if book_details['description']:
        # Truncate long descriptions
        desc = book_details['description']
        if len(desc) > 500:
            desc = desc[:500] + "..."
        response_text += f"【Description】 {desc}\n\n"
    
    response_text += f"【Author】 {book_details['author']}\n"
    response_text += f"【Genre】 {book_details['genres']}\n"
    response_text += f"【Rating】 {book_details['rating']:.1f}"
    
    # Add rating count if available
    if book_details['rating_count'] > 0:
        response_text += f" (based on {book_details['rating_count']} reviews)"
    
    response_text += "\n"
    
    # Add page count if available
    if book_details['pages'] > 0:
        response_text += f"【Pages】 {book_details['pages']}\n"
    
    # Add format if available
    if book_details['format'] != 'Unknown':
        response_text += f"【Format】 {book_details['format']}\n"
    
    response_text += "\nWould you like to find books similar to this one?"
    
    return jsonify({
        'fulfillmentText': response_text
    })

def handle_similar_books(query_result):
    """Handle request for books similar to a specific title."""
    parameters = query_result.get('parameters', {})
    book_title = parameters.get('book_title', '')
    
    # If no book title in parameters, try to get from context
    if not book_title:
        book_title = get_book_from_context(query_result.get('outputContexts', []))
    
    if not book_title:
        return jsonify({
            'fulfillmentText': 'Which book would you like to find similar titles for?'
        })
    
    similar_books = engine.find_similar_books_knn(book_title, n=3)
    
    if len(similar_books) == 0:
        return jsonify({
            'fulfillmentText': f'I couldn\'t find any books similar to "{book_title}". Would you like to try with a different book?'
        })
    
    # Format response
    response_text = f"Here are some books similar to '{book_title}':\n\n"
    
    for _, book in similar_books.iterrows():
        response_text += f"• {book['book_title']} – by {book['book_authors']}, "
        response_text += f"rating: {book['book_rating']:.1f}\n"
    
    response_text += "\nWould you like me to recommend other types of books? (Yes to start a new recommendation, No to end our conversation)"
    
    return jsonify({
        'fulfillmentText': response_text
    })

def handle_new_conversation():
    """Handle the user's request for a new recommendation (Yes after similar books)."""
    # This clears all previous contexts and starts fresh
    return jsonify({
        'fulfillmentText': 'Great! Let\'s start a new book recommendation. What genre are you interested in this time?'
    })

def handle_goodbye():
    """Handle the end of the conversation (No after similar books)."""
    # This clears all previous contexts and provides a goodbye message
    return jsonify({
        'fulfillmentText': 'Thank you for using our book recommendation service! I hope you found something interesting to read. Feel free to come back anytime for more recommendations. Goodbye!'
    })

@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """Direct API endpoint for book recommendations."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        genre = data.get('genre', '')
        style = data.get('style', '')
        rating_level = data.get('rating_level', '')
        length_level = data.get('length_level', '')
        
        # Map parameters
        min_rating = map_rating_level(rating_level)
        max_pages = map_length_level(length_level)
        
        # Get recommendations
        recommendations = engine.ensemble_recommendations(
            genre=genre,
            style=style,
            min_rating=min_rating,
            max_pages=max_pages,
            top_n=5
        )
        
        # Convert to list of dictionaries for JSON response
        results = []
        for _, book in recommendations.iterrows():
            book_dict = {
                'title': book['book_title'],
                'author': book['book_authors'],
                'rating': float(book['book_rating'])
            }
            
            # Add optional fields if available
            if 'book_pages' in book:
                book_dict['pages'] = int(book['book_pages'])
            if 'genres' in book:
                book_dict['genres'] = book['genres']
            
            results.append(book_dict)
        
        return jsonify({
            'recommendations': results,
            'count': len(results)
        })
        
    except Exception as e:
        print(f"Error in API recommendations: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"Starting Book Recommendation API on {host}:{port}, debug={debug}")
    app.run(host=host, port=port, debug=debug)
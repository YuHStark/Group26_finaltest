import numpy as np
import pandas as pd
import re
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class RecommendationEngine:
    def __init__(self, processed_data_path, tfidf_matrix_path=None, vectorizer_path=None):
        """Initialize the recommendation engine with processed book data and TF-IDF features."""
        self.df = pd.read_csv(processed_data_path)
        self.tfidf_matrix = None
        self.knn_model = None
        
        # Load TF-IDF matrix if path is provided
        if tfidf_matrix_path and os.path.exists(tfidf_matrix_path):
            print(f"Loading TF-IDF matrix from {tfidf_matrix_path}")
            self.tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
        else:
            # Default path based on processed data path
            default_path = os.path.join(os.path.dirname(processed_data_path), "tfidf_matrix.npz")
            if os.path.exists(default_path):
                print(f"Loading TF-IDF matrix from {default_path}")
                self.tfidf_matrix = sparse.load_npz(default_path)
        
        # Load vectorizer if path is provided (mainly for vocabulary)
        if vectorizer_path and os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        
        # Initialize KNN model for similar book recommendations
        if self.tfidf_matrix is not None:
            self._init_knn_model()
    
    def _init_knn_model(self):
        """Initialize the K-nearest neighbors model for finding similar books."""
        print("Initializing KNN model for similar book recommendations...")
        self.knn_model = NearestNeighbors(
            n_neighbors=6,  # One extra for the book itself
            algorithm='auto',
            metric='cosine'
        )
        self.knn_model.fit(self.tfidf_matrix)
        print("KNN model initialized")
    
    def content_based_filter(self, genre=None, style=None, min_rating=0, max_pages=None, top_n=5):
        """
        Filter books based on content criteria.
        
        Args:
            genre (str): The genre to filter for
            style (str): The writing style to look for
            min_rating (float): Minimum rating threshold
            max_pages (int): Maximum number of pages
            top_n (int): Number of books to return
            
        Returns:
            DataFrame containing filtered book recommendations
        """
        filtered_df = self.df.copy()
        
        # Filter by genre if specified
        if genre and genre != "":
            genre_pattern = re.compile(genre, re.IGNORECASE)
            filtered_df = filtered_df[filtered_df['genres'].apply(
                lambda x: bool(genre_pattern.search(str(x))))]
        
        # Filter by minimum rating
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['book_rating'] >= min_rating]
        
        # Filter by maximum page count
        if 'book_pages' in filtered_df.columns:
            if max_pages is None and length_level == "long":
                filtered_df = filtered_df[filtered_df['book_pages'] >= 500]
            # For short and medium books (with max_pages constraint)
            elif max_pages and max_pages > 0:
                filtered_df = filtered_df[filtered_df['book_pages'] <= max_pages]
        
        # Filter by style (text search in description) if specified
        if style and style != "":
            style_pattern = re.compile(style, re.IGNORECASE)
            filtered_df = filtered_df[filtered_df['book_desc'].apply(
                lambda x: bool(style_pattern.search(str(x))))]
        
        # Sort by rating
        filtered_df = filtered_df.sort_values(by='book_rating', ascending=False)
        
        return filtered_df.head(top_n)
    
    def popularity_rank_recommend(self, genre=None, min_rating=0, max_pages=None, top_n=5):
        """
        Recommend books based on popularity score.
        
        Args:
            genre (str): The genre to filter for (optional)
            min_rating (float): Minimum rating threshold
            max_pages (int): Maximum number of pages
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame containing popular book recommendations
        """
        # Calculate popularity score if not already in the dataframe
        if 'popularity_score' not in self.df.columns:
            self.df['popularity_score'] = (
                self.df['book_rating'] * self.df['book_rating_count']
            ) / (self.df['book_rating_count'] + 10)
        
        filtered_df = self.df.copy()
        
        # Apply genre filter if specified
        if genre and genre != "":
            genre_pattern = re.compile(genre, re.IGNORECASE)
            filtered_df = filtered_df[filtered_df['genres'].apply(
                lambda x: bool(genre_pattern.search(str(x))))]
        
        # Apply rating filter
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['book_rating'] >= min_rating]
        
        # Apply page count filter
        if max_pages and max_pages > 0:
            if 'book_pages' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['book_pages'] <= max_pages]
        
        # Sort by popularity score
        filtered_df = filtered_df.sort_values(by='popularity_score', ascending=False)
        
        return filtered_df.head(top_n)
    
    def find_similar_books_knn(self, book_title, n=3):
        """
        Find books similar to the given book title using KNN.
        
        Args:
            book_title (str): Title of the reference book
            n (int): Number of similar books to return
            
        Returns:
            DataFrame containing similar book recommendations
        """
        if self.tfidf_matrix is None or self.knn_model is None:
            # If TF-IDF matrix or KNN model not available, return empty result
            print("Warning: TF-IDF matrix or KNN model not initialized")
            return pd.DataFrame()
        
        # Find the book index
        book_indices = self.df[self.df['book_title'].str.contains(book_title, case=False, na=False)].index
        
        if len(book_indices) == 0:
            return pd.DataFrame()  # Book not found
        
        book_idx = book_indices[0]  # Take the first match
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(self.tfidf_matrix[book_idx].reshape(1, -1))
        
        # Skip the first result as it's the query book itself
        similar_books = self.df.iloc[indices[0][1:n+1]]
        
        return similar_books
    
    def ensemble_recommendations(self, genre=None, style=None, min_rating=0, max_pages=None, top_n=5):
        """
        Combine multiple recommendation algorithms for better results.
        
        Args:
            genre (str): Preferred book genre
            style (str): Preferred writing style
            min_rating (float): Minimum book rating
            max_pages (int): Maximum book pages
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame containing ensemble book recommendations
        """
        # Get recommendations from each algorithm
        content_recs = self.content_based_filter(genre, style, min_rating, max_pages, top_n=top_n*2)
        popularity_recs = self.popularity_rank_recommend(genre, min_rating, max_pages, top_n=top_n*2)
        
        # Combine and deduplicate recommendations
        # Using a scoring system that prioritizes books recommended by multiple algorithms
        all_books = pd.concat([
            content_recs.assign(algorithm='content'),
            popularity_recs.assign(algorithm='popularity')
        ])
        
        # Count occurrences of each book across algorithms
        book_counts = all_books.groupby('book_id').size().reset_index(name='algorithm_count')
        
        # Join with the original recommendations to get all metadata
        scored_recs = all_books.drop_duplicates(subset='book_id').merge(
            book_counts, on='book_id', how='left')
        
        # Sort by count (number of algorithms that recommended it) then by rating
        sort_columns = ['algorithm_count', 'book_rating']
        if 'popularity_score' in scored_recs.columns:
            sort_columns.append('popularity_score')
            
        final_recs = scored_recs.sort_values(
            by=sort_columns, 
            ascending=[False, False, False]
        )
        
        # Select required columns for the result
        result_columns = ['book_id', 'book_title', 'book_authors', 'book_rating']
        if 'book_pages' in final_recs.columns:
            result_columns.append('book_pages')
        if 'genres' in final_recs.columns:
            result_columns.append('genres')
            
        return final_recs.head(top_n)[result_columns]
    
    def get_book_details(self, book_title):
        """
        Get detailed information about a specific book.
        
        Args:
            book_title (str): Title of the book
            
        Returns:
            dict containing book details or None if not found
        """
        book = self.df[self.df['book_title'].str.contains(book_title, case=False, na=False)]
        
        if len(book) == 0:
            return None
        
        # Return the first matching book's details
        book = book.iloc[0]
        
        # Gather all available details
        details = {
            'title': book['book_title'],
            'author': book['book_authors'] if 'book_authors' in book else 'Unknown',
            'description': book['book_desc'] if 'book_desc' in book else '',
            'genres': book['genres'] if 'genres' in book else '',
            'rating': book['book_rating'] if 'book_rating' in book else 0,
            'rating_count': book['book_rating_count'] if 'book_rating_count' in book else 0,
            'pages': book['book_pages'] if 'book_pages' in book else 0,
            'format': book.get('book_format', 'Unknown')
        }
        
        return details

# Example usage
if __name__ == "__main__":
    # Default paths
    processed_data_path = "processed_data/processed_books.csv"
    tfidf_matrix_path = "processed_data/tfidf_matrix.npz"
    vectorizer_path = "processed_data/tfidf_vectorizer.pkl"
    
    engine = RecommendationEngine(
        processed_data_path=processed_data_path,
        tfidf_matrix_path=tfidf_matrix_path,
        vectorizer_path=vectorizer_path
    )
    
    # Example content-based filtering
    sci_fi_books = engine.content_based_filter(genre="Science Fiction", min_rating=4.0, max_pages=300)
    print("\nScience Fiction Books:")
    print(sci_fi_books[['book_title', 'book_authors', 'book_rating']])
    
    # Example popularity ranking
    popular_books = engine.popularity_rank_recommend(min_rating=4.0, max_pages=300)
    print("\nPopular Books:")
    print(popular_books[['book_title', 'book_authors', 'book_rating', 'book_rating_count']])
    
    # Example similar books
    similar_books = engine.find_similar_books_knn("The Hunger Games")
    print("\nBooks similar to 'The Hunger Games':")
    print(similar_books[['book_title', 'book_authors', 'genres']])
    
    # Example ensemble recommendations
    ensemble_recs = engine.ensemble_recommendations(genre="Fantasy", min_rating=4.0, max_pages=400)
    print("\nEnsemble Recommendations for Fantasy books:")
    print(ensemble_recs[['book_title', 'book_authors', 'book_rating']])

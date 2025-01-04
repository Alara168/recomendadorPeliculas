import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import re
from datetime import datetime
import json
import os

class MovieRecommender:
    def __init__(self, data_path):
        """
        Inicializa el sistema de recomendación
        Args:
            data_path (str): Ruta al dataset de películas
        """
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"No se encuentra el archivo {data_path}")

            self.data = pd.read_csv(data_path)
            if self.data.empty:
                raise ValueError("El archivo CSV está vacío")

            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.lemmatizer = WordNetLemmatizer()
            self.user_profiles = self.load_user_profiles()
            self.MIN_RATINGS_FOR_RECOMMENDATIONS = 5
            self.prepare_data()
            self.create_tfidf_matrix()

        except FileNotFoundError as e:
            st.error(f"Error: {e}. Por favor, verifica que el archivo exista en el directorio correcto.")
            raise
        except pd.errors.EmptyDataError:
            st.error("Error: El archivo CSV está vacío.")
            raise
        except Exception as e:
            st.error(f"Error inesperado al cargar los datos: {e}")
            raise

    def prepare_data(self):
        """
        Prepara los datos para el sistema de recomendación.
        Limpia y procesa las columnas necesarias y elimina duplicados.
        """
        try:
            required_columns = ['title', 'genre', 'year', 'critic_score', 'people_score']
            for col in required_columns:
                if col not in self.data.columns:
                    raise KeyError(f"La columna requerida '{col}' no está presente en el dataset.")

            # Eliminar películas duplicadas
            self.data = self.data.drop_duplicates(subset=['title', 'year'], keep='first').reset_index(drop=True)

            # Procesar la columna 'genre'
            self.data['genre'] = self.data['genre'].apply(lambda x: x.split('|') if pd.notnull(x) else [])

            # Completar valores faltantes
            self.data['critic_score'] = self.data['critic_score'].fillna(0)
            self.data['people_score'] = self.data['people_score'].fillna(0)
        except Exception as e:
            raise RuntimeError(f"Error al preparar los datos: {e}")

  
    def create_tfidf_matrix(self):
        self.data['synopsis'] = self.data['synopsis'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data['synopsis'])

    def load_user_profiles(self):
        """
        Carga los perfiles de usuario desde un archivo JSON si existe,
        si no, crea un nuevo diccionario vacío
        """
        try:
            if os.path.exists('user_profiles.json'):
                with open('user_profiles.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"No se pudieron cargar los perfiles de usuario: {e}")
        return {}

    def save_user_profiles(self):
        """
        Guarda los perfiles de usuario en un archivo JSON
        """
        try:
            # Filtrar las preferencias vacías
            for user_id, profile in self.user_profiles.items():
                profile = {k: v for k, v in profile.items() if v}  # Elimina claves con valores vacíos

            with open('user_profiles.json', 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"No se pudieron guardar los perfiles de usuario: {e}")

    def get_user_ratings_history(self, user_id):
        """
        Obtiene el historial de valoraciones del usuario
        """
        if user_id not in self.user_profiles:
            return pd.DataFrame()

        ratings = self.user_profiles[user_id]['ratings']
        ratings_data = []

        for movie_idx, rating in ratings.items():
            try:
                movie = self.data.iloc[int(movie_idx)]
                ratings_data.append({
                    'title': movie['title'],
                    'year': movie['year'],
                    'rating': rating,
                    'date_rated': self.user_profiles[user_id].get('rating_dates', {}).get(movie_idx, 'Unknown')
                })
            except IndexError:
                continue  # Ignorar si el índice de la película no existe

        return pd.DataFrame(ratings_data)

    def create_user_profile(self, user_id):
        """
        Crea un nuevo perfil de usuario si no existe
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'ratings': {},
                'rating_dates': {},
                'genre_preferences': {},
                'actor_preferences': {},
                'director_preferences': {},
                'year_preferences': {},
                'language_preferences': {}
            }
            self.save_user_profiles()
    
    def get_movie_similarity(self, movie_idx):
        movie_vector = self.tfidf_matrix[movie_idx]
        similarity_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        similar_indices = similarity_scores.argsort()[::-1][1:11]  # Top 10 similares, excluyendo la misma película
        return self.data.iloc[similar_indices][['title', 'year', 'genre']]


    def rate_movie(self, user_id, movie_idx, rating):
        """
        Permite al usuario valorar una película
        """
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)

        movie_idx_str = str(movie_idx)

        self.user_profiles[user_id]['ratings'][movie_idx_str] = rating
        self.user_profiles[user_id]['rating_dates'][movie_idx_str] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.save_user_profiles()

    def can_get_recommendations(self, user_id):
        """
        Verifica si el usuario tiene suficientes valoraciones para recibir recomendaciones
        """
        if user_id not in self.user_profiles:
            return False
        return len(self.user_profiles[user_id]['ratings']) >= self.MIN_RATINGS_FOR_RECOMMENDATIONS

    def get_user_recommendations(self, user_id):
        if user_id not in self.user_profiles:
            return pd.DataFrame()
        
        user_ratings = self.user_profiles[user_id]['ratings']
        if not user_ratings:
            return pd.DataFrame()
        
        # Obtener todas las películas valoradas por el usuario
        rated_movies = list(map(int, user_ratings.keys()))
        
        # Calcular la similitud con todas las películas valoradas
        similar_movies = []
        for idx in rated_movies:
            similar_movies.extend(self.get_movie_similarity(idx).index.tolist())
        
        # Eliminar duplicados y películas ya valoradas
        similar_movies = list(set(similar_movies) - set(rated_movies))
        
        # Ordenar las películas similares por su puntuación de similitud promedio
        movie_scores = {}
        for movie in similar_movies:
            scores = []
            for rated_movie in rated_movies:
                similarity = cosine_similarity(self.tfidf_matrix[movie], self.tfidf_matrix[rated_movie])[0][0]
                user_rating = user_ratings[str(rated_movie)]
                scores.append(similarity * user_rating)
            movie_scores[movie] = sum(scores) / len(scores)
        
        # Ordenar las películas por puntuación y obtener las top N
        top_movies = sorted(movie_scores, key=movie_scores.get, reverse=True)[:10]
        
        return self.data.loc[top_movies][['title', 'year', 'genre']]



    def update_user_preferences(self, user_id, genre=None, actor=None, director=None, year=None, language=None):
        """
        Actualiza las preferencias del usuario en las categorías especificadas.
        Solo actualiza si hay un valor.
        """
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)

        # Solo actualizamos las preferencias que no están vacías
        if genre:
            self.user_profiles[user_id]['genre_preferences'] = genre
        if actor:
            self.user_profiles[user_id]['actor_preferences'] = actor
        if director:
            self.user_profiles[user_id]['director_preferences'] = director
        if year:
            self.user_profiles[user_id]['year_preferences'] = year
        if language:
            self.user_profiles[user_id]['language_preferences'] = language

        # Guardamos las preferencias si se actualizan
        self.save_user_profiles()

def main():
    st.title("Sistema de Recomendación de Películas")

    csv_files = [f for f in os.listdir() if f.endswith('.csv')]

    if not csv_files:
        st.error("No se encontraron archivos CSV en el directorio actual.")
        st.info("Por favor, asegúrate de tener un archivo CSV con los datos de las películas en el mismo directorio que este script.")
        return

    if len(csv_files) > 1:
        selected_file = st.selectbox("Selecciona el archivo de datos:", csv_files)
    else:
        selected_file = csv_files[0]

    try:
        recommender = MovieRecommender(selected_file)

        user_id = st.text_input("ID de Usuario", "user1")

        tabs = st.tabs(["Películas Disponibles", "Valorar Películas", "Mis Valoraciones", "Recomendaciones"])

        with tabs[0]:
            st.subheader("Películas Disponibles")
            movies_display = recommender.data[['title', 'year', 'genre', 'critic_score', 'people_score']]
            st.dataframe(movies_display)

        with tabs[1]:
            st.subheader("Valorar Película")
            movie_titles = recommender.data['title'].tolist()
            selected_title = st.selectbox("Selecciona una película", movie_titles)
            movie_idx = recommender.data[recommender.data['title'] == selected_title].index[0]
            rating = st.slider("Valoración", 1, 5, 3)

            if st.button("Valorar"):
                recommender.rate_movie(user_id, movie_idx, rating)
                st.success(f"¡Has valorado '{selected_title}' con {rating} estrellas!")

        with tabs[2]:
            st.subheader("Mi Historial de Valoraciones")
            ratings_history = recommender.get_user_ratings_history(user_id)
            if not ratings_history.empty:
                st.dataframe(ratings_history)
            else:
                st.info("Aún no has valorado ninguna película")

        with tabs[3]:
            st.subheader("Recomendaciones")
            if st.button("Obtener Recomendaciones"):
                if recommender.can_get_recommendations(user_id):
                    recommendations = recommender.get_user_recommendations(user_id)
                    st.dataframe(recommendations)
                else:
                    ratings_needed = recommender.MIN_RATINGS_FOR_RECOMMENDATIONS
                    current_ratings = len(recommender.user_profiles.get(user_id, {}).get('ratings', {}))
                    st.warning(f"Necesitas valorar al menos {ratings_needed} películas para obtener recomendaciones personalizadas. " 
                              f"Actualmente tienes {current_ratings} valoraciones.")

    except Exception as e:
        st.error(f"Error al inicializar el sistema: {e}")
        st.info("Por favor, verifica que el archivo CSV tenga el formato correcto y contenga los datos necesarios.")

if __name__ == "__main__":
    main()

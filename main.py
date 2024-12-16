import os
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv('OPENAI_API_KEY')
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

def create_embedding(text):
    """Create an embedding using OpenAI's text-embedding-ada-002 model"""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def get_movie_recommendation(query, movie_description):
    """Get a movie recommendation using GPT-4"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": f"Based on this movie description: '{movie_description}'\nWhy would someone interested in '{query}' like or dislike this movie? Keep it brief."}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

# Create or get index
index_name = "movie-recommendations"
try:
    index = pc.Index(index_name)
    print("Using existing index")
except:
    print("Creating new index")
    index = pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Free tier region
        )
    )

# Movie database with more detailed descriptions
movies = [
    # Sci-Fi
    "The Matrix: A mind-bending sci-fi action film where a computer programmer discovers humanity is living in a simulated reality while machines harvest their energy",
    "Inception: A sci-fi heist movie about entering dreams within dreams, exploring themes of reality, memory, and loss",
    "Blade Runner 2049: A visually stunning neo-noir sci-fi about a replicant blade runner uncovering a secret that threatens to destabilize society",
    "Arrival: A thoughtful sci-fi drama about a linguist trying to communicate with aliens, exploring themes of time, language, and human connection",
    
    # Drama/Romance
    "When Harry Met Sally: A charming romantic comedy following two friends over several years as they deal with the question of whether men and women can ever be just friends",
    "The Notebook: A passionate love story spanning decades, following a poor young man and a rich young woman who fall in love in the 1940s",
    "Eternal Sunshine of the Spotless Mind: A unique romance about a couple erasing each other from their memories, exploring love, loss, and identity",
    "La La Land: A modern musical romance about an aspiring actress and a jazz musician pursuing their dreams in Los Angeles",
    
    # Documentary
    "Planet Earth: A stunning nature documentary series showcasing Earth's most remarkable species and breathtaking landscapes",
    "Free Solo: An edge-of-your-seat documentary following a rock climber attempting to scale El Capitan without ropes",
    "My Octopus Teacher: A touching documentary about a filmmaker forming an unusual bond with an octopus in a South African kelp forest",
    "The Last Dance: An in-depth documentary series chronicling Michael Jordan and the Chicago Bulls' journey to their final championship",
    
    # Horror/Thriller
    "The Shining: A psychological horror masterpiece about a family isolated in a haunted hotel during winter, as the father descends into madness",
    "Get Out: A thought-provoking horror film about a young Black man uncovering disturbing secrets while meeting his white girlfriend's family",
    "A Quiet Place: A tense thriller about a family surviving in a post-apocalyptic world where blind creatures hunt by sound",
    "Hereditary: A deeply unsettling horror film about a family experiencing terrifying supernatural occurrences after their grandmother's death",
    
    # Action/Adventure
    "Mad Max: Fury Road: A high-octane post-apocalyptic action film about survival and redemption in a wasteland",
    "The Dark Knight: A complex superhero thriller following Batman's battle against the chaotic and unpredictable Joker",
    "Raiders of the Lost Ark: A classic adventure film following Indiana Jones on a quest to find the lost Ark of the Covenant",
    "Mission: Impossible - Fallout: A thrilling spy movie with breathtaking action sequences and complex double-crosses",
    
    # Comedy
    "Superbad: A hilarious coming-of-age comedy about two high school friends trying to attend a party before graduating",
    "The Grand Budapest Hotel: A quirky comedy about a legendary hotel concierge and his young protégé caught in a murder mystery",
    "Bridesmaids: A funny and heartfelt comedy about female friendship and rivalry during wedding preparations",
    "Shaun of the Dead: A clever zombie comedy following a slacker's attempt to survive a zombie apocalypse with his best friend"
]

# Store movies in vector database
print("\nStoring movies in database...")
for i, movie in enumerate(movies):
    vector = create_embedding(movie)
    index.upsert(vectors=[{
        "id": str(i),
        "values": vector,
        "metadata": {"description": movie}
    }])

# Interactive search
while True:
    search_text = input("\nWhat kind of movie are you interested in? (or 'quit' to exit): ")
    if search_text.lower() == 'quit':
        break

    # Get vector for search query
    search_vector = create_embedding(search_text)

    # Find similar movies
    results = index.query(
        vector=search_vector,
        top_k=2,
        include_metadata=True
    )

    print("\nHere are some recommendations based on your interest:")
    print("------------------------------------------------")
    
    for match in results.matches:
        movie_desc = match.metadata['description']
        print(f"\nSimilarity Score: {match.score:.2f}")
        print(f"Movie: {movie_desc}")
        
        # Get AI-powered recommendation explanation
        explanation = get_movie_recommendation(search_text, movie_desc)
        print(f"\nWhy this recommendation: {explanation}\n")
        print("------------------------------------------------")
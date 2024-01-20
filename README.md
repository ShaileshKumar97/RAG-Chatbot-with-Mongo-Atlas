# Generative AI RAG Chatbot with Mongo Atlas

### Vector Store
- Mongo Atlas Vector Search (you can get a free version of it by creating an account)

### How to Run this app
- To run this app you need to install dependencies with `pip install -r requirements.txt`.
- Create a Mongo Database, collection and vector index with following properties
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536, # for openai embeddings
      "similarity": "cosine"
    }
  ]
}
```
- Set following environment variables in your `.env` file:
> - OPENAI_API_KEY
> - MONGO_URI
> - MONGODB_NAME
> - MONGODB_COLLECTION
> - ATLAS_VECTOR_SEARCH_INDEX_NAME
- Now run the `python app.py` in your terminal and bingo, your app is running on your `localhost`.

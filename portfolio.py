
import pandas as pd
import faiss
import numpy as np
import uuid

class Portfolio:
    def __init__(self, file_path="C:/Users/rahul/react/AI-part1/resource/my_portfolio.csv"):

        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.vector_dimension = 300  # Set this to your vector dimension
        self.index = faiss.IndexFlatL2(self.vector_dimension)  # Create a FAISS index
        self.ids = []  # List to store document IDs

    def load_portfolio(self):
        # Here, you'll need to convert your Techstack data into embeddings
        # For demonstration, we'll use random embeddings; replace this with actual embeddings.
        for _, row in self.data.iterrows():
            # Generate an embedding (replace this with your embedding logic)
            embedding = np.random.rand(self.vector_dimension).astype('float32')
            
            # Add embedding to the FAISS index
            self.index.add(np.array([embedding]))
            self.ids.append(str(uuid.uuid4()))  # Generate a unique ID for each entry

    def query_links(self, skills):
        # Convert the skills to embeddings (replace this with your actual logic)
        query_embedding = np.random.rand(self.vector_dimension).astype('float32').reshape(1, -1)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, k=2)  # Find 2 nearest neighbors
        
        # Retrieve links based on the indices
        results = []
        for index in indices[0]:
            if index >= 0:  # Ensure the index is valid
                results.append(self.data.iloc[index]["Links"])
        return results

# Example usage:
# portfolio = Portfolio()
# portfolio.load_portfolio()
# links = portfolio.query_links("Python, Machine Learning")
# print(links)
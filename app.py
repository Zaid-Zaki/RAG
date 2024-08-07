import os
import re
import pandas as pd
import numpy as np
import time
import torch
import pickle
from langchain.schema import Document
from nltk import tokenize
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
import nltk
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import mesop as me
import mesop.labs as mel
import base64
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')
MODEL_NAME = "gemini-1.5-flash-latest"

nltk.download('punkt')
nltk.download('stopwords')
genai.configure(api_key="AIzaSyBSdmQL-1XyHmuWPtI5bAX06sGheEUNwv8")

GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class TextProcessor:
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
        self.model_gte = AutoModel.from_pretrained('intfloat/e5-small-v2', torch_dtype=torch.float32).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.chunk_size = 600
        self.chunk_overlap = 350
        self.chunk_overlap_ratio = 0.7
        self.llm_tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-reranker-v2-base-multilingual')
        self.llm_model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            torch_dtype="auto",
            trust_remote_code=True,
        )
        genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("constructor")


    def clean_dataframe(self):
        print("clean dataframe")
        # Convert 'headings' column to lowercase if the element is a string
        self.df['headings'] = self.df['headings'].apply(lambda x: x.lower() if isinstance(x, str) else x)
        # Convert 'text' column to lowercase if the element is a string
        self.df['text'] = self.df['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)

        # Replace non-word characters in 'headings' column with spaces if the element is a string
        self.df['headings'] = self.df['headings'].apply(lambda x: re.sub(r'\W', ' ', x) if isinstance(x, str) else x)
        # Replace non-word characters in 'text' column with spaces if the element is a string
        self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\W', ' ', x) if isinstance(x, str) else x)

        self.df['headings'] = self.df['headings'].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)
        # Replace multiple spaces with a single space and strip leading/trailing spaces in 'text' column if the element is a string
        self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)

        # Fill any NaN values in the dataframe with empty strings
        print("finish dataframe")
        self.df.fillna('', inplace=True)

    def _chunk_text(self, text):
        print("hello chunk  text")
        # Tokenize the input text into sentences
        sentences = tokenize.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # If the current chunk is empty, add the sentence to the current chunk
            if len(current_chunk) == 0:
                current_chunk += sentence
            else:
                # Calculate the overlap size based on the chunk overlap ratio
                overlap_size = int(len(sentence) * self.chunk_overlap_ratio)

                # Add the sentence to the current chunk, starting from the overlap size
                current_chunk += sentence[overlap_size:]

                # If the length of the current chunk is greater than or equal to the chunk size
                if len(current_chunk) >= self.chunk_size:
                    # Add the current chunk to the list of chunks, after stripping any leading/trailing whitespace
                    chunks.append(current_chunk.strip())

                    # Reset the current chunk to an empty string
                    current_chunk = ""

        # If there is any remaining text in the current chunk after the loop, add it to the list of chunks
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Return the list of chunks
        print("bye chunk  text")
        return chunks

    def process_texts(self):
        print("welcome process text")
        # Clean the dataframe to preprocess the text data
        self.clean_dataframe()

        # Initialize an empty list to hold the chunked texts
        chunked_texts = []

        # Iterate over each row in the dataframe
        for index, row in self.df.iterrows():
            # Combine the 'headings', 'text', and 'images_links' columns into a single string
            combined_text = f"{row['headings']} {row['text']} {row['images_links']}"

            # Chunk the combined text into smaller parts using the _chunk_text method
            chunks = self._chunk_text(combined_text)

            # Extend the chunked_texts list with the newly created chunks
            chunked_texts.extend(chunks)
        print("munir khan")
        # Generate embeddings for the chunked texts using the _embeddings_whole method
        if os.path.exists(r"F:\RAG_algo\embedding.pkl"):
            print("successful load ")
            with open(r"F:\RAG_algo\embedding.pkl", 'rb') as file:
                embeddings = pickle.load(file)
        else:
            embeddings = self._embeddings_whole(chunked_texts)

        # Return the embeddings and the chunked texts
        print("finish process text")
        return embeddings, chunked_texts

    def _embeddings_whole(self, chunked_texts):
        print("embeddings whole hi")
        # Prepare the input texts for the tokenizer
        input_texts = chunked_texts

        # Tokenize the input texts with specified maximum length, padding, truncation, and return tensors as 'pt' (PyTorch tensors)
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model_gte(**batch_dict)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Move the embeddings to the CPU and convert them to numpy arrays
        embeddings = embeddings.cpu().numpy()

        # Correct usage of pickle.dump with file opening in binary write mode
        with open("F:/RAG_algo/embedding.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        print("bye embeddings whole")
        return embeddings



    def _cosine_similarity(self, query, embeddings):
        print("hi _cosine_similarity")
        # Generate an embedding for the query
        query_embedding = self._embed_query(query)

        # Compute the cosine similarity between the query embedding and the provided embeddings
        similarity_matrix = cosine_similarity([query_embedding], embeddings)

        # Get the indices of the embeddings sorted by similarity to the query in descending order
        top_text_indices = np.argsort(similarity_matrix, axis=1)[0][::-1]

        # Ensure the similarity matrix is of type float
        similarity_matrix = similarity_matrix.astype(float)
        print("bye _cosine_similarity")
        # Return the sorted indices and the similarity matrix
        return top_text_indices, similarity_matrix

    def _embed_query(self, query):
        print("hey embed querry")
        # Tokenize the input query with a maximum length of 512 tokens, add padding, and return PyTorch tensors
        batch_dict = self.tokenizer([query], max_length=512, padding=True, truncation=True, return_tensors='pt')

        # Move the tensors to the appropriate device (CPU or GPU)
        batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}

        # Disable gradient calculations for the model inference to save memory and computation
        with torch.no_grad():
            outputs = self.model_gte(**batch_dict)
        query_embedding = outputs.last_hidden_state.mean(dim=1)
        print("bye embed querry")
        return query_embedding.cpu().numpy()[0]

    def retrieve_texts(self, query, top_n=8):

        print("hey retrieve_texts")
        embeddings, chunked_texts = self.process_texts()
        top_text_indices, similarity_matrix = self._cosine_similarity(query, embeddings)
        retrieved_texts = []
        for idx in top_text_indices[:top_n]:
            text_info = {}
            if idx < len(self.df):
                chunked_texts = self._chunk_text(f"{self.df.iloc[idx]['headings']} {self.df.iloc[idx]['text']} ")
                retrieved_text = " ".join(chunked_texts) if chunked_texts else ''
                retrieved_text = re.sub(r'\s+', ' ', retrieved_text).strip()
                if retrieved_text.isupper():
                    retrieved_text = retrieved_text.capitalize()
                emphasized_text = self._emphasize_words(retrieved_text)
                text_info['text'] = emphasized_text
                text_info['similarity_score'] = float(similarity_matrix[0][idx])
                text_info['images'] = self.df.iloc[idx]['images_links'].split(',') if isinstance(
                    self.df.iloc[idx]['images_links'], str) and self.df.iloc[idx]['images_links'].strip() else []
                retrieved_texts.append(text_info)

        most_relevant_doc = self.find_most_relevant_document(query, [text_info['text'] for text_info in
                                                                            retrieved_texts])
        response = {
            'most_relevant_document': most_relevant_doc
            # 'images': most_relevant_doc_images
        }
        print("hey retrieve_texts")
        return most_relevant_doc

    def _emphasize_words(self, text):
        print("hello _emphasize_words")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        emphasized_text = []
        for sentence in sentences:
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                emphasized_text.append(sentence)
        return ' '.join(emphasized_text)

    def find_most_relevant_document(self, query, documents, model_name='gemini-1.5-flash-latest'):
        print("hello find most relvant document")
        template = """
                context:
                {context}

                QUESTION:
                {question}

                INSTRUCTIONS:
                Answer the user's QUESTION using the context text above.
                Keep your answer grounded in the facts of the context.
                If the context doesn't contain the facts to answer the QUESTION, return 'I only have information related to bitsol.'
                """

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        prompt = PromptTemplate.from_template(template)
        doc_objects = [Document(page_content=doc) for doc in documents]
        db = FAISS.from_documents(doc_objects, hf)

        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        response = qa_chain.invoke(query)

        return response.get("result")


def load_data():
    file_key = r"./dataframe.csv"
    df = pd.read_csv(file_key)
    return df

df = load_data()
text_processor = TextProcessor(df)
@me.page(path="/", title="Website Retrieval Bot")
def app():
    def process_query(query: str):
        start_time = time.time()
        response = text_processor.retrieve_texts(query)
        end_time = time.time()
        print(f" execution time: {end_time - start_time:.4f} seconds")
        return response


    return mel.text_to_text(
        process_query,
        title="Website Retrieval Bot"
    )

import os
import re
import pandas as pd
import numpy as np
import time
import torch
import pickle
from style import chat
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from google.api_core import exceptions as google_exceptions
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import mesop as me
import mesop.labs as mel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')
MODEL_NAME = "gemini-1.5-flash-latest"
genai.configure(api_key="AIzaSyBSdmQL-1XyHmuWPtI5bAX06sGheEUNwv8")
GENERATION_CONFIG = {"temperature": 1,"top_p": 0.95,"top_k": 64,"max_output_tokens": 8192,}
class TextProcessor:
    def __init__(self, df):
        self.df = df
        self.calander_df=pd.read_excel(r"C:\Users\HP\Downloads\calendly.xlsx")
        self.chunk_size = 600
        self.chunk_overlap = 350
        self.chunk_overlap_ratio = 0.7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        model_name_gte = 'intfloat/e5-small-v2'
        llm_model_name = 'jinaai/jina-reranker-v2-base-multilingual'
        local_model_gte_path = './models/intfloat-e5-small-v2'
        local_llm_model_path = './models/jina-reranker-v2-base-multilingual'
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_gte_path, local_files_only=True)
            self.model_gte = AutoModel.from_pretrained(local_model_gte_path, torch_dtype=torch.float32,local_files_only=True).to(self.device)
        except EnvironmentError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_gte)
            self.model_gte = AutoModel.from_pretrained(model_name_gte, torch_dtype=torch.float32).to(self.device)
            self.tokenizer.save_pretrained(local_model_gte_path)
            self.model_gte.save_pretrained(local_model_gte_path)
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(local_llm_model_path, local_files_only=True)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(
                local_llm_model_path,torch_dtype="auto",local_files_only=True,trust_remote_code=True)
        except EnvironmentError:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(llm_model_name,torch_dtype="auto",trust_remote_code=True)
            self.llm_tokenizer.save_pretrained(local_llm_model_path)
            self.llm_model.save_pretrained(local_llm_model_path)
        genai.GenerativeModel(model_name=MODEL_NAME,generation_config=GENERATION_CONFIG,)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clean_dataframe(self):
        self.df['Headings'] = self.df['Headings'].apply(lambda x: x.lower() if isinstance(x, str) else x)
        self.df['Headings'] = self.df['Headings'].apply(lambda x: re.sub(r'\W', ' ', x) if isinstance(x, str) else x)
        self.df['Headings'] = self.df['Headings'].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)
        self.df.fillna('', inplace=True)

    def _chunk_text(self, text):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) == 0:
                current_chunk += sentence
            else:
                overlap_size = int(len(sentence) * self.chunk_overlap_ratio)
                current_chunk += sentence[overlap_size:]
                if len(current_chunk) >= self.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def process_texts(self):
        self.clean_dataframe()
        chunked_texts = []
        links = []
        for index, row in self.df.iterrows():
            chunks = self._chunk_text(row['Headings'])
            chunked_texts.extend(chunks)
            links.extend([row['Links']] * len(chunks))
        if os.path.exists(r"F:\RAG_algo\embedding.pkl"):
            with open(r"F:\RAG_algo\embedding.pkl", 'rb') as file:
                embeddings = pickle.load(file)
        else:
            embeddings = self._embeddings_whole(chunked_texts)
        return embeddings, chunked_texts, links

    def _embeddings_whole(self, chunked_texts):
        input_texts = chunked_texts
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model_gte(**batch_dict)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = embeddings.cpu().numpy()
        with open("F:/RAG_algo/embedding.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

    def _cosine_similarity(self, query, embeddings):
        query_embedding = self._embed_query(query)
        similarity_matrix = cosine_similarity([query_embedding], embeddings)
        top_text_indices = np.argsort(similarity_matrix, axis=1)[0][::-1]
        similarity_matrix = similarity_matrix.astype(float)
        return top_text_indices, similarity_matrix

    def _embed_query(self, query):
        batch_dict = self.tokenizer([query], max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model_gte(**batch_dict)
        query_embedding = outputs.last_hidden_state.mean(dim=1)
        return query_embedding.cpu().numpy()[0]

    def retrieve_texts(self, query, top_n=8):
        embeddings, chunked_texts, links = self.process_texts()
        top_text_indices, similarity_matrix = self._cosine_similarity(query, embeddings)
        retrieved_texts = []
        retrieved_links = []
        valid_indices = [idx for idx in top_text_indices if idx < len(chunked_texts)]
        for idx in valid_indices[:top_n]:
            retrieved_text = chunked_texts[idx]
            retrieved_text = re.sub(r'\s+', ' ', retrieved_text).strip()
            if retrieved_text.isupper():
                retrieved_text = retrieved_text.capitalize()
            emphasized_text = self._emphasize_words(retrieved_text)
            retrieved_texts.append({'text': emphasized_text})
            retrieved_links.append(links[idx])
        most_relevant_doc = self.find_most_relevant_document(
            query,
            [text_info['text'] for text_info in retrieved_texts],
            retrieved_links
        )
        return most_relevant_doc

    def _emphasize_words(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        emphasized_text = []
        for sentence in sentences:
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                emphasized_text.append(sentence)
        return ' '.join(emphasized_text)



    def find_most_relevant_document(self, query, documents, links, model_name='gemini-1.5-flash-latest',
                                    index_name='my_faiss_index', conversation_history=None):

        if conversation_history and len(conversation_history) >= 2:
            last_two_inputs = conversation_history[-2:]
        else:
            last_two_inputs = conversation_history or []

        template = """
                        
                        System: You are a helpful assistant for the GLA website. Maintain context from the recent conversation history provided.

                    Recent conversation history: {chat_history}
                    Context: {context}
                    Available Links: {links}
                    Human: {question}
                    
                    Assistant: Let me help you with that. I'll use the context and available links to provide the most relevant information.
                    
                    Instructions:
                    Use Context and Data:
                    - Utilize the context and recent conversation history to answer the question.
                    - Offer detailed responses based on the provided facts and data.
                    
                    Relevance:
                    - If the context includes relevant GLA information, provide that information and explain its relevance.
                    - Avoid stating that you don't have the information; instead, offer related insights based on available data.
                    
                    No Direct Match:
                    - If the context and recent conversation history do not contain relevant information:
                      - Search for keyword matches.
                      - Provide information related to GLA services based on keyword relevance.
                    
                    Keyword Matches:
                    - For exact keyword matches:
                      - Example: "marketing" -> "GLA offers comprehensive marketing services including digital marketing, SEO, and content strategy."
                    - For partial keyword matches:
                      - Example: "market" -> "GLA has various marketing solutions. How can I assist you further?"
                    - If there is no keyword match, provide a general summary of GLA's services.
                    
                    Types of Questions:
                    - Answer "how," "what," "why," and "where" questions based on the provided context and data.
                    
                    Continuity:
                    - Maintain continuity with recent conversation history for follow-up questions or previous topics.
                    - Address any relevant older context if applicable to the current question.
                    
                    Response Structure:
                    - Ensure responses are structured logically and are grounded in the provided information.
                    - Provide clear, concise answers with a logical flow.
                    
                    Unclear Queries:
                    - If the input is unclear, respond with: "Please provide a clear question so I can assist you better."
                    
                    Follow-up Questions:
                    - Refer to the conversation history when answering follow-up questions to maintain relevance and coherence.
                    
                    Detailed Explanation Instructions:
                    - Length and Clarity: Provide precise and clear answers in 2 to 3 lines maximum.
                    - Examples and Analogies: Use examples or analogies if necessary to clarify points.
                    - Flow and Coherence: Ensure responses have a logical flow and are coherent.
                    - Include links when relevant: Provide clickable links in markdown format [name](link address) if it enhances the response.
                    
                    Variation in Responses:
                    - Unique Wording: Use varied phrasing and sentence structure to keep interactions engaging and dynamic.
                    
                    For website or product development queries, tailor responses based on the provided context and include relevant links where applicable.


        """

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question", "links"]
        )

        doc_objects = [Document(page_content=f"{doc}\nRelevant Link: {link}", metadata={"link": link})
                       for doc, link in zip(documents, links)]
        if os.path.exists(f"{index_name}.faiss"):
            db = FAISS.load_local(index_name, hf)
        else:
            db = FAISS.from_documents(doc_objects, hf)
            db.save_local(index_name)

        online_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            online_model,
            retriever=db.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            },
            return_source_documents=True,
            return_generated_question=True,
            chain_type="stuff"
        )


        links_string = "\n".join(links)
        context = " ".join(documents)



        try:
            # Prepare inputs for the chain
            chain_inputs = {
                "question": query,
                "chat_history": last_two_inputs,
                "context": context,
                "links": links_string
            }

            # Prepare inputs for the memory
            memory_inputs = {
                "question": query
            }

            # Use the invoke method
            response = qa_chain.invoke(chain_inputs, return_only_outputs=True)

            # Manually save the context to memory
            self.conversation_memory.save_context(memory_inputs, {"answer": response["answer"]})

        except Exception as e:
            import traceback
            raise

        return response.get("answer")


def load_data():
    file_key = r"F:\RAG_algo\Gla2.xlsx"
    df = pd.read_excel(file_key)
    return df

df = load_data()
text_processor = TextProcessor(df)
@me.page(path="/", title="Website Retrieval Bot")
def app():
    def process_query(query: str,history: list[mel.ChatMessage]):
        start_time = time.time()
        response = text_processor.retrieve_texts(query)
        end_time = time.time()
        print(f" execution time: {end_time - start_time:.4f} seconds")
        return response
    # return mel.text_to_text(
    #     process_query,
    #     title="Website Retrieval Bot")

    chat(process_query, title="Website Retrieval Bot")
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableSequence
import json
import ast

import openai
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import pandas as pd
import chardet
from bs4 import BeautifulSoup
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from langchain.schema.runnable import RunnablePassthrough
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBSdmQL-1XyHmuWPtI5bAX06sGheEUNwv8"

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')
MODEL_NAME = "gemini-1.5-flash-latest"
genai.configure(api_key="AIzaSyBSdmQL-1XyHmuWPtI5bAX06sGheEUNwv8")

GENERATION_CONFIG = {"temperature": 1,"top_p": 0.95,"top_k": 64,"max_output_tokens": 8192,}
def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    return parsed.netloc == base_domain and parsed.scheme in ('http', 'https')

def scrape_website(start_url, max_pages=30, output_file='scraped_data.json'):
    base_domain = urlparse(start_url).netloc
    scraped_content = {}
    urls_to_scrape = [start_url]
    scraped_urls = set()
    new_links = set()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        while urls_to_scrape and len(scraped_urls) < max_pages:
            url = urls_to_scrape.pop(0)
            print(f"Scraping URL: {url}")
            if url in scraped_urls:
                print(f"URL already scraped: {url}")
                continue
            try:
                page.goto(url)
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                links = page.eval_on_selector_all('a[href]', 'elements => elements.map(el => el.href)')

                additional_links = []
                img_links = [img.get('src') for img in soup.find_all('img', src=True)]
                additional_links.extend(img_links)
                button_links = []
                print("Processing buttons...")
                for button in soup.find_all('button'):
                    data_attrs = [button.get(f'data-{attr}') for attr in button.attrs if attr.startswith('data-')]
                    button_links.extend(data_attrs)

                    style = button.get('style', '')
                    bg_image_url = re.search(r'url\(["\']?(.*?)["\']?\)', style)
                    if bg_image_url:
                        button_links.append(bg_image_url.group(1))

                additional_links.extend(button_links)
                print(f"Found additional links: {additional_links}")

                for link in additional_links:
                    absolute_url = urljoin(url, link)
                    if is_valid_url(absolute_url, base_domain) and absolute_url not in scraped_urls and absolute_url not in new_links:
                        new_links.add(absolute_url)

                for link in links:
                    absolute_url = urljoin(url, link)
                    if is_valid_url(absolute_url, base_domain) and absolute_url not in scraped_urls and absolute_url not in new_links:
                        new_links.add(absolute_url)

                print(f"Links to follow: {new_links}")

                urls_to_scrape.extend(new_links)
                new_links.clear()

                scraped_content[url] = {'links': links}
                scraped_urls.add(url)
                print(f"Scraped URL added: {url}")

            except Exception as e:
                print(f"Error scraping {url}: {e}")

        browser.close()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_content, f, indent=2, ensure_ascii=False)

    data = list(scraped_content.keys())
    total_scraped = len(data)
    print(f"Total pages scraped: {total_scraped}")
    if total_scraped < max_pages:
        print(f"Fewer pages scraped than requested. Found {total_scraped} pages.")
    return data

def loading(output_file):
    loaded_content = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_content = json.load(f)
        print(f"Loaded pages: {len(loaded_content)}")
    except Exception as e:
        print(f"Error loading file {output_file}: {e}")

    return list(loaded_content.keys())

def extract_content_from_urls(url_list, output_file='extracted_content.txt'):
    content_dict = {}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in url_list:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            detected_encoding = chardet.detect(response.content)['encoding']
            if detected_encoding is None:
                detected_encoding = 'utf-8'

            decoded_content = response.content.decode(detected_encoding, errors='replace')
            soup = BeautifulSoup(decoded_content, 'html.parser')

            headings = [(h.get_text(), url) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            paragraphs = [(p.get_text(), url) for p in soup.find_all('p')]
            div_texts = [(div.get_text(), url) for div in soup.find_all('div')]

            images = [
                {
                    'alt': img.get('alt', ''),
                    'src': img.get('src', ''),
                    'url': url
                }
                for img in soup.find_all('img')
            ]

            multilingual_texts = [
                (element.get_text(separator=' ', strip=True), url)
                for element in soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            ]

            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]

            content_dict[url] = {
                'Headings': headings,
                'Paragraphs': paragraphs,
                'Div Texts': div_texts,
                'Images': images,
                'Multilingual Texts': multilingual_texts,
                'Links': links
            }

        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}. Error: {e}")
            content_dict[url] = {
                'Headings': [],
                'Paragraphs': [],
                'Div Texts': [],
                'Images': [],
                'Multilingual Texts': [],
                'Links': []
            }

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(content_dict, file, indent=4, ensure_ascii=False)

    print(f"All content saved to {output_file}")
    return content_dict

def extracteddata(output_file='extracted_content.txt'):
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_content = json.load(f)
        data = list(loaded_content.values())
    except Exception as e:
        print(f"Error loading file {output_file}: {e}")
        return {}

    first_three_items = list(loaded_content.items())[:2]
    return loaded_content


def preprocess_text(text):
    if not isinstance(text, str):
        print("Text is not a string, returning empty string.")
        return ''

    # Debug: Print initial text
    print(f"Initial text: {text}")

    # Remove URLs and unwanted patterns
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\\n|\\t|\\r', ' ', text)  # Remove escape sequences
    text = re.sub(r'\[.*?\]', '', text)  # Remove any text within square brackets
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    # Debug: Print text after removals
    print(f"Text after removals: {text}")

    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
    except Exception as e:
        print(f"Error parsing text with BeautifulSoup: {e}")
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text).strip()

    print(f"Final processed text: {text}")
    return text


def preprocess_website_data(website_data, output_file="Preprocess.json"):
    preprocessed_data = {}
    for url, content in website_data.items():
        preprocessed_content = {}
        for content_type, items in content.items():
            print(f"Processing content type: {content_type}")  # Debug statement

            if isinstance(items, list):
                preprocessed_content[content_type] = []
                for item in items:
                    print(f"Processing item: {item}")  # Debug statement

                    if isinstance(item, list) and len(item) == 2:
                        text, item_url = item
                        text = preprocess_text(text)  # Preprocess only the text
                        preprocessed_content[content_type].append({'text': text, 'url': item_url})
                    elif isinstance(item, dict):
                        # Expecting item to be a dictionary with 'text' and 'url'
                        text = preprocess_text(item.get('text', ''))
                        item_url = item.get('url', url)
                        preprocessed_content[content_type].append({'text': text, 'url': item_url})
                    else:
                        # Handle unexpected formats
                        text = preprocess_text(str(item))
                        preprocessed_content[content_type].append({'text': text, 'url': url})

            elif isinstance(items, str):
                preprocessed_content[content_type] = {
                    'text': preprocess_text(items),
                    'url': url
                }
            else:
                preprocessed_content[content_type] = items

        preprocessed_data[url] = preprocessed_content

    with open(output_file, 'w') as file:
        json.dump(preprocessed_data, file, indent=4)

    print(f"Preprocessed data saved to {output_file}")
    return preprocessed_data



def separate_headings_paragraphs(data_dict):
    headings = []
    paragraphs = []
    div_text = []
    multilingual_text = []
    images = []
    for url, content in data_dict.items():
        if "Headings" in content:
            headings.extend(content["Headings"])
        if "Paragraphs" in content:
            paragraphs.extend(content["Paragraphs"])
        if "Div Texts" in content:
            div_text.extend(content["Div Texts"])
        if "Images" in content:
            images.extend(content["Images"])
        if "Multilingual Texts" in content:
            multilingual_text.extend(content["Multilingual Texts"])
    return headings, paragraphs, div_text, images, multilingual_text

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if w not in stop_words])

# Function to create content blocks
def create_content_blocks(headings, paragraphs, div_text, images, multilingual_text):
    content_blocks = []
    for h in headings:
        content_blocks.append(h)
    for p in paragraphs:
        content_blocks.append(p)
    for d in div_text:
        content_blocks.append(d)
    for i in images:
        content_blocks.append(i)
    for m in multilingual_text:
        content_blocks.append(m)
    return content_blocks

# Function to compute similarity matrix
def compute_similarity_matrix(content_blocks, top_n=10):
    texts = np.array(content_blocks)
    if texts.size == 0:
        logging.error("No valid text content to process.")
        return np.array([])
    texts = [str(text) for text in texts]
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        tfidf_matrix = normalize(tfidf_matrix, axis=1, norm='l2')
        f = tfidf_matrix.shape[1]
        t = AnnoyIndex(f, 'angular')
        for i in range(len(tfidf_matrix)):
            t.add_item(i, tfidf_matrix[i])
        t.build(10)
        top_similarities = {}
        for i in range(len(tfidf_matrix)):
            neighbors = t.get_nns_by_item(i, top_n + 1)
            top_similarities[i] = {j: 1 - t.get_distance(i, j) for j in neighbors if j != i}
    except ValueError as e:
        logging.error(f"Error in TF-IDF computation: {e}")
        return np.array([])
    return top_similarities

# Function to load JSON file
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        logging.error(f"The file {file_path} is not a valid JSON file.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None


def cluster(similarity_dict, similarity_threshold=0.4):
    num_blocks = len(similarity_dict)
    related_clusters = []
    non_related_blocks = []
    seen_clusters = set()  # To keep track of unique clusters

    for i in range(num_blocks):
        if i in [block for cluster in related_clusters for block in cluster]:
            continue

        related_blocks = [j for j, similarity in similarity_dict.get(i, {}).items() if
                          similarity >= similarity_threshold]

        if related_blocks:
            # Create the current cluster
            current_cluster = [i] + related_blocks
            # Convert the cluster to a frozenset to use as a key for uniqueness check
            cluster_key = frozenset(current_cluster)

            # Check if the cluster is unique
            if cluster_key not in seen_clusters:
                related_clusters.append(current_cluster)
                seen_clusters.add(cluster_key)
        else:
            non_related_blocks.append(i)

    return related_clusters, non_related_blocks

# Function to process results and include links
def process_results(text_content_blocks, links, grouped_indices, independent_indices):
    dependent_text = []
    independent_text = []
    dependent_links = []
    independent_links = []
    for group in grouped_indices:
        group_texts = [text_content_blocks[i] for i in group]
        group_links = [links[i] for i in group]
        dependent_text.extend(group_texts)
        dependent_links.extend(group_links)
    independent_text = [text_content_blocks[i] for i in independent_indices]
    independent_links = [links[i] for i in independent_indices]
    return dependent_text, dependent_links, independent_text, independent_links

# Main function to process and save results
def main(headings, paragraphs, div_text, images, multilingual_text, similarity_threshold=0.3):
    content_blocks = create_content_blocks(headings, paragraphs, div_text, images, multilingual_text)
    text_content_blocks = [block for block in content_blocks if not isinstance(block, list) or block[0] != 'image']
    links = [content.get('url', '') for content in headings + paragraphs + div_text + images + multilingual_text]
    similarity_matrix = compute_similarity_matrix(text_content_blocks)
    grouped_indices, independent_indices = cluster(similarity_matrix, similarity_threshold)
    return process_results(text_content_blocks, links, grouped_indices, independent_indices)




def process_cluster_with_llm_and_save(df, file_name='summary.json'):
    # Convert DataFrame to list of text and links
    texts_and_links = []
    for index, row in df.iterrows():
        heading = row['headings']
        link = row['links']
        if heading and link:
            texts_and_links.append(f"{heading}: {link}")

    # Join the texts and links into a single string
    cluster_text = "\n".join(texts_and_links)

    prompt = ChatPromptTemplate.from_template("""
                Analyze the following large cluster of text containing 1561 entries and provide an extremely comprehensive and detailed summary:
                {cluster_text}
                Please follow these guidelines:
                1. Process and summarize ALL 1000 entries from the input text.
                2. Create a Python dictionary where the keys are summarized sentences and the values are the corresponding links.
                3. Ensure that each sentence has a corresponding link.
                4. Format each key (text entry) as a complete, coherent sentence with 10-15 words.
                5. It is crucial to cover ALL information from the original input. Do not omit any details.
                6. Create as many sentences as necessary to capture all the information from the 1561 entries.
                7. Each sentence should be 10-15 words long.
                8. If an entry contains multiple ideas, create multiple sentences to cover all aspects.
                9. The final output should be very long, potentially containing hundreds or even thousands of sentences.
                10. Format the output as a valid Python dictionary.
                11. Create overlap sentences also. 
                12. Make sure you cover all 1000 sentences.
                13. Make sure my overall dictionary data should be big not too much small

                Output the dictionary in the following format:
                {{
                    "First sentence summarizing part of the information in 10-15 words.": "Link 1",
                    "making a sentence of that corresponding entity and eleborating it": "Link 1",
                    
                    ...
                    [Continue this pattern for all 1000 entries]
                }}

                Remember: 
                - Process ALL 1000 entries.
                - Create as many sentences as needed to cover all information from every entry.
                - Keep each sentence to 10-15 words.
                - The final output should be extremely long and comprehensive.
                - Do not summarize or condense the overall information. Capture all details.
                - make my dataset bigger
                """)

    chain = (
            {"cluster_text": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    try:
        result = chain.invoke(cluster_text)
        print("Result type:", type(result))


        # Parse the string into a dictionary
        result_dict = {}
        # Use regex to find all key-value pairs
        pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', result)
        for key, value in pairs:
            result_dict[key] = value

        if not result_dict:
            print("Failed to parse LLM output as a dictionary. Raw output:")
            return None

        with open(file_name, 'w') as f:
            json.dump(result_dict, f, indent=4)

        print(f"Results successfully saved to {file_name}.")
        return result_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def filter_text_only(data_list):
    return [item['text'] for item in data_list if 'text' in item]


def save_lists_to_json(text_list, links_list, text_filename, links_filename):
    # Save the text list
    with open(text_filename, 'w') as text_file:
        json.dump(text_list, text_file, indent=4)

    # Save the links list
    with open(links_filename, 'w') as links_file:
        json.dump(links_list, links_file, indent=4)


def load_lists_from_files(text_file_path, links_file_path):
    with open(text_file_path, 'r') as text_file:
        text_list = text_file.read().splitlines()

    with open(links_file_path, 'r') as links_file:
        links_list = links_file.read().splitlines()

    return text_list, links_list

start_url="https://glafamily.com/"
# links_generated=scrape_website(start_url, max_pages=30, output_file='scraped_data.json')
# loading_links_generated=loading(output_file='scraped_data.json')
# contentOfWebiste=extract_content_from_urls(loading_links_generated, output_file='extracted_content.txt')
# loading_contentOfWebiste=extracteddata(output_file='extracted_content.txt')
# preprocessed_data=preprocess_website_data(loading_contentOfWebiste,"Preprocess.json")
# preprocessed_data_loading = load_json_file(file_path="Preprocess.json")
#
# if preprocessed_data_loading:
#     preprocessed_data_dict = dict(preprocessed_data_loading)
#     headings, paragraphs, div_text, images, multilingual_text = separate_headings_paragraphs(preprocessed_data_dict)
#     dependent_text, dependent_links, independent_text, independent_links = main(headings, paragraphs, div_text, images, multilingual_text)
#
# filtered_dependent_text = filter_text_only(dependent_text)
# filtered_independent_text=filter_text_only(independent_text)
# merged_list_text = filtered_dependent_text + filtered_independent_text
# merge_list_links= dependent_links+ independent_links
# save_lists_to_json(merged_list_text, merge_list_links, 'text_list.txt', 'links_list.txt')
merged_list_text_l, merge_list_links = load_lists_from_files('text_list.txt', 'links_list.txt')
# print(len(merged_list_text_l))
# print(len(merge_list_links))
#
df = pd.DataFrame({"headings": merged_list_text_l,"links": merge_list_links })
df = df[df['headings'] != ""]
df = df.drop_duplicates(subset='headings')
df.to_excel("Total.xlsx", index=False)
file_path = r"F:\RAG_algo\Total.xlsx"
df = pd.read_excel(file_path)
print(len(df))
m=process_cluster_with_llm_and_save(df, file_name='summary.json')


file_name='summary.json'
final_load_summary = load_json_file(file_name)
headings = list(final_load_summary.keys())
links = list(final_load_summary.values())
df = pd.DataFrame({
    'Headings': headings,
    'Links': links
})

df.to_excel("Gla.xlsx", index=False)






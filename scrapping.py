from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableSequence
import json
from collections import defaultdict


import ast
from playwright.sync_api import TimeoutError
from sklearn.decomposition import TruncatedSVD
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

GENERATION_CONFIG = {"temperature": 1, "top_p": 0.95, "top_k": 64, "max_output_tokens": 8192, }


def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    return parsed.netloc == base_domain and parsed.scheme in ('http', 'https')

def clean_link(link):
    """Filter out unwanted link types."""
    if link.startswith('mailto:') or link.startswith('javascript:') or link.startswith('#') or link.lower() == 'javascript:void(0);':
        return None
    return link

def scrape_website(start_url, max_pages=150, links_file='links.json'):
    base_domain = urlparse(start_url).netloc
    scraped_content = {}
    urls_to_scrape = [start_url]
    scraped_urls = set()
    all_links = set()

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
                page.goto(url, timeout=60000, wait_until='domcontentloaded')

                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')

                # Extract links from <a> tags
                links = page.eval_on_selector_all('a[href]', 'elements => elements.map(el => el.href)')

                # Extract links from buttons
                print("Processing buttons...")
                for button in soup.find_all('button'):
                    data_attrs = [button.get(f'data-{attr}') for attr in button.attrs if attr.startswith('data-')]
                    links.extend(data_attrs)

                    style = button.get('style', '')
                    bg_image_url = re.search(r'url\(["\']?(.*?)["\']?\)', style)
                    if bg_image_url:
                        links.append(bg_image_url.group(1))

                # Extract links from spans
                print("Processing spans...")
                for span in soup.find_all('span'):
                    data_attrs = [span.get(f'data-{attr}') for attr in span.attrs if attr.startswith('data-')]
                    links.extend(data_attrs)
                    style = span.get('style', '')
                    bg_image_url = re.search(r'url\(["\']?(.*?)["\']?\)', style)
                    if bg_image_url:
                        links.append(bg_image_url.group(1))

                # Extract links from paragraphs
                print("Processing paragraphs...")
                for paragraph in soup.find_all('p'):
                    links.extend([a['href'] for a in paragraph.find_all('a', href=True)])

                # Extract links from the footer
                footer = soup.find('footer')
                if footer:
                    links.extend([a['href'] for a in footer.find_all('a', href=True)])

                # Extract links from all divs
                divs = soup.find_all('div')
                for div in divs:
                    links.extend([a['href'] for a in div.find_all('a', href=True)])

                # Clean and deduplicate links
                cleaned_links = {clean_link(link) for link in links if clean_link(link)}

                # Add cleaned links to all_links set and queue for scraping if not visited
                for link in cleaned_links:
                    absolute_url = urljoin(url, link)
                    if is_valid_url(absolute_url, base_domain) and absolute_url not in scraped_urls:
                        if absolute_url not in all_links:
                            all_links.add(absolute_url)
                            urls_to_scrape.append(absolute_url)

                print(f"Links to follow: {urls_to_scrape}")
                scraped_content[url] = {'links': list(cleaned_links)}
                scraped_urls.add(url)
                print(f"Scraped URL added: {url}")
            except TimeoutError:
                print(f"Timeout error scraping {url}. Skipping this URL.")
            except Exception as e:
                print(f"Error scraping {url}: {e}")

        browser.close()

    # Save scraped links as JSON
    with open(links_file, 'w', encoding='utf-8') as f:
        json.dump({'links': list(all_links)}, f, indent=4)

    total_scraped = len(scraped_content)
    print(f"Total pages scraped: {total_scraped}")
    if total_scraped < max_pages:
        print(f"Fewer pages scraped than requested. Found {total_scraped} pages.")

    return list(scraped_content.keys())
def loading(output_file):
    loaded_content = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_content = json.load(f)
        print(f"Loaded links: {len(loaded_content.get('links', []))}")
    except json.JSONDecodeError:
        print(f"Error loading file {output_file}: File is empty or not in valid JSON format.")
    except Exception as e:
        print(f"Error loading file {output_file}: {e}")

    return loaded_content.get('links', [])


def extract_content_from_urls(url_list, output_file='extracted_content.txt'):
    content_dict = {}
    request_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for url in url_list:
        try:
            response = requests.get(url, headers=request_headers, timeout=10)
            response.raise_for_status()  # This will raise an HTTPError for bad responses
            detected_encoding = chardet.detect(response.content)['encoding']
            if detected_encoding is None:
                detected_encoding = 'utf-8'
            decoded_content = response.content.decode(detected_encoding, errors='replace')
            soup = BeautifulSoup(decoded_content, 'html.parser')

            # Extracting different elements
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

            links = [{'text': a.get_text(strip=True), 'href': urljoin(url, a.get('href'))} for a in
                     soup.find_all('a', href=True)]

            sections = [(section.get_text(), url) for section in soup.find_all('section')]
            footers = [(footer.get_text(), url) for footer in soup.find_all('footer')]
            headers = [(header.get_text(), url) for header in soup.find_all('header')]

            # Extracting text from all divs
            divs = soup.find_all('div')
            divs_texts = [(div.get_text(), url) for div in divs]

            # Extract nested divs
            nested_divs = []
            for div in divs:
                nested_texts = [nested_div.get_text() for nested_div in div.find_all('div', recursive=False)]
                nested_divs.append((div.get_text(), nested_texts, url))

            content_dict[url] = {
                'Headings': headings,
                'Paragraphs': paragraphs,
                'Div Texts': div_texts,
                'Images': images,
                'Multilingual Texts': multilingual_texts,
                'Links': links,
                'Sections': sections,
                'Footers': footers,
                'Headers': headers,
                'Nested Divs': nested_divs,
                'All Divs Texts': divs_texts
            }

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred while retrieving {url}: {http_err}")
            content_dict[url] = {
                'Headings': [],
                'Paragraphs': [],
                'Div Texts': [],
                'Images': [],
                'Multilingual Texts': [],
                'Links': [],
                'Sections': [],
                'Footers': [],
                'Headers': [],
                'Nested Divs': [],
                'All Divs Texts': []
            }
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred while retrieving {url}: {req_err}")
            content_dict[url] = {
                'Headings': [],
                'Paragraphs': [],
                'Div Texts': [],
                'Images': [],
                'Multilingual Texts': [],
                'Links': [],
                'Sections': [],
                'Footers': [],
                'Headers': [],
                'Nested Divs': [],
                'All Divs Texts': []
            }
        except Exception as e:
            print(f"An unexpected error occurred while processing {url}: {e}")
            content_dict[url] = {
                'Headings': [],
                'Paragraphs': [],
                'Div Texts': [],
                'Images': [],
                'Multilingual Texts': [],
                'Links': [],
                'Sections': [],
                'Footers': [],
                'Headers': [],
                'Nested Divs': [],
                'All Divs Texts': []
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
    sections = []
    Footers = []
    Headers = []
    NestedDivs = []
    divWithIds=[]

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
        if 'Sections' in content:
            sections.extend(content["Sections"])
        if 'Footers' in content:
            Footers.extend(content["Footers"])
        if 'Headers' in content:
            Headers.extend(content["Headers"])
        if 'Nested Divs' in content:
            NestedDivs.extend(content["Nested Divs"])
        if 'All Divs Texts' in content:
            divWithIds.extend(content['All Divs Texts'])

    return headings, paragraphs, div_text, images, multilingual_text, sections, Footers, Headers, NestedDivs,divWithIds


def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if w not in stop_words])


# Function to create content blocks
def create_content_blocks(headings, paragraphs, div_text, images, multilingual_text, sections, Footers, Headers,
                          NestedDivs,divWithIds):
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
    for q in sections:
        content_blocks.append(q)
    for r in Footers:
        content_blocks.append(r)

    for u in Headers:
        content_blocks.append(u)
    for z in NestedDivs:
        content_blocks.append(z)
    for yy in divWithIds:
        content_blocks.append(yy)
    return content_blocks


# Function to compute similarity matrix
def compute_similarity_matrix(content_blocks, top_n=10):

    texts = np.array(content_blocks)
    if texts.size == 0:
        logging.error("No valid text content to process.")
        return {}

    texts = [str(text) for text in texts]
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Reduce dimensionality
        svd = TruncatedSVD(n_components=100)  # Adjust the number of components as needed
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        reduced_matrix = normalize(reduced_matrix, axis=1, norm='l2')

        f = reduced_matrix.shape[1]
        t = AnnoyIndex(f, 'angular')

        for i in range(len(reduced_matrix)):
            t.add_item(i, reduced_matrix[i])

        t.build(10)

        top_similarities = {}
        for i in range(len(reduced_matrix)):
            neighbors = t.get_nns_by_item(i, top_n + 1)
            top_similarities[i] = {j: 1 - t.get_distance(i, j) for j in neighbors if j != i}

    except ValueError as e:
        logging.error(f"Error in TF-IDF computation: {e}")
        return {}

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




def cluster(similarity_dict, similarity_threshold=0.6):
    num_blocks = len(similarity_dict)
    related_clusters = []
    non_related_blocks = []
    seen_clusters = set()

    print(f"Number of blocks: {num_blocks}")

    cluster_dict = defaultdict(list)
    for i in range(num_blocks):
        if i in [block for cluster in related_clusters for block in cluster]:
            continue

        related_blocks = [j for j, similarity in similarity_dict.get(i, {}).items() if similarity >= similarity_threshold]
        if related_blocks:
            cluster = [i] + related_blocks
            cluster_key = frozenset(cluster)
            if cluster_key not in seen_clusters:
                cluster_dict[cluster_key] = cluster
                seen_clusters.add(cluster_key)
        else:
            non_related_blocks.append(i)

    related_clusters = list(cluster_dict.values())

    print(f"Total clusters: {len(related_clusters)}")
    print(f"Total non-related blocks: {len(non_related_blocks)}")
    return related_clusters, non_related_blocks


# Function to process results and include links
def process_results(text_content_blocks, links, grouped_indices, independent_indices):
    dependent_text = []
    independent_text = []
    dependent_links = []
    independent_links = []
    print("len links: ",len(links))
    for group in grouped_indices:
        group_texts = [text_content_blocks[i] for i in group]
        print("group:", group)
        group_links = [links[i] for i in group]
        dependent_text.extend(group_texts)
        dependent_links.extend(group_links)
    independent_text = [text_content_blocks[i] for i in independent_indices]
    independent_links = [links[i] for i in independent_indices]
    return dependent_text, dependent_links, independent_text, independent_links

# Main function to process and save results
def main(headings, paragraphs, div_text, images, multilingual_text, sections, Footers, Headers, NestedDivs,divWithIds,similarity_threshold=0.3):
    content_blocks = create_content_blocks(headings, paragraphs, div_text, images, multilingual_text, sections, Footers,
                                           Headers, NestedDivs,divWithIds)
    text_content_blocks = [block for block in content_blocks if not isinstance(block, list) or block[0] != 'image']
    links = [content.get('url', '') for content in headings + paragraphs + div_text + images + multilingual_text+ sections+Footers+ Headers+NestedDivs+divWithIds]
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

    # Split the cluster_text into chunks
    chunk_size = 2* 1024 * 1024  # 15 MB
    chunks = [cluster_text[i:i+chunk_size] for i in range(0, len(cluster_text), chunk_size)]

    # Create a dictionary to store the results
    result_dict = {}

    for chunk_idx, chunk in enumerate(chunks):
        prompt = ChatPromptTemplate.from_template("""

            Analyze the following large cluster of text containing entries and provide an extremely comprehensive and detailed summary:
            {cluster_text}
            Please follow these guidelines:

            1. Create a Python dictionary where the keys are summarized sentences and the values are the corresponding links.
            2. Ensure that each sentence has a corresponding link.
            3. Format each key (text entry) as a complete, coherent sentence with 10-15 words.
            4. Do not omit any details and make as many sentences with corresponding links as possible.
            5. Each sentence should be 10-15 words long.
            6. If an entry contains multiple ideas, create multiple sentences to cover all aspects.
            7. The final output should be very long, potentially containing hundreds or even thousands of sentences.
            8. Format the output as a valid Python dictionary.
            9. Create overlapping sentences to ensure comprehensive coverage.
            10. Extract all information and generate additional sentences to cover all details.
            11. Make the dictionary as large as possible, covering every piece of information.
            12. For entries with empty text but existing links, generate sentences based on the link content or URL structure.
            13. Create sentences from individual words or phrases if necessary to maximize information extraction.

            Output the dictionary in the following format:
            {{
                "First sentence summarizing part of the information in {{}}" .format(cluster_text): "Link 1",
                "Second sentence elaborating on the corresponding entity and expanding it": "Link 1",
                "Third sentence covering a different aspect of the same information": "Link 1",
                "Fourth sentence summarizing a new piece of information in {{}}" .format(cluster_text): "Link 2",
                "Fifth sentence generated from the URL structure of Link 2": "Link 2",
                "Sixth sentence created from individual words or phrases in the cluster": "Link 3",
                ...
                [Continue this pattern for all entries, creating multiple sentences per entry as needed]
            }}

            Remember: 
            - Create as many sentences as needed to cover all information from every entry.
            - Keep each sentence to 10-15 words.
            - The final output should be extremely long and comprehensive.
            - Do not summarize or condense the overall information. Capture all details.
            - Make more and more sentences in the dictionary.
            - Don't omit any information from the data.
            - Make as large a dictionary as possible to cover all details.
            - It is crucial to cover ALL information. Do not skip anything.
            - Generate sentences for entries with empty text but existing links.
            - Create sentences from individual words or phrases to maximize information extraction.
            - Create sentences from links.
            - dont forget to miss any details.
            - for extra information overlap the words with sentences.
            - for more extra information in some text you will see links so from that links make sentences.

        """)
        chain = (
                {"cluster_text": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
        )

        try:
            chunk_result = chain.invoke(chunk)
            print(f"Processed chunk {chunk_idx + 1}/{len(chunks)}")
            chunk_result_dict = {}
            pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', chunk_result)
            for key, value in pairs:
                chunk_result_dict[key] = value

            if not chunk_result_dict:
                print("Failed to parse LLM output as a dictionary. Raw output:")
                print(chunk_result)
                continue

            # Merge the chunk result into the main result_dict
            result_dict.update(chunk_result_dict)

        except Exception as e:
            print(f"An error occurred while processing a chunk: {e}")
            continue

    if result_dict:
        with open(file_name, 'w') as f:
            json.dump(result_dict, f, indent=4)
        print(f"Results successfully saved to {file_name}.")
    else:
        print("Failed to generate any valid results.")

    return result_dict
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

def makeitform(row):
    clean = row["headings"]
    cleaned_text = re.sub(r'[\\[\],]', '', clean)
    return cleaned_text




start_url = "https://glafamily.com/"
links_file = "links.json"
#links_generated = scrape_website(start_url, max_pages=150)
# loading_links_generated = loading(output_file='links.json')
# contentOfWebiste = extract_content_from_urls(loading_links_generated, output_file='extracted_content.txt')
# loading_contentOfWebiste = extracteddata(output_file='extracted_content.txt')
# preprocessed_data = preprocess_website_data(loading_contentOfWebiste, "Preprocess.json")
# preprocessed_data_loading = load_json_file(file_path="Preprocess.json")
#
# if preprocessed_data_loading:
#     preprocessed_data_dict = dict(preprocessed_data_loading)
#     headings, paragraphs, div_text, images, multilingual_text, sections, Footers, Headers, NestedDivs,divWithIds = separate_headings_paragraphs(preprocessed_data_dict)
#     dependent_text, dependent_links, independent_text, independent_links = main(headings, paragraphs, div_text, images, multilingual_text, sections,Footers, Headers, NestedDivs, divWithIds)
# filtered_dependent_text = filter_text_only(dependent_text)
# filtered_independent_text = filter_text_only(independent_text)
# merged_list_text = filtered_dependent_text + filtered_independent_text
# merge_list_links = dependent_links + independent_links
# save_lists_to_json(merged_list_text, merge_list_links, 'text_list.txt', 'links_list.txt')
#merged_list_text_l, merge_list_links = load_lists_from_files('text_list.txt', 'links_list.txt')
# print(len(merged_list_text_l))
# print(len(merge_list_links))
# df = pd.DataFrame({"headings": merged_list_text_l,"links": merge_list_links })
# df = df[df['headings'] != ""]
# df = df.drop_duplicates(subset='headings')
# df.to_excel("Total.xlsx", index=False)
# file_path = r"F:\RAG_algo\Total.xlsx"
# df = pd.read_excel(file_path)
# df['cleaned_text'] = df.apply(makeitform, axis=1)
# q_frame = pd.DataFrame({"headings": df['cleaned_text'], "links": df["links"]})
# q_frame.to_excel("y.xlsx", index=False)
file_path = r"F:\RAG_algo\y.xlsx"
df=pd.read_excel(file_path)
m=process_cluster_with_llm_and_save(df, file_name='summary.json')

file_name='summary.json'
final_load_summary = load_json_file(file_name)
headings = list(final_load_summary.keys())
links = list(final_load_summary.values())
df = pd.DataFrame({
    'Headings': headings,
    'Links': links
})

df.to_excel("Gla3.xlsx", index=False)
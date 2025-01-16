import os
import google.generativeai as genai
from time import time, sleep
import pandas as pd
import argparse
import json
import random
from sentence_transformers import SentenceTransformer, util
from llama_index.core import VectorStoreIndex, Document, Settings
from tenacity import retry, stop_after_attempt, wait_exponential
import pickle
import hashlib
from pathlib import Path

def get_env():
    env_dict = {}
    with open(file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict

# Initialize Gemini
GEMINI_API_KEY = get_env()["GOOGLE_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# Configure Gemini model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

"""Sentence-BERT for evaluate semantic similarity"""
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_similarity(response, ground_truth):
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)
    return cosine_score.item()

# Add cache-related functions
def get_cache_path(question, knowledge=""):
    """Generate a unique cache path based on question and knowledge"""
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create a hash of the question and knowledge to use as filename
    content_hash = hashlib.md5((question + knowledge).encode()).hexdigest()
    return cache_dir / f"response_{content_hash}.pkl"

def save_to_cache(cache_path, response):
    """Save response to cache"""
    with open(cache_path, 'wb') as f:
        pickle.dump(response, f)

def load_from_cache(cache_path):
    """Load response from cache"""
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Update the generate_with_gemini function to match kvcache.py
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry_error_callback=lambda retry_state: ""
)
def generate_with_gemini(prompt, max_tokens=300, chunk_size=120000):
    """Generate text using Gemini API with proper delay handling"""
    try:
        # Split long prompts into chunks if needed
        words = prompt.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        prompt = chunks[0] if chunks else prompt
        
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        
        # Add delay to avoid rate limits
        sleep(2)  # Increased to 2 seconds between requests
        
        if not response.text:
            print("Warning: Empty response received")
            sleep(3)  # Additional delay for empty responses
            return ""
            
        return response.text
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        if "429" in str(e):  # Rate limit error
            print("Rate limit hit, waiting longer...")
            sleep(5)  # Longer delay for rate limits
        else:
            sleep(2)  # Normal error delay
        return ""

# Rest of your existing retriever implementations
def getOpenAIRetriever(documents: list[str], similarity_top_k: int = 1):
    """OpenAI RAG model"""
    import openai
    openai.api_key = get_env()["OPENAI_API_KEY"]        
    
    from llama_index.embeddings.openai import OpenAIEmbedding
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", 
                                         api_key=get_env()["OPENAI_API_KEY"])
    
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    
    return OpenAI_retriever, t2 - t1

def getBM25Retriever(documents: list[str], similarity_top_k: int = 1):
    from llama_index.core.node_parser import SentenceSplitter  
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    splitter = SentenceSplitter(chunk_size=512)
    
    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    t2 = time()
    
    return bm25_retriever, t2 - t1

def parse_squad_data(raw):
    dataset = { "ki_text": [], "qas": [] }
    
    for k_id, data in enumerate(raw['data']):
        article = []
        for p_id, para in enumerate(data['paragraphs']):
            article.append(para['context'])
            for qa in para['qas']:
                if qa['answers']:  # Only include QAs with answers
                    ques = qa['question']
                    answers = [ans['text'] for ans in qa['answers']]
                    dataset['qas'].append({"title": data['title'], 
                                         "paragraph_index": tuple((k_id, p_id)),
                                         "question": ques, 
                                         "answers": answers})
        dataset['ki_text'].append({"id": k_id, "title": data['title'], "paragraphs": article})
    
    return dataset

def get_squad_dataset(filepath: str, max_knowledge: int = None, max_paragraph: int = None, max_questions: int = None):
    # Open and read the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = parse_squad_data(data)
    
    print("max_knowledge", max_knowledge, "max_paragraph", max_paragraph, "max_questions", max_questions)
    
    # Set the limit Maximum Articles
    max_knowledge = max_knowledge if max_knowledge != None and max_knowledge < len(parsed_data['ki_text']) else len(parsed_data['ki_text'])
    
    k_ids = []
    text_list = []
    # Get the knowledge Articles
    for article in parsed_data['ki_text'][:max_knowledge]:
        k_ids.append(article['id'])
        max_para = max_paragraph if max_paragraph != None and max_paragraph < len(article['paragraphs']) else len(article['paragraphs'])
        text_list.append(article['title'])
        text_list.append('\n'.join(article['paragraphs'][0:max_para]))
    
    # Get QA pairs
    questions = []
    answers = []
    for qa in parsed_data['qas']:
        if qa['paragraph_index'][0] in k_ids and (max_paragraph == None or qa['paragraph_index'][1] < max_paragraph):
            questions.append(qa['question'])
            answers.append(qa['answers'][0])
    
    if max_questions:
        questions = questions[:max_questions]
        answers = answers[:max_questions]
    
    dataset = zip(questions, answers)
    return text_list, dataset

def get_hotpotqa_dataset(datapath: str, max_knowledge: int = None, random_seed: int = None):
    """Load HotpotQA dataset"""
    # Load the dataset
    with open(datapath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle if random seed is provided
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(data)
    
    # Limit the number of knowledge items if specified
    if max_knowledge is not None:
        data = data[:max_knowledge]
    
    text_list = []
    dataset = []
    
    for item in data:
        # Extract supporting facts and question-answer pairs
        context = ""
        for title, sentences in item['context']:
            context += f"\nTitle: {title}\n"
            context += "\n".join(sentences)
        
        text_list.append(Document(text=context))
        dataset.append((item['question'], item['answer']))
    
    return text_list, dataset

def rag_test(args: argparse.Namespace):
    """Main testing function"""
    cumulative_similarity = 0
    cumulative_retrieve_time = 0
    cumulative_generate_time = 0
    
    # Set random seed if provided
    if args.randomSeed is not None:
        random.seed(args.randomSeed)
    
    # Load dataset based on the specified dataset type
    if args.dataset == "hotpotqa-train":
        datapath = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge, args.randomSeed)
    elif args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v2.0.json"
        text_list, dataset = get_squad_dataset(datapath, args.maxKnowledge, args.maxParagraph, args.maxQuestion)
    
    # Convert text_list to documents
    documents = [doc if isinstance(doc, Document) else Document(text=doc) for doc in text_list]
    
    # Initialize retriever based on index type
    if args.index == "openai":
        retriever, prepare_time = getOpenAIRetriever(documents, args.topk)
    elif args.index == "bm25":
        retriever, prepare_time = getBM25Retriever(documents, args.topk)
    
    print(f"Retriever {args.index} prepared in {prepare_time} seconds")
    
    # Convert dataset to list and limit questions if specified
    dataset = list(dataset)
    if args.maxQuestion:
        dataset = dataset[:args.maxQuestion]
    
    # Process each question
    for i, (question, ground_truth) in enumerate(dataset):
        try:
            # Add delay between batches of questions
            if i > 0 and i % 10 == 0:
                print("Taking a break to avoid rate limits...")
                sleep(5)  # 5 second break every 10 questions
            
            t1 = time()
            retrieved_nodes = retriever.retrieve(question)
            retrieve_time = time() - t1
            
            # Construct prompt with retrieved context
            context = "\n".join([node.text for node in retrieved_nodes])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            
            # Generate response with retry mechanism
            generate_t1 = time()
            response = generate_with_gemini(prompt)
            generate_time = time() - generate_t1
            
            if not response:  # If we got an empty response
                print(f"Skipping question {i} due to empty response")
                continue
                
            # Calculate similarity
            similarity = get_bert_similarity(response, ground_truth)
            
            # Update cumulative metrics
            cumulative_similarity += similarity
            cumulative_retrieve_time += retrieve_time
            cumulative_generate_time += generate_time
            
            # Print current result
            print(f"[{i}]: Semantic Similarity: {similarity:.5f},\tretrieve time: {retrieve_time},\tgenerate time: {generate_time}")
            print(f"Q: {question}")
            print(f"A: {response}")
            print(f"GT: {ground_truth}")
            
            # Print cumulative results
            print(f"[{i}]: [Cumulative]: Semantic Similarity: {cumulative_similarity/(i+1):.5f},\t retrieve time: {cumulative_retrieve_time/(i+1)},\t generate time: {cumulative_generate_time/(i+1)}")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            sleep(2)
            continue
    
    # Print final results
    print("\nFinal Results:")
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {cumulative_similarity/len(dataset)}")
    print(f"Average retrieve time: {cumulative_retrieve_time/len(dataset)}")
    print(f"Average generate time: {cumulative_generate_time/len(dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with Gemini integration")
    parser.add_argument('--index', choices=['openai', 'bm25'], required=True, 
                       help='Index to use (openai or bm25)')
    parser.add_argument('--dataset', required=True, 
                       choices=['kis', 'kis_sample', 'squad-dev', 'squad-train',
                               'hotpotqa-dev', 'hotpotqa-train', 'hotpotqa-test'],
                       help='Dataset to use')
    parser.add_argument('--similarity', choices=['bertscore'], required=True,
                       help='Similarity metric to use')
    parser.add_argument('--output', required=True, type=str,
                       help='Output file to save the results')
    parser.add_argument('--maxQuestion', type=int, default=None,
                       help='Maximum number of questions to test')
    parser.add_argument('--maxKnowledge', type=int, default=None,
                       help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', type=int, default=None,
                       help='Maximum number of paragraph to use')
    parser.add_argument('--topk', type=int, default=1,
                       help='Top K retrievals to use')
    parser.add_argument('--randomSeed', type=int, default=None,
                       help='Random seed to use')
    
    args = parser.parse_args()
    
    print("maxKnowledge:", args.maxKnowledge, 
          "maxParagraph:", args.maxParagraph, 
          "maxQuestion:", args.maxQuestion, 
          "randomSeed:", args.randomSeed)
    
    # Create unique output filename if file exists
    def unique_path(path, i=0):
        if os.path.exists(path):
            return unique_path(path + "_" + str(i), i + 1)
        return path
    
    if os.path.exists(args.output):
        args.output = unique_path(args.output)
        
    rag_test(args) 
import os
import google.generativeai as genai
from time import time, sleep
import pandas as pd
import argparse
import json
import random
from sentence_transformers import SentenceTransformer, util
from llama_index.core import Document
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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: ""
)
def generate_with_gemini(prompt, knowledge="", max_tokens=300, chunk_size=120000, use_cache=True):
    """Generate text using Gemini API with knowledge context and caching"""
    try:
        # Check cache first if enabled
        if use_cache:
            cache_path = get_cache_path(prompt, knowledge)
            cached_response = load_from_cache(cache_path)
            if cached_response is not None:
                print(f"Cache HIT - Using cached response")
                return cached_response, True  # Return tuple (response, is_cached)
        
        print(f"Cache MISS - Generating new response with API")
        # If not in cache, generate new response
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        
        # Add appropriate delays for API calls
        if "429" in str(response):  # Rate limit error
            print("Rate limit hit - waiting longer...")
            sleep(10)  # 10 second delay for rate limits
            return "", False  # Return empty response to trigger retry
        else:
            sleep(3)  # 3 second delay between normal API calls
        
        # Cache the new response
        if use_cache and response.text:
            save_to_cache(cache_path, response.text)
        
        return response.text, False  # Return tuple (response, is_cached)
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        sleep(5)  # 5 second delay on error
        return "", False

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

def get_hotpotqa_dataset(datapath: str, max_knowledge: int = None, random_seed: int = None):
    """Load HotpotQA dataset"""
    with open(datapath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(data)
    
    if max_knowledge is not None:
        data = data[:max_knowledge]
    
    knowledge_list = []
    dataset = []
    
    for item in data:
        context = ""
        for title, sentences in item['context']:
            context += f"\nTitle: {title}\n"
            context += "\n".join(sentences)
        
        knowledge_list.append(context)
        dataset.append((item['question'], item['answer']))
    
    return knowledge_list, dataset

def get_squad_dataset(filepath: str, max_knowledge: int = None, random_seed: int = None):
    """Load SQuAD dataset"""
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Extract articles and QA pairs
    articles = []
    qa_pairs = []
    
    for article in data['data']:
        context = f"Title: {article['title']}\n\n"
        for paragraph in article['paragraphs']:
            context += paragraph['context'] + "\n\n"
            for qa in paragraph['qas']:
                if qa['answers']:  # Only include QAs with answers
                    qa_pairs.append((qa['question'], qa['answers'][0]['text']))
        articles.append(context)
    
    # Apply random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        combined = list(zip(articles, [qa_pairs[i:i+len(qa_pairs)//len(articles)] 
                                     for i in range(0, len(qa_pairs), len(qa_pairs)//len(articles))]))
        random.shuffle(combined)
        articles, qa_groups = zip(*combined)
        qa_pairs = [qa for group in qa_groups for qa in group]
    
    # Limit knowledge if specified
    if max_knowledge is not None:
        articles = articles[:max_knowledge]
        qa_pairs = qa_pairs[:max_knowledge * (len(qa_pairs)//len(articles))]
    
    return articles, qa_pairs

def cag_test(args):
    """Main testing function for CAG"""
    results = {
        "prompts": [],
        "responses": [],
        "cache_time": [],
        "generate_time": [],
        "similarity": []
    }
    
    # Set random seed if provided
    if args.randomSeed is not None:
        random.seed(args.randomSeed)
    
    # Load dataset
    if args.dataset == "hotpotqa-train":
        datapath = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        knowledge_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge, args.randomSeed)
    elif args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v2.0.json"
        knowledge_list, dataset = get_squad_dataset(datapath, args.maxKnowledge, args.randomSeed)
    
    # Prepare knowledge context - Measure actual preparation time
    prepare_t1 = time()
    combined_knowledge = "\n\n==========\n\n".join(knowledge_list)
    
    # Create initial chat session with knowledge context
    if not args.usePrompt:
        chat = model.start_chat(history=[])
        # Prime the chat with knowledge context
        chat.send_message(f"Here is the knowledge context to use for answering questions:\n{combined_knowledge}")
    prepare_t2 = time()
    prepare_time = prepare_t2 - prepare_t1
    
    print(f"Knowledge preparation completed in {prepare_time:.2f} seconds")
    with open(args.output, "a") as f:
        f.write(f"Knowledge preparation completed in {prepare_time:.2f} seconds\n")
    
    # Limit number of questions if specified
    dataset = list(dataset)
    max_questions = args.maxQuestion if args.maxQuestion is not None else len(dataset)
    dataset = dataset[:max_questions]
    
    # Process each question
    total_questions = len(dataset)
    cache_hits = 0
    cache_misses = 0
    
    for i, (question, ground_truth) in enumerate(dataset):
        try:
            # Process all cache hits first
            cache_t1 = time()
            cache_path = get_cache_path(question)
            cached_response = load_from_cache(cache_path)
            cache_t2 = time()
            
            # Generate response
            generate_t1 = time()
            response, is_cached = generate_with_gemini(
                f"Question: {question}\nAnswer:",
                use_cache=True
            )
            generate_t2 = time() - generate_t1
            
            # If we got an empty response, retry after a delay
            retry_count = 0
            while not response and retry_count < 3:
                print(f"Empty response received, retrying ({retry_count + 1}/3)...")
                sleep(5)  # Wait 5 seconds before retry
                response, is_cached = generate_with_gemini(
                    f"Question: {question}\nAnswer:",
                    use_cache=True
                )
                retry_count += 1
            
            if not response:
                print(f"Skipping question {i} after {retry_count} failed attempts")
                continue
            
            # Update cache statistics
            if is_cached:
                cache_hits += 1
                generate_t2 = 0.001  # Minimal time for cached responses
            else:
                cache_misses += 1
            
            # Calculate similarity
            similarity = get_bert_similarity(response, ground_truth)
            
            # Print results with cache status
            cache_status = "HIT" if is_cached else "MISS"
            print(f"\n[{i}/{total_questions}]: [CACHE {cache_status}] Semantic Similarity: {round(similarity, 5)},",
                  f"cache time: {cache_t2 - cache_t1:.4f},",
                  f"generate time: {generate_t2:.4f}")
            
            # Add batch delay only for cache misses
            if not is_cached and i > 0 and i % 5 == 0:
                print(f"\nProcessed {i}/{total_questions} questions")
                print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
                print("Taking a break to avoid rate limits...")
                sleep(5)  # 5 second break every 5 cache misses
            
            # Store results
            results["prompts"].append(question)
            results["responses"].append(response)
            results["cache_time"].append(cache_t2 - cache_t1)
            results["generate_time"].append(generate_t2)
            results["similarity"].append(similarity)
            
            # Log to file with more detailed information
            with open(args.output, "a") as f:
                f.write(f"[{i}/{total_questions}]: [CACHE {cache_status}] Semantic Similarity: {round(similarity, 5)},\t"
                       f"cache time: {cache_t2 - cache_t1:.4f},\t"
                       f"generate time: {generate_t2:.4f}\n")
                f.write(f"Q: {question}\n")
                f.write(f"A: {response}\n")
                f.write(f"GT: {ground_truth}\n")
                f.write("-" * 80 + "\n")
                
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            sleep(5)
            continue
    
    # Print final cache statistics
    print(f"\nFinal Cache Statistics:")
    print(f"Total questions: {total_questions}")
    print(f"Cache hits: {cache_hits} ({(cache_hits/total_questions)*100:.2f}%)")
    print(f"Cache misses: {cache_misses} ({(cache_misses/total_questions)*100:.2f}%)")
    
    # Calculate and log final statistics
    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    
    print(f"\nFinal Results:")
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"Average cache time: {avg_cache_time}")
    print(f"Average generate time: {avg_generate_time}")
    
    with open(args.output, "a") as f:
        f.write(f"\nFinal Results:\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"Average cache time: {avg_cache_time}\n")
        f.write(f"Average generate time: {avg_generate_time}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAG test with Groq integration")
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
    parser.add_argument('--usePrompt', action='store_true',
                       help='Use without knowledge cache')
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
    
    cag_test(args)

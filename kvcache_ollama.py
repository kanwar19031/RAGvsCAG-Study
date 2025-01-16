import os
from time import time, sleep
import pandas as pd
import argparse
import json
import random
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential
import pickle
import hashlib
from pathlib import Path
from ollama import chat
from ollama import ChatResponse

def get_env():
    env_dict = {}
    with open(file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict

"""Sentence-BERT for evaluate semantic similarity"""
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_similarity(response, ground_truth):
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)
    return cosine_score.item()

def get_cache_path(question, knowledge=""):
    """Generate a unique cache path based on question and knowledge"""
    cache_key = f"{question}_{knowledge}"
    filename = hashlib.md5(cache_key.encode()).hexdigest() + ".pkl"
    return os.path.join("data_cache", filename)

def save_to_cache(cache_path, response):
    """Save response to cache"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(response, f)

def load_from_cache(cache_path):
    """Load response from cache"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
    return None

def initialize_ollama(model_name):
    """Initialize Ollama model"""
    try:
        from ollama import pull
        print(f"Ensuring model {model_name} is pulled and ready...")
        pull(model_name)
        print("Model ready!")
    except Exception as e:
        print(f"Warning: Could not initialize model: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: ""
)
def generate_with_ollama(prompt, knowledge="", max_tokens=300, use_cache=True):
    """Generate text using Ollama with caching"""
    try:
        # Check cache first if enabled
        if use_cache:
            cache_path = get_cache_path(prompt, knowledge)
            cached_response = load_from_cache(cache_path)
            if cached_response is not None:
                print(f"Cache HIT - Using cached response")
                return cached_response, True
        
        print(f"Cache MISS - Generating new response")
        
        # Format the complete prompt
        formatted_prompt = format_prompt(prompt, knowledge)
        
        # Format messages for chat
        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful AI assistant that provides concise, accurate answers based on the given context.'
            },
            {
                'role': 'user',
                'content': formatted_prompt
            }
        ]
        
        # Call Ollama API with optimized parameters
        response: ChatResponse = chat(
            model="llama3.2",
            messages=messages,
            options={
                'temperature': 0,
                'num_predict': max_tokens,
                'top_k': 10,
                'top_p': 0.7,
                'repeat_penalty': 1.1,
                'num_ctx': 1024,
                'num_thread': 8,
                'stop': ["</s>", "Question:", "Context:"]
            }
        )
        
        result = response.message.content.strip()
        
        # Cache the new response
        if use_cache and result:
            save_to_cache(cache_path, result)
        
        return result, False
        
    except Exception as e:
        print(f"Error generating with Ollama: {e}")
        sleep(2)
        return "", False

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

def get_squad_dataset(datapath: str, max_knowledge: int = None, max_paragraph: int = None, max_question: int = None, random_seed: int = None):
    """Load SQuAD dataset"""
    with open(datapath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if random_seed is not None:
        random.seed(random_seed)
    
    knowledge_list = []
    dataset = []
    
    for article in data['data']:
        if max_knowledge and len(knowledge_list) >= max_knowledge:
            break
            
        for paragraph in article['paragraphs']:
            if max_paragraph and len(knowledge_list) >= max_paragraph:
                break
                
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                if max_question and len(dataset) >= max_question:
                    break
                    
                if not qa['is_impossible']:
                    question = qa['question']
                    answer = qa['answers'][0]['text']
                    
                    knowledge_list.append(context)
                    dataset.append((question, answer))
    
    if random_seed is not None:
        combined = list(zip(knowledge_list, dataset))
        random.shuffle(combined)
        knowledge_list, dataset = zip(*combined)
    
    return knowledge_list, dataset

def format_prompt(question, knowledge=""):
    """Format prompt for Llama model"""
    if knowledge:
        return f"""Context: {knowledge}

Question: {question}

Please provide a concise answer based on the given context."""
    else:
        return f"""Question: {question}

Please provide a concise answer."""

def cag_test(args):
    """Main testing function for CAG"""
    # Initialize metrics
    cache_hits = 0
    cache_misses = 0
    results = {
        "prompts": [],
        "responses": [],
        "cache_time": [],
        "generate_time": [],
        "similarity": []
    }
    
    # Initialize model
    initialize_ollama(args.model)
    
    # Load dataset
    if args.dataset == "hotpotqa-train":
        datapath = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        knowledge_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge, args.randomSeed)
    elif args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v2.0.json"
        knowledge_list, dataset = get_squad_dataset(datapath, args.maxKnowledge, args.maxParagraph, args.maxQuestion, args.randomSeed)
    
    # Process questions in batches
    batch_size = 3  # Small batch size for CPU
    total_questions = len(dataset)
    if args.maxQuestion:
        dataset = dataset[:args.maxQuestion]
        total_questions = len(dataset)
    
    for i in range(0, total_questions, batch_size):
        batch = dataset[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_questions-1)//batch_size + 1}")
        
        for question_idx, (question, ground_truth) in enumerate(batch):
            current_idx = i + question_idx
            try:
                print(f"\nProcessing question {current_idx + 1}/{total_questions}:")
                print(f"Q: {question}")
                
                # Measure cache time
                cache_t1 = time()
                response, is_cached = generate_with_ollama(
                    prompt=question,
                    knowledge=knowledge_list[current_idx] if not args.usePrompt else "",
                    use_cache=True
                )
                cache_t2 = time()
                
                # Update cache statistics
                if is_cached:
                    cache_hits += 1
                    generate_t2 = 0  # No generation time for cache hits
                else:
                    cache_misses += 1
                    generate_t2 = cache_t2 - cache_t1  # Generation time for cache misses
                
                # Calculate similarity
                similarity = get_bert_similarity(response, ground_truth)
                
                # Print results
                print(f"A: {response}")
                print(f"GT: {ground_truth}")
                print(f"\nResults for question {current_idx + 1}:")
                print(f"Cache Status: {'HIT' if is_cached else 'MISS'}")
                print(f"Semantic Similarity: {similarity:.5f}")
                print(f"Cache time: {cache_t2 - cache_t1:.2f}s")
                if not is_cached:
                    print(f"Generate time: {generate_t2:.2f}s")
                
                # Store results
                results["prompts"].append(question)
                results["responses"].append(response)
                results["cache_time"].append(cache_t2 - cache_t1)
                results["generate_time"].append(generate_t2)
                results["similarity"].append(similarity)
                
                # Print cumulative metrics
                avg_similarity = sum(results["similarity"]) / len(results["similarity"])
                avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
                avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
                
                print(f"\nCumulative metrics after {current_idx + 1} questions:")
                print(f"Average Similarity: {avg_similarity:.5f}")
                print(f"Average Cache time: {avg_cache_time:.2f}s")
                print(f"Average Generate time: {avg_generate_time:.2f}s")
                print(f"Cache hits: {cache_hits}/{current_idx + 1} ({(cache_hits/(current_idx + 1))*100:.2f}%)")
                print("-" * 80)
                
            except Exception as e:
                print(f"Error processing question {current_idx + 1}: {e}")
                sleep(2)
                continue
        
        # Print batch summary
        print(f"\nBatch {i//batch_size + 1} Summary:")
        print(f"Total processed: {current_idx + 1}/{total_questions}")
        print("=" * 80)
    
    # Print final statistics
    print(f"\nFinal Cache Statistics:")
    print(f"Total questions: {total_questions}")
    print(f"Cache hits: {cache_hits} ({(cache_hits/total_questions)*100:.2f}%)")
    print(f"Cache misses: {cache_misses} ({(cache_misses/total_questions)*100:.2f}%)")
    print(f"Average Semantic Similarity: {sum(results['similarity'])/len(results['similarity']):.5f}")
    print(f"Average Cache time: {sum(results['cache_time'])/len(results['cache_time']):.2f}s")
    print(f"Average Generate time: {sum(results['generate_time'])/len(results['generate_time']):.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAG test with Ollama")
    parser.add_argument('--dataset', required=True,
                       choices=['hotpotqa-train', 'squad-train'],
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
                       help='Maximum number of paragraphs to use (for SQuAD)')
    parser.add_argument('--model', type=str, default="llama3.2",
                       help='Ollama model to use')
    parser.add_argument('--usePrompt', action='store_true',
                       help='Use without knowledge cache')
    parser.add_argument('--randomSeed', type=int, default=None,
                       help='Random seed to use')
    
    args = parser.parse_args()
    
    print("Configuration:")
    print(f"Model: {args.model}")
    print(f"Max Knowledge: {args.maxKnowledge}")
    print(f"Max Questions: {args.maxQuestion}")
    print(f"Use Prompt Only: {args.usePrompt}")
    print(f"Random Seed: {args.randomSeed}")
    
    cag_test(args) 
"""
RAG (Retrieval-Augmented Generation) implementation using Ollama.
Optimized for CPU-based local inference with Llama 3.2 3B model.
"""

# Standard library imports
import os
import json
import random
from time import time, sleep

# Third-party imports
import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer, util
from llama_index.core import VectorStoreIndex, Document, Settings
from tenacity import retry, stop_after_attempt, wait_exponential
from ollama import chat, ChatResponse

# Initialize models
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_env():
    """Load environment variables from .env file"""
    env_dict = {}
    with open(file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        return {k: v.strip('"') for k, v in (line.strip().split("=") for line in f)}

def get_bert_similarity(response, ground_truth):
    """Calculate semantic similarity using BERT embeddings"""
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)
    return util.pytorch_cos_sim(query_embedding, text_embedding).item()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: ""
)
def generate_with_ollama(prompt, model_name="llama3.2", max_tokens=300):
    """Generate text using Ollama chat API with CPU optimizations"""
    try:
        messages = [
            {'role': 'system', 'content': 'Answer concisely based on the context.'},
            {'role': 'user', 'content': prompt}
        ]
        
        response: ChatResponse = chat(
            model=model_name,
            messages=messages,
            options={
                'temperature': 0,
                'num_predict': max_tokens,
                'top_k': 10,
                'top_p': 0.7,
                'repeat_penalty': 1.1,
                'num_ctx': 1024,
                'num_thread': 8,
                'stop': ["</s>"]
            }
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"Error generating with Ollama: {e}")
        sleep(1)
        return ""

# Rest of the retriever implementations remain the same as in rag.py
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
    bm25_retriever.persist("./bm25_retriever")

    return bm25_retriever, t2 - t1

# Dataset loading functions remain the same
def get_hotpotqa_dataset(datapath: str, max_knowledge: int = None, random_seed: int = None):
    """Load HotpotQA dataset"""
    with open(datapath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(data)
    
    if max_knowledge is not None:
        data = data[:max_knowledge]
    
    text_list = []
    dataset = []
    
    for item in data:
        context = ""
        for title, sentences in item['context']:
            context += f"\nTitle: {title}\n"
            context += "\n".join(sentences)
        
        text_list.append(Document(text=context))
        dataset.append((item['question'], item['answer']))
    
    return text_list, dataset

def get_squad_dataset(datapath: str, max_knowledge: int = None, max_paragraph: int = None, max_question: int = None):
    """Load SQuAD dataset with specified limits"""
    with open(datapath, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    
    if max_knowledge is not None:
        data = data[:max_knowledge]
    
    text_list = []
    dataset = []
    
    for article in data:
        for paragraph in article['paragraphs'][:max_paragraph]:
            context = paragraph['context']
            text_list.append(Document(text=context))
            
            for qa in paragraph['qas'][:max_question]:
                if not qa.get('is_impossible', False):  # Skip impossible questions
                    dataset.append((qa['question'], qa['answers'][0]['text']))
    
    return text_list, dataset

def initialize_ollama(model_name):
    """Initialize Ollama with CPU optimizations"""
    try:
        from ollama import pull
        print(f"Ensuring model {model_name} is pulled and ready...")
        pull(model_name)
        print("Model ready!")
    except Exception as e:
        print(f"Warning: Could not initialize model: {e}")

def rag_test(args: argparse.Namespace):
    """Main testing function"""
    cumulative_similarity = 0
    cumulative_retrieve_time = 0
    cumulative_generate_time = 0
    
    # Set random seed if provided
    if args.randomSeed is not None:
        random.seed(args.randomSeed)
    
    # Load dataset
    if args.dataset == "hotpotqa-train":
        datapath = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge, args.randomSeed)
    elif args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v2.0.json"
        text_list, dataset = get_squad_dataset(datapath, args.maxKnowledge, args.maxParagraph, args.maxQuestion)
    
    # Convert text_list to documents
    documents = [doc if isinstance(doc, Document) else Document(text=doc) for doc in text_list]
    
    # Initialize retriever
    if args.index == "openai":
        retriever, prepare_time = getOpenAIRetriever(documents, args.topk)
    elif args.index == "bm25":
        retriever, prepare_time = getBM25Retriever(documents, args.topk)
    
    print(f"Retriever {args.index} prepared in {prepare_time} seconds")
    
    # Initialize model once at startup
    initialize_ollama(args.model)
    
    # Process questions in smaller batches for CPU
    batch_size = 3  # Reduced batch size
    total_processed = 0
    
    dataset = list(dataset)
    if args.maxQuestion:
        dataset = dataset[:args.maxQuestion]
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_responses = []
        
        print(f"\nProcessing batch {i//batch_size + 1}/{len(dataset)//batch_size + 1}")
        
        for question_idx, (question, ground_truth) in enumerate(batch):
            current_idx = i + question_idx
            print(f"\nProcessing question {current_idx + 1}/{len(dataset)}:")
            print(f"Q: {question}")
            
            try:
                # Retrieval step
                t1 = time()
                retrieved_nodes = retriever.retrieve(question)
                retrieve_time = time() - t1
                print(f"Retrieved context in {retrieve_time:.2f}s")
                
                # Generation step
                context = "\n".join([node.text for node in retrieved_nodes])
                prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                
                print("Generating response...")
                t1 = time()
                response = generate_with_ollama(prompt, model_name=args.model)
                generate_time = time() - t1
                print(f"Generated response in {generate_time:.2f}s")
                print(f"A: {response}")
                print(f"GT: {ground_truth}")
                
                # Calculate similarity immediately
                similarity = get_bert_similarity(response, ground_truth)
                
                # Update cumulative metrics
                cumulative_similarity += similarity
                cumulative_retrieve_time += retrieve_time
                cumulative_generate_time += generate_time
                
                # Print immediate results
                print(f"\nResults for question {current_idx + 1}:")
                print(f"Semantic Similarity: {similarity:.5f}")
                print(f"Retrieve time: {retrieve_time:.2f}s")
                print(f"Generate time: {generate_time:.2f}s")
                
                # Print cumulative metrics
                print(f"\nCumulative metrics after {current_idx + 1} questions:")
                print(f"Average Similarity: {cumulative_similarity/(current_idx+1):.5f}")
                print(f"Average Retrieve time: {cumulative_retrieve_time/(current_idx+1):.2f}s")
                print(f"Average Generate time: {cumulative_generate_time/(current_idx+1):.2f}s")
                print("-" * 80)
                
                # Store response for batch summary
                batch_responses.append({
                    'question': question,
                    'response': response,
                    'ground_truth': ground_truth,
                    'retrieve_time': retrieve_time,
                    'generate_time': generate_time,
                    'similarity': similarity
                })
                
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing question {current_idx + 1}: {e}")
                continue
        
        # Print batch summary
        print(f"\nBatch {i//batch_size + 1} Summary:")
        print(f"Processed {len(batch_responses)}/{len(batch)} questions")
        print(f"Total processed: {total_processed}/{len(dataset)}")
        print("=" * 80)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {cumulative_similarity/len(dataset)}")
    print(f"Average retrieve time: {cumulative_retrieve_time/len(dataset)}")
    print(f"Average generate time: {cumulative_generate_time/len(dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with Ollama integration")
    parser.add_argument('--index', choices=['openai', 'bm25'], required=True, 
                       help='Index to use (openai or bm25)')
    parser.add_argument('--dataset', required=True, 
                       choices=['squad-train', 'hotpotqa-train'],
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
    parser.add_argument('--model', type=str, default="llama2:latest",
                       help='Ollama model to use (default: llama2:latest)')
    parser.add_argument('--randomSeed', type=int, default=None,
                       help='Random seed to use')
    
    args = parser.parse_args()
    
    print("Configuration:")
    print(f"Model: {args.model}")
    print(f"Max Knowledge: {args.maxKnowledge}")
    print(f"Max Paragraph: {args.maxParagraph}")
    print(f"Max Questions: {args.maxQuestion}")
    print(f"Random Seed: {args.randomSeed}")
    
    rag_test(args) 
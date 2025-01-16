import os
from groq import Groq
from time import time
import pandas as pd
import argparse
import json
import random
from sentence_transformers import SentenceTransformer, util
from llama_index.core import VectorStoreIndex, Document, Settings

def get_env():
    env_dict = {}
    with open(file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict

# Initialize Groq client
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

"""Sentence-BERT for evaluate semantic similarity"""
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_similarity(response, ground_truth):
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)
    return cosine_score.item()

def generate_with_groq(prompt, max_tokens=300):
    """Generate text using Groq's API"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Set to 0 for deterministic output
            max_tokens=max_tokens,
            stream=False  # Changed to False for simpler handling
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating with Groq: {e}")
        return ""

# Rest of the retriever implementations remain the same
def getOpenAIRetriever(documents: list[str], similarity_top_k: int = 1):
    """OpenAI RAG model"""
    import openai
    openai.api_key = get_env()["OPENAI_API_KEY"]        
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-3.5-turbo")
    
    from llama_index.embeddings.openai import OpenAIEmbedding
    # Set the embed_model in llama_index
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=get_env()["OPENAI_API_KEY"], title="openai-embedding")
    # model_name: "text-embedding-3-small", "text-embedding-3-large"
    
    # Create the OpenAI retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    
    return OpenAI_retriever, t2 - t1
    

def getGeminiRetriever(documents: list[str], similarity_top_k: int = 1):
    """Gemini Embedding RAG model"""
    GOOGLE_API_KEY = get_env()["GOOGLE_API_KEY"]
    from llama_index.embeddings.gemini import GeminiEmbedding
    model_name = "models/embedding-001"
    # Set the embed_model in llama_index
    Settings.embed_model = GeminiEmbedding( model_name=model_name, api_key=GOOGLE_API_KEY, title="gemini-embedding")
    
    # Create the Gemini retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    Gemini_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    
    return Gemini_retriever, t2 - t1
    
def getBM25Retriever(documents: list[str], similarity_top_k: int = 1):
    from llama_index.core.node_parser import SentenceSplitter  
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    splitter = SentenceSplitter(chunk_size=512)
    
    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)
    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    t2 = time()
    bm25_retriever.persist("./bm25_retriever")

    return bm25_retriever, t2 - t1

def get_kis_dataset(filepath: str):
    df = pd.read_csv(filepath)
    dataset = zip(df['sample_question'], df['sample_ground_truth'])
    text_list = df["ki_text"].to_list()
    
    return text_list, dataset

def parse_squad_data(raw):
    dataset = { "ki_text": [], "qas": [] }
    
    for k_id, data in enumerate(raw['data']):
        article = []
        for p_id, para in enumerate(data['paragraphs']):
            article.append(para['context'])
            for qa in para['qas']:
                ques = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                dataset['qas'].append({"title": data['title'], "paragraph_index": tuple((k_id, p_id)) ,"question": ques, "answers": answers})
        dataset['ki_text'].append({"id": k_id, "title": data['title'], "paragraphs": article})
    
    return dataset

def get_squad_dataset(filepath: str, max_knowledge: int = None, max_paragraph: int = None, max_questions: int = None):
    # Open and read the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = parse_squad_data(data)
    
    print("max_knowledge", max_knowledge, "max_paragraph", max_paragraph, "max_questions", max_questions)
    
    # Set the limit Maximum Articles, use all Articles if max_knowledge is None or greater than the number of Articles
    max_knowledge = max_knowledge if max_knowledge != None and max_knowledge < len(parsed_data['ki_text']) else len(parsed_data['ki_text'])
    
    # Shuffle the Articles and Questions
    if random_seed != None:
        random.seed(random_seed)
        random.shuffle(parsed_data["ki_text"])
        random.shuffle(parsed_data["qas"])
        k_ids = [i['id'] for i in parsed_data["ki_text"][:max_knowledge]]

        
    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data['ki_text'][:max_knowledge]:
        max_para = max_paragraph if max_paragraph != None and max_paragraph < len(article['paragraphs']) else len(article['paragraphs'])
        text_list.append(article['title'])
        text_list.append('\n'.join(article['paragraphs'][0:max_para]))
    
    # Check if the knowledge id of qas is less than the max_knowledge
    questions = [qa['question'] for qa in parsed_data['qas'] if qa['paragraph_index'][0] in k_ids and (max_paragraph == None or qa['paragraph_index'][1] < max_paragraph)]
    answers = [qa['answers'][0] for qa in parsed_data['qas'] if qa['paragraph_index'][0]  in k_ids and (max_paragraph == None or qa['paragraph_index'][1] < max_paragraph)]
    
    dataset = zip(questions, answers)
    
    return text_list, dataset


def get_hotpotqa_dataset(datapath: str, max_knowledge: int = None, random_seed: int = None):
    """Load HotpotQA dataset"""
    import json
    
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
        t1 = time()
        retrieved_nodes = retriever.retrieve(question)
        retrieve_time = time() - t1
        
        # Construct prompt with retrieved context
        context = "\n".join([node.text for node in retrieved_nodes])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # Generate response
        t1 = time()
        response = generate_with_groq(prompt)
        generate_time = time() - t1
        
        # Calculate similarity
        similarity = get_bert_similarity(response, ground_truth)
        
        # Update cumulative metrics
        cumulative_similarity += similarity
        cumulative_retrieve_time += retrieve_time
        cumulative_generate_time += generate_time
        
        # Print current result
        print(f"[{i}]: Semantic Similarity: {similarity:.5f},\tretrieve time: {retrieve_time},\tgenerate time: {generate_time}")
        
        # Print cumulative results
        print(f"[{i}]: [Cumulative]: Semantic Similarity: {cumulative_similarity/(i+1):.5f},\t retrieve time: {cumulative_retrieve_time/(i+1)},\t generate time: {cumulative_generate_time/(i+1)}")
    
    # Print final results
    print("\nFinal Results:")
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {cumulative_similarity/len(dataset)}")
    print(f"Average retrieve time: {cumulative_retrieve_time/len(dataset)}")
    print(f"Average generate time: {cumulative_generate_time/len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with Groq integration")
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
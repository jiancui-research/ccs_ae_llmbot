import pandas as pd
import json
import os 
import logging
import time
import argparse
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import google.generativeai as genai
import anthropic
import cohere
from openai import OpenAI
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from tqdm import tqdm
from pdb import set_trace
from rouge_score import rouge_scorer

# Initialize rouge scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

prompts = {
    "gemini": {
        "user_prompt": "Relying soly on your memorization during training, generate the sentence that immediately follows the provided sentence. The next sentence MUST include the following words: {force_words}. The next sentence MUST NOT include any of these words: {bad_words}. Please output only the next sentence without additional text or prologue. The given sentence is: ",
        "system_prompt": None
    },
    "anthropic": {
        "user_prompt": "The next sentence MUST include the following words: {force_words}. The next sentence MUST NOT include any of these words: {bad_words}. The given sentence is: ",
        "system_prompt": "Relying soly on your memorization during training, generate the sentence that immediately follows the provided sentence. Please output only the next sentence without additional text or prologue."
    },
    "openai": {
        "user_prompt": "The next sentence MUST include the following words: {force_words}. The next sentence MUST NOT include any of these words: {bad_words}. The given sentence is: ",
        "system_prompt": "Relying soly on your memorization during training, generate the sentence that immediately follows the provided sentence. Please output only the next sentence without additional text or prologue."
    },
    "cohere": {
        "user_prompt": "Relying soly on your memorization during training, generate the sentence that immediately follows the provided sentence. The next sentence MUST include the following words: {force_words}. The next sentence MUST NOT include any of these words: {bad_words}. Please output only the next sentence without additional text or prologue. The given sentence is: ",
        "system_prompt": None
    }
}



genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
cohere_client = cohere.Client(os.environ.get('COHERE_API_KEY'))
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Functions to get responses from models
def get_gemini_response(model_name, system_prompt, user_prompt, temperature):
    logging.info(f"Retrieving {model_name} response...")

    try:
        gemini_model = genai.GenerativeModel(f'{model_name}', generation_config={"temperature": temperature})
        messages = [{'role': 'user', 'parts': [user_prompt]}]
        response = gemini_model.generate_content(messages)
        return response.text
    except Exception as e:
        logging.error(f"Failed to get response from Gemini: {e}")
        return None

def get_anthropic_response(model_name, system_prompt, user_prompt, temperature):
    logging.info(f"Retrieving {model_name} response...")

    try:
        messages = [{"role": "user", "content": user_prompt}]
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=512,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        logging.error(f"Failed to get response from Anthropic: {e}")
        return None

def get_cohere_response(model_name, system_prompt, user_prompt, temperature):
    logging.info(f"Retrieving {model_name} response...")

    try:
        response = cohere_client.chat(model=model_name, message=user_prompt, temperature=temperature)
        return response.text
    except Exception as e:
        logging.error(f"Failed to get response from Cohere: {e}")
        return None

def get_openai_response(model_name, system_prompt, user_prompt, temperature):
    print(f"Retrieving {model_name} response...")
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}"},
        ],
        model=model_name,
        temperature=temperature,
    )
    response = chat_completion.choices[0].message.content

    return response

def extract_important_words(sentence, max_words):
    """Extract important content words from a sentence, filtering out stop words."""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    # Define important parts of speech to keep (nouns, verbs, adjectives, adverbs)
    important_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    force_words_with_pos = [(word, tag) for word, tag in tagged if tag in important_pos]
    stop_words = set(stopwords.words('english'))
    force_words = [word for word, tag in force_words_with_pos if word.lower() not in stop_words]

    return force_words

def get_bad_words(response, true_sentence, max_words):
    """Identify words in the model response that don't appear in the true sentence."""
    # Extract words from response and true sentence
    true_sentence_words = set(word_tokenize(true_sentence))
    response_words = word_tokenize(response)
    
    # Get important words from response that aren't in true sentence
    response_words_tagged = pos_tag(response_words)
    stop_words = set(stopwords.words('english'))
    important_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    
    bad_words = [
        word for word, tag in response_words_tagged 
        if tag in important_pos
        and word.lower() not in stop_words
        and word.lower() not in {w.lower() for w in true_sentence_words}  # Case-insensitive comparison
    ]
    
    return bad_words

parser = argparse.ArgumentParser(description='Process the context window size.')
# parser.add_argument('--fill_missing', action='store_true', help='Fill missing responses')
parser.add_argument('--gemini_model', type=str, default='gemini-1.5-flash', help='Gemini model name')
parser.add_argument('--anthropic_model', type=str, default='claude-3-haiku-20240307', help='Anthropic model name')
parser.add_argument('--cohere_model', type=str, default='command-r-08-2024', help='Cohere model name')
parser.add_argument('--openai_model', type=str, default='gpt-4o-mini', help='OpenAI model name')
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--temperature', type=float, default=0.01, help='Temperature')
parser.add_argument('--word_list_max', type=int, default=10, help='Maximum number of words in force and bad word lists')

args = parser.parse_args()

# Set the output filename and input filename based on the rule
out_fname = f'results/closedsource_sent_responses.xlsx'
input_fname = f'../data/sent_completion_disallow_2023-05-10.json'

# Load existing responses to skip already processed IDs
try:
    existing_responses = pd.read_excel(out_fname)
    processed_ids = set(existing_responses['ID'])
except Exception as e:
    logging.error("Failed to load existing responses, processing all entries.")
    existing_responses = pd.DataFrame()
    processed_ids = set()

# Load input JSON based on the rule
with open(input_fname, 'r') as f:
    data = json.load(f)

# Main execution loop
responses = []
for domain, documents in tqdm(data.items()):
    for doc_id, doc_data in tqdm(documents.items(), total=len(documents), desc=f"Processing {domain}"):
        unique_id = f"{domain}_{doc_id}"
        if unique_id in processed_ids:
            logging.info(f'{unique_id} has already been processed!')
            continue
        
        input_sent = doc_data['context_sentence']
        true_next_sent = doc_data['next_sentence_true']
        full_article = doc_data['full']
        
        # Extract force words from true next sentence
        force_words = extract_important_words(true_next_sent, args.word_list_max)
        
        # Initialize model-specific bad words
        openai_bad_words = []
        gemini_bad_words = []
        anthropic_bad_words = []
        cohere_bad_words = []
        
        responses_dict = {
            'ID': unique_id,
            'domain': domain,
            'category': doc_data['category'],
            'created_at': doc_data['time'],
            'context_sentence': input_sent,
            'next_sentence_true': doc_data['next_sentence_true']
        }

        # Track best responses and scores
        best_responses = {
            'openai': {'response': None, 'rouge_l': -1},
            'gemini': {'response': None, 'rouge_l': -1},
            'anthropic': {'response': None, 'rouge_l': -1},
            'cohere': {'response': None, 'rouge_l': -1}
        }

        # Iterate multiple times to refine responses
        for iteration in range(args.iterations):
            logging.info(f"Processing iteration {iteration + 1}/{args.iterations} for document {unique_id}")
            
            # Get responses from different models
            if args.openai_model and best_responses['openai']['rouge_l'] < 0.9:
                formatted_user_prompt = prompts['openai']['user_prompt'].format(
                    force_words=force_words,
                    bad_words=openai_bad_words
                ) + input_sent
                openai_response = get_openai_response(args.openai_model, prompts['openai']['system_prompt'], 
                                                    formatted_user_prompt, args.temperature)
                if openai_response:
                    rouge_l = scorer.score(true_next_sent, openai_response)['rougeL'].fmeasure
                    if rouge_l > best_responses['openai']['rouge_l']:
                        best_responses['openai']['response'] = openai_response
                        best_responses['openai']['rouge_l'] = rouge_l
                    new_bad_words = get_bad_words(openai_response, true_next_sent, args.word_list_max)
                    openai_bad_words = list(set(openai_bad_words + new_bad_words))

            if args.gemini_model and best_responses['gemini']['rouge_l'] < 0.9:
                formatted_user_prompt = prompts['gemini']['user_prompt'].format(
                    force_words=force_words,
                    bad_words=gemini_bad_words
                ) + input_sent
                gemini_response = get_gemini_response(args.gemini_model, prompts['gemini']['system_prompt'],
                                                    formatted_user_prompt, args.temperature)
                if gemini_response:
                    rouge_l = scorer.score(true_next_sent, gemini_response)['rougeL'].fmeasure
                    if rouge_l > best_responses['gemini']['rouge_l']:
                        best_responses['gemini']['response'] = gemini_response
                        best_responses['gemini']['rouge_l'] = rouge_l
                    new_bad_words = get_bad_words(gemini_response, true_next_sent, args.word_list_max)
                    gemini_bad_words = list(set(gemini_bad_words + new_bad_words))

            if args.anthropic_model and best_responses['anthropic']['rouge_l'] < 0.9:
                formatted_user_prompt = prompts['anthropic']['user_prompt'].format(
                    force_words=force_words,
                    bad_words=anthropic_bad_words
                ) + input_sent
                anthropic_response = get_anthropic_response(args.anthropic_model, prompts['anthropic']['system_prompt'],
                                                        formatted_user_prompt, args.temperature)
                if anthropic_response:
                    rouge_l = scorer.score(true_next_sent, anthropic_response)['rougeL'].fmeasure
                    if rouge_l > best_responses['anthropic']['rouge_l']:
                        best_responses['anthropic']['response'] = anthropic_response
                        best_responses['anthropic']['rouge_l'] = rouge_l
                    new_bad_words = get_bad_words(anthropic_response, true_next_sent, args.word_list_max)
                    anthropic_bad_words = list(set(anthropic_bad_words + new_bad_words))

            if args.cohere_model and best_responses['cohere']['rouge_l'] < 0.9:
                formatted_user_prompt = prompts['cohere']['user_prompt'].format(
                    force_words=force_words,
                    bad_words=cohere_bad_words
                ) + input_sent
                cohere_response = get_cohere_response(args.cohere_model, prompts['cohere']['system_prompt'],
                                                    formatted_user_prompt, args.temperature)
                if cohere_response:
                    rouge_l = scorer.score(true_next_sent, cohere_response)['rougeL'].fmeasure
                    if rouge_l > best_responses['cohere']['rouge_l']:
                        best_responses['cohere']['response'] = cohere_response
                        best_responses['cohere']['rouge_l'] = rouge_l
                    new_bad_words = get_bad_words(cohere_response, true_next_sent, args.word_list_max)
                    cohere_bad_words = list(set(cohere_bad_words + new_bad_words))

            
            time.sleep(1)  # Rate limiting between iterations
        
        # Add responses to responses_dict
        if args.openai_model and best_responses['openai']['response']:
            responses_dict[f'{args.openai_model}_response'] = best_responses['openai']['response']
            responses_dict[f'{args.openai_model}_rouge_l'] = best_responses['openai']['rouge_l']
            
        if args.gemini_model and best_responses['gemini']['response']:
            responses_dict[f'{args.gemini_model}_response'] = best_responses['gemini']['response']
            responses_dict[f'{args.gemini_model}_rouge_l'] = best_responses['gemini']['rouge_l']
            
        if args.anthropic_model and best_responses['anthropic']['response']:
            responses_dict[f'{args.anthropic_model}_response'] = best_responses['anthropic']['response']
            responses_dict[f'{args.anthropic_model}_rouge_l'] = best_responses['anthropic']['rouge_l']
            
        if args.cohere_model and best_responses['cohere']['response']:
            responses_dict[f'{args.cohere_model}_response'] = best_responses['cohere']['response']
            responses_dict[f'{args.cohere_model}_rouge_l'] = best_responses['cohere']['rouge_l']
        
        responses.append(responses_dict)
        # Periodic saving every 2 documents
        if len(responses) % 2 == 0:
            logging.info("Periodic saving...")
            df_responses = pd.DataFrame(responses)
            df_final = pd.concat([existing_responses, df_responses], ignore_index=True)
            df_final.to_excel(out_fname, index=False)
            logging.info("Saved responses to Excel.")


        time.sleep(1)  # Rate limiting between documents

# Final save
df_responses = pd.DataFrame(responses)
df_final = pd.concat([existing_responses, df_responses], ignore_index=True)
df_final.to_excel(out_fname, index=False)
logging.info("Finished processing all documents.")
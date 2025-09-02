import re
import json
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging

from pdb import set_trace
# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

def precompile_rules(robots_txt_data):
    compiled_rules = {}
    for bot, rules in robots_txt_data.items():
        # Initialize compiled rules storage
        compiled_rules[bot] = {'allow': [], 'disallow': []}
        
        # Process each rule type (allow, disallow)
        for rule_type in ['allow', 'disallow']:
            # Initialize a set to hold processed rules for deduplication
            processed_rules = set()
            
            # Deduplicate and normalize rules
            for rule in rules.get(rule_type, []):
                # Check if rule starts with an asterisk and replace it with '/*' if true
                if rule.startswith('*'):
                    rule = '/' + rule
                
                # Normalize generic patterns by collapsing multiple asterisks to a single one
                normalized_rule = re.sub(r'\*+', '*', rule)
                
                # Add the normalized rule to the set (automatically deduplicates)
                processed_rules.add(normalized_rule)
            
            # Compile regexes for each processed and deduplicated rule
            compiled_rules[bot][rule_type] = [
                re.compile('^' + re.escape(rule).replace('\\*', '.*')) for rule in processed_rules
            ]
    
    return compiled_rules

def is_path_accessible(bot_name, path, robots_txt_data):
    def matches_rule(path, rules):
        for rule in rules:
            if re.match(rule, path):
                return True
        return False

    allow_rules = robots_txt_data.get(bot_name, {}).get('allow', [])
    disallow_rules = robots_txt_data.get(bot_name, {}).get('disallow', [])
    
    if matches_rule(path, disallow_rules) and not matches_rule(path, allow_rules):
        return False

    if bot_name not in robots_txt_data or (not allow_rules and not disallow_rules):
        universal_allow = robots_txt_data.get('*', {}).get('allow', [])
        universal_disallow = robots_txt_data.get('*', {}).get('disallow', [])
        
        if matches_rule(path, universal_disallow) and not matches_rule(path, universal_allow):
            return False

    return True

def get_bias(bot_info):
    DIR = set()
    compiled_bot_info = precompile_rules(bot_info)
    # print(compiled_bot_info)
    for rules in bot_info.values():
        for path_list in rules.values():
            DIR.update(path_list)

    Du = DIR.copy() if '*' not in bot_info else set(bot_info['*']['allow'])
    for dir_ in DIR:
        if is_path_accessible('*', dir_, compiled_bot_info):
            Du.add(dir_)

    bias_scores = {}
    for robot, rules in bot_info.items():
        Dr = [dir_ for dir_ in DIR if is_path_accessible(robot, dir_, compiled_bot_info)]
        
        bias = len(Dr) - len(Du)
        bias_category = 'favored' if bias > 0 else 'disfavored' if bias < 0 else 'no bias'
        
        bias_scores[robot] = {'bias_score': bias, 'category': bias_category}

    return bias_scores

bias_scores_file_path = '../measurement_data/bias_scores.json'
try:
    with open(bias_scores_file_path, 'r') as f:
        existing_bias_scores = json.load(f)
except FileNotFoundError:
    existing_bias_scores = {}
    logging.info("Bias scores file not found. Creating a new one.")

url2botinfo_path = '../data_prep/data_processed/url2botinfo_20250407.json'
with open(url2botinfo_path, 'r') as f:
    url2botinfo = json.load(f)

def save_progress(progress_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(progress_data, f, indent=4)

save_interval = 50
timeout_duration = 60 
blacklist_urls = set([ # exclude some urls due to large amount of bots are listed
    'coliss.com', 
    ])

with ThreadPoolExecutor(max_workers=5) as executor:
    for i, url in tqdm(enumerate(url2botinfo.keys()), total=len(url2botinfo)):
        logging.info(f"{i}, {url}")
        if url in blacklist_urls:
            logging.info(f"{url} in blacklist")
            continue
        if url not in existing_bias_scores:
            future = executor.submit(get_bias, url2botinfo[url])
            try:
                bias_scores = future.result(timeout=timeout_duration)
                existing_bias_scores[url] = bias_scores
            except TimeoutError:
                logging.error(f"TimeoutError: get_bias function took longer than {timeout_duration} seconds for URL: {url}")
                continue  # Skip this URL and move to the next one

            if len(existing_bias_scores) % save_interval == 0:
                logging.info(f"Saving progress at {i} URLs...")
                save_progress(existing_bias_scores, bias_scores_file_path)
        else:
            logging.info(f'{url} saved, skipping...')

save_progress(existing_bias_scores, bias_scores_file_path)
logging.info("All updates saved successfully.")

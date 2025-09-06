import os
from transformers import AutoTokenizer, OPTForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import torch
import copy
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import json
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readjson(filepath):
    data = {}
    with open(filepath,'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def savejson(filepath, datas):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)

def generateTextTopN(model, tokenizer, prompt, originalText, force_words_ids, bad_words_ids):
    original_tokenizer_ids = tokenizer(originalText, add_special_tokens=False).input_ids
    seq_len = len(original_tokenizer_ids)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids

    if len(bad_words_ids) == 0:
        bad_words_ids.append([])
    if len(force_words_ids[0]) == 0 and len(bad_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=seq_len,
            num_beams=20,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=20,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    elif len(force_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            bad_words_ids=bad_words_ids,
            max_new_tokens=seq_len,
            num_beams=20,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=20,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    elif len(bad_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            force_words_ids=force_words_ids,
            max_new_tokens=seq_len,
            num_beams=20,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=20,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            force_words_ids=force_words_ids,
            bad_words_ids=bad_words_ids,
            max_new_tokens=seq_len,
            num_beams=20,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=20,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    generate_texts = []
    for mm in range(0, 20):
        generate_text = tokenizer.decode(outputs[mm], skip_special_tokens=True)
        generate_text = generate_text.replace(prompt, '')
        generate_text = generate_text.replace('\n', ' ')
        generate_text = generate_text.strip()
        generate_texts.append(generate_text)
    return generate_texts

def main(datasetPath, modelName, access_token, savePath):
    # load data
    allDatas = readjson(datasetPath)
    # select your model and tokenizer
    tokenizer = None
    model = None
    if model == 'gpt2-xl':
        tokenizer = GPT2Tokenizer.from_pretrained(modelName)
        model = GPT2LMHeadModel.from_pretrained(modelName).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(modelName, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(modelName, token=access_token).to(device)

    # select your dataset
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    i = 0
    j = 0
    for website in allDatas:
        datas = allDatas[website]
        allResults = {}
        results = {}
        for filename in datas:
            data = datas[filename]
            promptText, originalText, context_tokenizer_ids = getPromptWithTime(data, tokenizer)

            try:
                original_tokenizer_ids = tokenizer(originalText, add_special_tokens=False).input_ids
                bad_words_dict = {}
                # Bad words list
                i = 0
                bad_words_ids = []
                for i in range(0, tokenizer.vocab_size):
                    if i not in context_tokenizer_ids:
                        bad_words_dict[i] = 1
                for i in bad_words_dict:
                    bad_words_ids.append([i])
                bad_words_dict = None
                vocab = None
                temps = originalText.split(' ')
                # Temporary forced words
                words = {}
                for i in range(0, len(temps) - 1):
                    words[temps[i] + ' ' + temps[i + 1]] = 1

                best_generate_text = ''
                best_similarity = -1000
                best_word = ''
                small62 = False
                for i in range(0, 9):
                    if small62:
                        break
                    if best_similarity < 0.99:
                        force_words_ids = []
                        force_words_ids.append([])
                        # Generate Top N with constraints
                        generate_texts = generateTextTopN(model, tokenizer, promptText, originalText, force_words_ids,
                                                          bad_words_ids)
                        temp_bad_words_ids = copy.deepcopy(bad_words_ids)
                        for generate_text in generate_texts:
                            # temp_bad_words_ids = copy.deepcopy(bad_words_ids)
                            generate_text_ids = tokenizer(generate_text, add_special_tokens=False).input_ids
                            for generate_text_id in generate_text_ids:
                                if generate_text_id not in original_tokenizer_ids:
                                    if [generate_text_id] not in temp_bad_words_ids:
                                        temp_bad_words_ids.append([generate_text_id])
                            # Computer similarity
                            scores = scorer.score(originalText, generate_text)
                            rougeL = scores['rougeL']
                            similarity_score = rougeL.fmeasure
                            if similarity_score > best_similarity:
                                best_generate_text = generate_text
                                best_similarity = similarity_score
                        if tokenizer.vocab_size - len(temp_bad_words_ids) <= 62:
                            small62 = True
                            break
                        bad_words_ids = copy.deepcopy(temp_bad_words_ids)
                        temp_bad_words_ids = None
                    else:
                        break
                result = {}
                result['category'] = data['category']
                result['context_sentence'] = promptText
                result['next_sentence_true'] = originalText
                result['full'] = data['full']
                result['next_sentence_generated'] = best_generate_text
                result['words_length'] = tokenizer.vocab_size - len(bad_words_ids)
                result['similarity'] = best_similarity
                results[filename] = result
            except Exception as e:
                print(promptText + '**************' + originalText + '**************' + best_generate_text)
                print(best_similarity)
                print(e)
                torch.cuda.empty_cache()
                continue
        allResults[website] = results

    savejson(savePath, allResults)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constrained generation & ROUGE evaluation pipeline")
    parser.add_argument("datasetPath", help="Path to input JSON dataset")
    parser.add_argument("modelName", help="HF model id, e.g., 'gpt2-xl'")
    parser.add_argument("access_token", nargs="?", default=None, help="HF access token (optional for private models)")
    parser.add_argument("savePath", help="Path to output JSON file")
    args = parser.parse_args()

    main(args.datasetPath, args.modelName, args.access_token, args.savePath)
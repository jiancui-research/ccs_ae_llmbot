import os
from transformers import AutoTokenizer, OPTForCausalLM
import torch
import copy
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readlines(filePath):
    lines = []
    with open(filePath, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def saveFile(filePath,lines):
    with open(filePath,'a', encoding="utf-8") as f:
        for line in lines:
            f.write(line)

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
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    elif len(force_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            bad_words_ids=bad_words_ids,
            max_new_tokens=seq_len,
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    elif len(bad_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            force_words_ids=force_words_ids,
            max_new_tokens=seq_len,
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            force_words_ids=force_words_ids,
            bad_words_ids=bad_words_ids,
            max_new_tokens=seq_len,
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    generate_texts = []
    for mm in range(0, 40):
        generate_text = tokenizer.decode(outputs[mm], skip_special_tokens=True)
        generate_text = generate_text.replace(prompt, '')
        generate_text = generate_text.replace('\n', ' ')
        generate_text = generate_text.strip()
        generate_texts.append(generate_text)
    return generate_texts


def generateText(model, tokenizer, prompt, originalText, force_words, bad_words_ids):
    original_tokenizer_ids = tokenizer(originalText, add_special_tokens=False).input_ids
    seq_len = len(original_tokenizer_ids)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

    if len(bad_words_ids) == 0:
        bad_words_ids.append([])
    if len(force_words_ids[0]) == 0 and len(bad_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=seq_len,
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    elif len(force_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            bad_words_ids=bad_words_ids,
            max_new_tokens=seq_len,
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    elif len(bad_words_ids[0]) == 0:
        outputs = model.generate(
            input_ids=input_ids,
            force_words_ids=force_words_ids,
            max_new_tokens=seq_len,
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            force_words_ids=force_words_ids,
            bad_words_ids=bad_words_ids,
            max_new_tokens=seq_len,
            num_beams=40,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=40,
            no_repeat_ngram_size=1,
            early_stopping=True
        )
    generate_text = generate_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generate_text

def getPrompt(filePath, tokenizer):
    lines = readlines(filePath)
    promptText = lines[1].replace('\n','')
    promptText = promptText.replace('\\n',' ')
    originalText = lines[2].replace('\n','')
    originalText = originalText.replace('\\n', ' ')
    context_tokenizer_ids = {}
    context = lines[3].replace('\n','')
    context = context.replace('\\n',' ')
    sentences = sent_tokenize(context)
    for sentence in sentences:
        line_tokenizer_ids = tokenizer(sentence, add_special_tokens=False).input_ids
        for line_tokenizer_id in line_tokenizer_ids:
            context_tokenizer_ids[line_tokenizer_id] = 1
    return promptText,originalText,context_tokenizer_ids

def main():
    # select your model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", add_prefix_space=True)
    model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", return_dict=True).to(device)
    # select your dataset
    savePath = './result/news_opt_result.txt'
    filedir = './data/news/'
    filenames = os.listdir(filedir)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    i = 0
    j = 0
    for filename in filenames:
        n = int(filename.replace('.txt',''))
        filePath = filedir+filename
        promptText, originalText, context_tokenizer_ids = getPrompt(filePath, tokenizer)
        try:
            original_tokenizer_ids = tokenizer(originalText, add_special_tokens=False).input_ids
            bad_words_dict = {}
            #Bad words list
            i = 0
            for i in range(0, tokenizer.vocab_size):
                if i not in context_tokenizer_ids:
                    bad_words_dict[i]=1
            for i in bad_words_dict:
                bad_words_list.append([i])
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
            bad_words_ids = copy.copy(bad_words_list)
            # Get the best forced words
            for word in words:
                force_words = [word]
                similarity_score = 0
                try:
                    generate_text = generateText(model, tokenizer, promptText, originalText, force_words,
                                                 bad_words_ids)
                    # Compute similarity
                    scores = scorer.score(originalText, generate_text)
                    rougeL = scores['rougeL']
                    similarity_score = rougeL.fmeasure
                    if similarity_score > best_similarity:
                        best_generate_text = generate_text
                        best_similarity = similarity_score
                        best_word = word
                        
                except Exception as e:
                    print(e)
                    continue

            for i in range(0, 10):
                if best_similarity < 0.99:
                    force_words = [best_word]
                    force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids
                    # Generate Top N with constraints
                    generate_texts = generateTextTopN(model, tokenizer, promptText, originalText, force_words_ids,
                                                 bad_words_ids)
                    for generate_text in generate_texts:
                        generate_text_ids = tokenizer(generate_text, add_special_tokens=False).input_ids
                        for generate_text_id in generate_text_ids:
                            if generate_text_id not in original_tokenizer_ids:
                                if [generate_text_id] not in bad_words_ids:
                                    bad_words_ids.append([generate_text_id])
                        # Computer similarity
                        scores = scorer.score(originalText, generate_text)
                        rougeL = scores['rougeL']
                        similarity_score = rougeL.fmeasure
                        if similarity_score > best_similarity:
                            best_generate_text = generate_text
                            best_similarity = similarity_score
                else:
                    break
            alllines = []
            newline = promptText + '**************' + originalText + '**************' + best_generate_text + '**************' + str(
                force_words) + '**************' + str(best_similarity) + '\n'
            alllines.append(newline)
            saveFile(savePath, alllines)

        except Exception as e:
            print(promptText + '**************' + originalText + '**************' + best_generate_text)
            print(best_similarity)
            torch.cuda.empty_cache()
            continue
main()












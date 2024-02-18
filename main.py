import os
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch, helpers
import openai
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

directory_path = 'C:\\Users\\petru\\PycharmProjects\\dataMining\\Resources\\wikiData'
filename = 'C:\\Users\\petru\\PycharmProjects\\dataMining\\Resources\\questions.txt'
openai.api_key = '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'


def rename_files():
    global directory_path
    for i, filename_directory in enumerate(os.listdir(directory_path)):
        if filename_directory.endswith('.txt'):
            filename_base, _ = os.path.splitext(filename_directory)
            filename_parts = filename_base.split('xml-')
            new_filename = f"{filename_parts[0]}xml-{i}.txt"
            try:
                old_file_path = os.path.join(directory_path, filename_directory)
                new_file_path = os.path.join(directory_path, new_filename)
                os.rename(old_file_path, new_file_path)
            except Exception as e:
                print(f"Refactored")


rename_files()


def advanced_normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def load_jeopardy_questions(filename):
    questions = []
    with open(filename, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
        for i in range(0, len(lines), 3):
            category, clue, answer = lines[i:i + 3]
            questions.append({'category': category, 'clue': clue, 'answer': answer})
    return questions


def load_wikipedia_pages(directory_path):
    wikipedia_pages = []
    page_pattern = re.compile(r'\[\[(.*?)\]\](.*?)\n(?=\[\[|$)', re.DOTALL)
    for i in range(80):
        filename = os.path.join(directory_path, f'enwiki-20140602-pages-articles.xml-{i}.txt')
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            for match in page_pattern.finditer(content):
                title, page_content = match.groups()
                wikipedia_pages.append({"title": title.strip(), "content": page_content.strip()})
    return wikipedia_pages


def index_wikipedia_pages(es, index_name, wikipedia_pages):
    actions = [
        {
            "_index": index_name,
            "_source": {
                "title": page['title'],
                "content": advanced_normalize_text(page['content'])
            }
        }
        for page in wikipedia_pages
    ]
    helpers.bulk(es, actions)
    logging.info("Finished bulk indexing Wikipedia pages.")


request_counter = [0]


def search_and_rerank_with_chatgpt(es, index_name, clue):
    global request_counter

    response = es.search(index=index_name, body={
        "query": {
            "match": {
                "content": advanced_normalize_text(clue)
            }
        },
        "size": 5
    })
    candidate_titles = [hit['_source']['title'] for hit in response['hits']['hits']]

    prompt = f"Given the Jeopardy clue '{clue}', rank these Wikipedia pages by relevance:\n"
    for i, title in enumerate(candidate_titles, start=1):
        prompt += f"{i}. {title}\n"
    prompt += "Rank the pages by their numbers."


    if request_counter[0] >= 2:
        logging.info("Pausing for 22 seconds to manage API rate limits...")
        time.sleep(22)
        request_counter[0] = 0
    else:
        request_counter[0] += 1

    try:
        ranked_response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=60,
            n=1,
            stop=None,
            temperature=0.3
        )
        ranked_list = ranked_response.choices[0].text.strip()
        ranked_indices = [int(num) for num in re.findall(r'\d+', ranked_list)]
        ranked_titles = [candidate_titles[i - 1] for i in ranked_indices if i - 1 < len(candidate_titles)]
        if request_counter[0] >= 2:
            logging.info("Pausing for 22 seconds to manage API rate limits...")
            time.sleep(22)
            request_counter[0] = 0
        else:
            request_counter[0] += 1
        return ranked_titles
    except Exception as e:
        logging.error(f"Error in reranking with ChatGPT: {e}")
        return candidate_titles


def evaluate_match_es_for_p_at_1(es, index_name, questions):
    correct_at_1 = 0

    for question in questions:
        ranked_titles = search_and_rerank_with_chatgpt(es, index_name, question['clue'])
        is_correct = ranked_titles[0].lower() == question['answer'].lower() if ranked_titles else False

        if is_correct:
            correct_at_1 += 1

        print(
            f"Clue: {question['clue']}, Top Ranked Answer: {ranked_titles[0] if ranked_titles else 'No match found'}, Correct Answer: {question['answer']}, Match Status: {'Correct' if is_correct else 'Incorrect'}")

    precision_at_1 = correct_at_1 / len(questions) if questions else 0
    print(f"Precision at 1 with Reranking: {precision_at_1:.2f}")


def main():
    es = Elasticsearch(hosts=["http://192.168.0.150:9200"])
    index_name = "jeopardy_wikipedia"
    questions = load_jeopardy_questions(filename)
    # wikipedia_pages = load_wikipedia_pages(directory_path)

    # index_wikipedia_pages(es, index_name, wikipedia_pages)
    evaluate_match_es_for_p_at_1(es, index_name, questions)


if __name__ == "__main__":
    main()

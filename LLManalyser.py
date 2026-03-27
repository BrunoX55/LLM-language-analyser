import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def init_client():
    """
    Initierar Anthropic-klienten med API-nyckel.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set ANTHROPIC_API_KEY correctly.")
    return Anthropic(api_key=api_key)


def get_responses(client, questions, model="claude-2", max_tokens=1000):
    """
    Hämtar svar från AI för en lista med frågor.
    Returnerar en lista med svar.
    """
    responses = []
    for q in questions:
        try:
            response = client.completions.create(
                model=model,
                prompt=f"{HUMAN_PROMPT}{q}{AI_PROMPT}",
                max_tokens_to_sample=max_tokens,
                stop_sequences=[HUMAN_PROMPT]
            )
            responses.append(response.completion)
        except Exception as e:
            print(f"Error for question '{q}': {e}")
            responses.append("")
    return responses


def analyze_texts(texts):
    """
    Räknar ord, meningar och ordfrekvens.
    Returnerar totalord, totalsentens, genomsnittlig ordlängd, meningslängd och ordlista.
    """
    all_words = []
    total_word_count = 0
    total_sentence_count = 0

    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        total_word_count += len(words)
        all_words.extend(words)

        sentences = re.split(r'[.!?]', text)
        total_sentence_count += len([s for s in sentences if s.strip()])

    average_word_count = total_word_count / len(texts) if texts else 0
    average_sentence_length = total_word_count / total_sentence_count if total_sentence_count else 0
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(100)

    return {
        "total_words": total_word_count,
        "total_sentences": total_sentence_count,
        "average_words_per_response": average_word_count,
        "average_sentence_length": average_sentence_length,
        "common_words": common_words
    }


def save_to_csv(common_words, filename="top_words.csv"):
    """
    Sparar top 100 ord och deras frekvens till CSV.
    """
    df = pd.DataFrame(common_words, columns=["word", "count"])
    df.to_csv(filename, index=False)
    print(f"Saved top words to {filename}")


def plot_wordcloud(common_words, filename="wordcloud.png"):
    """
    Genererar och sparar ett ordmoln från ordlistan.
    """
    word_dict = dict(common_words)
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(filename)
    plt.show()
    print(f"Saved word cloud to {filename}")


def main():
    client = init_client()
    questions = ["Write a text about apple", "Write a text about book", "Write a text about car"]
    responses = get_responses(client, questions)

    analysis = analyze_texts(responses)
    print(f"Average words per response: {analysis['average_words_per_response']:.2f}")
    print(f"Average sentence length: {analysis['average_sentence_length']:.2f}")

    save_to_csv(analysis["common_words"])
    plot_wordcloud(analysis["common_words"])


if __name__ == "__main__":
    main()

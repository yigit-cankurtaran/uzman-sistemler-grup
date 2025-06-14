import json
import random
from collections import defaultdict

# config: kaç soru sorulacak (max 36)
NUM_QUESTIONS = 25
QUIZ_SOURCE = "sorular.json"
RESULT_OUTPUT = "cevaplar.json"

def load_questions(path):
    with open(path, 'r') as f:
        return json.load(f)

def select_random_questions(questions, n):
    if n > len(questions):
        raise ValueError("İstenen soru sayısı mevcut sorulardan fazla.")
    return random.sample(questions, n)

def run_quiz(questions):
    results = defaultdict(list)

    print("RIASEC Test Başlıyor. Lütfen her soruyu 1 (kesinlikle katılmıyorum) ile 5 (kesinlikle katılıyorum) arasında puanlayın.\n")
    
    for idx, q in enumerate(questions, 1):
        # idx = index, q = json içindeki obje
        while True:
            try:
                score = int(input(f"{idx}. {q['question']} [1-5]: "))
                if score < 1 or score > 5:
                    raise ValueError
                break
            except ValueError:
                print("Lütfen 1 ile 5 arasında bir tam sayı girin.")
        
        results[q['dimension']].append(score) # skorumuzu boyutla ekle, dictionary

    return results

def compute_scores(results):
    # son skor için ortalama
    avg_scores = {dim: round(sum(scores) / len(scores), 2) for dim, scores in results.items()}
    return avg_scores

def save_results(score_dict, path):
    with open(path, 'w') as f:
        json.dump(score_dict, f, indent=2)
    print(f"\n✅ Skorlar '{path}' dosyasına kaydedildi.")

def main():
    all_questions = load_questions(QUIZ_SOURCE)
    selected_questions = select_random_questions(all_questions, NUM_QUESTIONS)
    raw_results = run_quiz(selected_questions)
    avg_scores = compute_scores(raw_results)
    save_results(avg_scores, RESULT_OUTPUT)

if __name__ == "__main__":
    main()


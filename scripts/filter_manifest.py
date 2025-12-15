import json
import random
import os

# --- НАСТРОЙКИ ---
INPUT_FILE = "assets/train_merged.json"  # Твой текущий файл
OUTPUT_FILE = "train_final.json"         # Файл для обучения
MAX_DURATION = 15.0

def filter_and_shuffle():
    print(f"🔪 Читаю {INPUT_FILE}...")
    
    valid_lines = []
    removed_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                duration = item.get('duration', 0)
                
                # Фильтр по длительности
                if duration <= MAX_DURATION:
                    valid_lines.append(line)
                else:
                    removed_count += 1
                    
            except json.JSONDecodeError:
                continue

    print(f"📉 Удалено {removed_count} длинных файлов.")
    print(f"🎲 Перемешиваю {len(valid_lines)} оставшихся записей...")
    random.shuffle(valid_lines)

    print(f"💾 Сохраняю в {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in valid_lines:
            f_out.write(line)

    print("✅ Готово! Используй этот файл для токенизатора и обучения.")
    
    # Небольшой расчет для тебя
    total_hours = 0
    for line in valid_lines[:1000]: # Прикинем по первым 1000
        total_hours += json.loads(line)['duration']
    
    avg = total_hours / 1000
    est_total_hours = (avg * len(valid_lines)) / 3600
    print(f"📊 Итоговый объем данных: ~{est_total_hours:.0f} часов.")

if __name__ == "__main__":
    filter_and_shuffle()

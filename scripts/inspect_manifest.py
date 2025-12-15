import json
import re
from collections import Counter
import sys

MANIFEST_FILE = "assets/train_merged.json"
MAX_DURATION_LIMIT = 15.0

def analyze_manifest(filepath):
    print(f"🔍 Начинаю анализ файла: {filepath} ...")
    
    stats = {
        "total_lines": 0,
        "sources": Counter(),
        "durations": [],
        "over_limit_count": 0,
        "bad_chars": set(),
        "dirty_texts_samples": []
    }
    
    dirty_regex = re.compile(r'[^а-яё ]')

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                stats["total_lines"] += 1
                try:
                    item = json.loads(line)
                    
                    # 1. Анализ путей (откуда данные?)
                    path = item.get('audio_filepath', '').lower()
                    if 'golos' in path or 'sber' in path or 'farfield' in path:
                        stats["sources"]['Golos/Sber'] += 1
                    elif 'sova' in path or 'youtube' in path:
                        stats["sources"]['Sova'] += 1
                    else:
                        stats["sources"]['Other'] += 1

                    # 2. Анализ длительности
                    duration = item.get('duration', 0)
                    stats["durations"].append(duration)
                    if duration > MAX_DURATION_LIMIT:
                        stats["over_limit_count"] += 1

                    # 3. Анализ чистоты текста
                    text = item.get('text', '').lower()
                    bad_chars = dirty_regex.findall(text)
                    
                    if bad_chars:
                        stats["bad_chars"].update(bad_chars)
                        if len(stats["dirty_texts_samples"]) < 5:
                            stats["dirty_texts_samples"].append(text)

                except json.JSONDecodeError:
                    print(f"⚠️ Ошибка JSON на строке {line_num}")
                    continue
                
                # Прогресс бар для спокойствия (каждые 50к строк)
                if stats["total_lines"] % 50000 == 0:
                    print(f"   ...обработано {stats['total_lines']} строк...")

    except FileNotFoundError:
        print(f"❌ Файл {filepath} не найден!")
        return

    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    if stats["total_lines"] == 0:
        print("❌ Файл пуст!")
        return

    total_dur_hours = sum(stats["durations"]) / 3600
    avg_dur = sum(stats["durations"]) / len(stats["durations"])
    max_dur = max(stats["durations"])

    print("\n" + "="*40)
    print(f"📊 ОТЧЕТ ПО МАНИФЕСТУ: {filepath}")
    print("="*40)
    print(f"✅ Всего записей:     {stats['total_lines']}")
    print(f"⏱  Общая длительность: {total_dur_hours:.2f} часов")
    print("-" * 20)
    
    print("📁 Источники данных (по путям файлов):")
    for source, count in stats["sources"].items():
        print(f"   - {source}: {count} ({(count/stats['total_lines'])*100:.1f}%)")
    
    print("-" * 20)
    print("⏳ Длительность аудио:")
    print(f"   - Средняя: {avg_dur:.2f} сек")
    print(f"   - Макс:    {max_dur:.2f} сек")
    
    if stats["over_limit_count"] > 0:
        print(f"⚠️ ВНИМАНИЕ: {stats['over_limit_count']} файлов длиннее {MAX_DURATION_LIMIT} сек!")
        print("   Нужно удалить их перед обучением, иначе OOM Error.")
    else:
        print(f"✅ Все файлы короче {MAX_DURATION_LIMIT} сек.")

    print("-" * 20)
    print("🔤 Чистота текста:")
    if stats["bad_chars"]:
        print(f"⚠️ Найдено {len(stats['bad_chars'])} мусорных символов!")
        print(f"   Примеры мусора: {list(stats['bad_chars'])[:20]}")
        print(f"   Примеры грязных строк:")
        for s in stats["dirty_texts_samples"]:
            print(f"   > '{s}'")
    else:
        print("✅ Текст идеально чистый (только кириллица и пробелы).")
    print("="*40)

if __name__ == "__main__":
    analyze_manifest(MANIFEST_FILE)
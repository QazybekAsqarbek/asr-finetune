import nemo.collections.asr as nemo_asr
import json

# --- НАСТРОЙКИ ---
# Путь берем из твоих логов (последний сохраненный)
# MODEL_PATH = "/tf/askarbek/hw1/experiments/FastConformer-Hybrid-TDT-CTC-BPE/2025-12-06_11-24-19/checkpoints/FastConformer-Hybrid-TDT-CTC-BPE.nemo"
# MODEL_PATH = "/tf/askarbek/hw1/telegram_bot/models/FastConformer-Hybrid-TDT-CTC-BPE.nemo"
MODEL_PATH = "/tf/askarbek/hw1/experiments/FastConformer-Hybrid-TDT-CTC-BPE/2025-12-10_18-03-40/checkpoints/FastConformer-Hybrid-TDT-CTC-BPE.nemo"
TEST_MANIFEST = "f_test.json" # Или f_test_25nov.json (проверь имя файла!)
# OUTPUT_FILE = "finetune_predictions.txt"
OUTPUT_FILE = "finetune_predictions_151225.txt"


def run_inference():
    print(f"🔄 Загружаю модель из: {MODEL_PATH}")
    try:
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=MODEL_PATH)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    print("📄 Читаю список файлов...")
    files = []
    with open(TEST_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            files.append(item['audio_filepath'])

    print(f"🚀 Начинаю распознавание {len(files)} файлов (это займет пару минут)...")
    # batch_size поменьше, чтобы точно не вылетело
    transcriptions = asr_model.transcribe(files, batch_size=16)

    print(f"💾 Сохраняю в {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for hypothesis in transcriptions:
            # У гибридных моделей output может быть сложным, берем текст
            text = hypothesis.text if hasattr(hypothesis, 'text') else hypothesis
            f_out.write(str(text) + "\n")

    print("✅ Готово! Можно скачивать файл.")

if __name__ == "__main__":
    run_inference()
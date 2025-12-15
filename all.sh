# ===== prepare tokenizer ===
# rm -rf tokenizer_final

# python scripts/process_asr_text_tokenizer.py \
#   --manifest="train_final.json" \
#   --data_root="./tokenizer_final/" \
#   --vocab_size=380 \
#   --tokenizer="spe" \
#   --spe_type="bpe" \
#   --spe_max_sentencepiece_length=4 \
#   --log
# ===========================


# ===== train ===============
# python3 scripts/speech_to_text_hybrid_rnnt_ctc_bpe_ru.py
# ===========================


# ===== measure WER ===============
# # 1. make prediction
# python3 scripts/inference.py

# # 2. calculate WER
# python3 scripts/calculate_wer.py
# ==================================
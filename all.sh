python ./scripts/process_asr_text_tokenizer.py \
  --manifest="assets/train_merged.json" \
  --data_root="./tokenizer380/" \
  --vocab_size=380 \
  --tokenizer="spe" \
  --spe_type="bpe" \
  --spe_max_sentencepiece_length=4 \
  --log

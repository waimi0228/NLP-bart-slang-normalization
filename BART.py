import os
import pandas as pd
import nltk
import torch
import evaluate

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# 環境設置 & CUDA
os.environ["OMP_NUM_THREADS"] = "16"
print("CUDA available:", torch.cuda.is_available())
print("Device count   :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device :", torch.cuda.get_device_name(0))

nltk.download('punkt', quiet=True)

def main():
    # 讀取CSV
    data_path = os.path.join("data", "data0519.csv")
    df = pd.read_csv(data_path,
                     sep=",",
                     quotechar='"',
                     engine="python",
                     encoding="utf-8")
    if "_Unnamed: 0" in df.columns:
        df = df.drop(columns="_Unnamed: 0")
    print(df)
    # Dataset
    raw = Dataset.from_pandas(df, preserve_index=False)
    ds  = raw.train_test_split(test_size=0.2) #訓練集測試集 8:2

    # 加载模型與分詞器
    model_name = "fnlp/bart-base-chinese"
    tokenizer  = BertTokenizerFast.from_pretrained(model_name)
    model      = BartForConditionalGeneration.from_pretrained(model_name)

    # 預處理
    def preprocess(batch):
        inputs = tokenizer(batch["input_text"],
                           padding="max_length",
                           truncation=True,
                           max_length=128)
        inputs.pop("token_type_ids", None)

        labels = tokenizer(batch["target_text"],
                           padding="max_length",
                           truncation=True,
                           max_length=128)
        labels.pop("token_type_ids", None)

        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = ds.map(preprocess,
                       batched=True,
                       remove_columns=["input_text", "target_text"])

    # ROUGE 評估指標
    rouge = evaluate.load("rouge")
    def compute_metrics(pred):
        decoded_preds = tokenizer.batch_decode(pred.predictions,
                                               skip_special_tokens=True)
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in lab]
            for lab in pred.label_ids
        ]
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)
        scores = rouge.compute(predictions=decoded_preds,
                               references=decoded_labels)
        return {"rougeL": scores["rougeL"].mid.fmeasure}

    # 定義 DataCollator
    base_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def custom_collator(features):
        batch = base_collator(features)
        batch.pop("token_type_ids", None)
        return batch

    # 設定訓練參數
    training_args = TrainingArguments(
        output_dir="bart_corrector", # 模型與檢查點儲存目錄
        num_train_epochs=5, # 總訓練輪數（完整遍歷訓練集的次數）
        per_device_train_batch_size=5, # 每張 GPU 用於訓練的 batch 大小
        per_device_eval_batch_size=4,  # 每張 GPU用於驗證的 batch 大小

        do_train=True,    # 啟用訓練階段
        do_eval=True,     # 啟用每步/每輪後的驗證
        eval_steps=500,   # 每 500 個步驟執行一次驗證
        save_steps=500,   # 每 500 個步驟儲存一次檢查點

        logging_steps=50,    # 每 50 個步驟輸出一次訓練日誌
        learning_rate=2e-5,  # 優化器的學習率
        warmup_steps=100,    # 前 100 步進行學習率 warm-up
        weight_decay=0.01,   # 權重衰減強度
    )

    # 初始化 Trainer 開始訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=custom_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # 9. 保存模型
    trainer.save_model("bart_corrector_best")
    tokenizer.save_pretrained("bart_corrector_best")

    # 10. 推理测试
    def correct(text: str) -> str:
        device = model.device
        inputs = tokenizer(text,
                           return_tensors="pt",
                           padding="max_length",
                           truncation=True,
                           max_length=128).to(device)
        out_ids = model.generate(inputs.input_ids,
                                 attention_mask=inputs.attention_mask,
                                 max_length=128,
                                 num_beams=4)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print("範例改寫：", correct("又來了啦煩欸"))
    print("範例改寫：", correct("他是I人"))
    print("範例改寫：", correct("我是什麼很賤的人嗎？為什麼來跟我借錢了?"))
    print("範例改寫：", correct("他們給你跪了"))
    print("範例改寫：", correct("干牠"))
    print("範例改寫：", correct("我真的防房了"))


if __name__ == "__main__":
    main()

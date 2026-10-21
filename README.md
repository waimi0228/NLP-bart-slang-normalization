# 中国語ネットスラングの標準語変換（BART）

SNS などで使われる中国語のネットスラングや不自然な表現を、
自然な標準語に書き換える Seq2Seq モデルを作成しました。

| 入力 | 出力 |
|---|---|
| 你這樣真的很母湯 | 你這樣真的很不行 |
| 你踹共 | 你出來說 |

## 手法
- 事前学習済みモデル `fnlp/bart-base-chinese` を fine-tuning
- 自作データセット（誤った表現 → 修正後の表現 のペア）、train : test = 8 : 2
- 評価指標：ROUGE-L
- 学習設定：5 epochs, learning rate 2e-5

## 使用技術
Python / PyTorch / Hugging Face Transformers / datasets / evaluate

## 実行方法
```bash
pip install -r requirements.txt
python BART.py
```

## レポート
詳細は [NLP_report_for_slang_normalization.pdf](NLP_report_for_slang_normalization.pdf) を参照。

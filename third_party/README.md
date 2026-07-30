# Third-party sentiment benchmarks

Shallow clones used only for **adversarial validation** of BIBM social-listening
sentiment labels. They are not imported by the production pipeline or dashboard.

| Directory | Upstream | Role in audit |
|-----------|----------|---------------|
| `sentiment/vaderSentiment` | [cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment) (MIT) | High-star English social/media lexicon baseline |
| `sentiment/xlm-t` | [cardiffnlp/xlm-t](https://github.com/cardiffnlp/xlm-t) | Reference for multilingual Twitter XLM-R sentiment (matches project config model family) |
| `sentiment/banglabert` | [csebuetnlp/banglabert](https://github.com/csebuetnlp/banglabert) | Authoritative Bangla NLU / BLUB (incl. sentiment classification task) |
| `sentiment/bangla-bert` | [sagorbrur/bangla-bert](https://github.com/sagorbrur/bangla-bert) (MIT) | Bangla BERT + published sentiment-analysis numbers |

## Inference models (Hugging Face; not retrained here)

- Multilingual / EN-leaning social: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (same ID as `config/brands.yaml`)
- Bangla sentiment (SentiGOLD fine-tune): `ahs95/banglabert-sentiment-analysis` (5-class → mapped to 3-class)
- VADER: loaded from the local clone under `sentiment/vaderSentiment`

## Rebuild clones

```powershell
cd third_party\sentiment
git clone --depth 1 https://github.com/cjhutto/vaderSentiment.git
git clone --depth 1 https://github.com/cardiffnlp/xlm-t.git
git clone --depth 1 https://github.com/csebuetnlp/banglabert.git
git clone --depth 1 https://github.com/sagorbrur/bangla-bert.git
```

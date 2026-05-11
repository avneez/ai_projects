# Ticker NER (Named Entity Recognition)

## Purpose
Extract stock ticker symbols from unstructured text (tweets, news, Telegram messages).

## Libraries Used
- **spaCy 3.x** - Industrial-strength NLP library
- **Custom trained NER model** - Fine-tuned on financial texts
- **Regular expressions** - Pattern matching for cashtags ($AAPL)

## Architecture & Logic

### 1. Pattern-Based Extraction (Fast Path)
```
Input: "$AAPL is up 5% today. Check $TSLA too!"
Regex: \$[A-Z]{1,5}\b
Output: ["AAPL", "TSLA"]
```

**Why regex first?**
- 90% of tickers in social media use cashtag format ($TICKER)
- Instant extraction (<1ms)
- No ML overhead

### 2. spaCy NER Model (Fallback)
```
Input: "Apple Inc reported strong earnings"
spaCy NER: [("Apple Inc", ORG)] → Resolve to AAPL
Output: ["AAPL"]
```

**Training Process:**
- Base model: `en_core_web_trf` (transformer-based)
- Fine-tuned on 10k+ labeled financial texts
- Custom entity type: `STOCK_TICKER`
- Training data sources: SEC filings, financial news, earnings calls

### 3. Ticker Validation
```
Extracted: ["AAPL", "CEO", "USA"]
Validate against NYSE/NASDAQ symbol list
Valid: ["AAPL"]
Rejected: ["CEO", "USA"] (false positives)
```

**Symbol List Sources:**
- NASDAQ FTP server (daily updates)
- NYSE API
- Cached in Redis (TTL: 24 hours)

### 4. Context-Based Disambiguation
```
Text: "Apple announced new products"
Context words: ["announced", "products"] → Company context
Ticker: AAPL ✓

Text: "I ate an apple today"
Context words: ["ate"] → Food context
Ticker: None ✗
```

## Data Flow
```
Raw Text
  ↓
Pattern Matching (regex) → Found? → Return tickers
  ↓ Not found
spaCy NER Extraction
  ↓
Entity Resolution (company name → ticker)
  ↓
Validation (check against symbol list)
  ↓
Context Filtering
  ↓
Final ticker list
```

## Key Implementation Points

### Performance Optimization
- **Regex first**: 90% cases handled in <1ms
- **Batch processing**: Process 32 texts at once through spaCy
- **GPU acceleration**: For transformer model inference
- **Caching**: Common company names cached (AAPL = "Apple Inc", etc.)

### Handling Edge Cases
1. **Multiple tickers in one text**: Extract all, remove duplicates
2. **Misspellings**: Fuzzy matching with `fuzzywuzzy` (disabled for performance)
3. **Non-US tickers**: Filter by market (NYSE, NASDAQ only)
4. **Delisted stocks**: Daily symbol list updates

## Model Training Details

### Dataset Creation
```
- 10,000 manually labeled texts
- Sources: Twitter, financial news, SEC filings
- Annotations: [text, [(start, end, "STOCK_TICKER")]]
- Tools: Label Studio for annotation
```

### Training Configuration
```python
# spaCy config (simplified)
[training]
train_corpus = financial_news
dev_corpus = validation_set
accumulate_gradient = 3
optimizer = Adam
learning_rate = 0.001
```

### Model Evaluation
- **Precision**: 94% (few false positives)
- **Recall**: 87% (some tickers missed)
- **F1 Score**: 90%
- **Inference speed**: 15ms per text (GPU), 50ms (CPU)

## Integration Points
- **Input**: Kafka topic `validated_news`
- **Output**: Enriched messages with `tickers_extracted` field
- **Fallback**: If NER fails, text still processed (empty ticker list)

## Interview Q&A

**Q: Why spaCy over Hugging Face transformers?**
A: spaCy is production-optimized (faster, easier deployment), has built-in NER pipeline, and supports custom entity types. Transformers are research-focused.

**Q: How do you handle "AAPL" vs "Apple Inc"?**
A: Maintain a mapping dict: `{"Apple Inc": "AAPL", "Tesla Motors": "TSLA"}`. Updated from SEC CIK database.

**Q: What if a company has multiple tickers (e.g., GOOGL vs GOOG)?**
A: Return both, let downstream services decide. Usually keep primary ticker (higher volume).

**Q: How do you keep the symbol list updated?**
A: Daily cron job fetches from NASDAQ FTP, updates Redis cache. Alert if fetch fails.

**Q: Accuracy vs Speed trade-off?**
A: Regex (fast, 85% accuracy) → spaCy (slower, 95% accuracy). Two-tier approach optimizes both.

**Q: How to handle non-English text?**
A: Language detection (`langdetect`) → filter to English only. For multilingual: use `xx_ent_wiki_sm` (multilingual model).

**Q: What's the biggest challenge?**
A: Context disambiguation ("Apple the fruit" vs "Apple the company"). Solved with context window analysis and word embeddings.

## Cost Considerations
- **spaCy model size**: 500MB (transformer-based)
- **GPU memory**: 2GB for batch inference
- **Alternative**: `en_core_web_sm` (15MB, CPU-friendly, 85% accuracy)

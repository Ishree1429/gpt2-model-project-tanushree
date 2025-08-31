# GPT-Style Lyrics Generator 

This project builds a small GPT Transformer language model  to generate new song lyrics. The focus is on a clean learning roadmap, reproducibility, and clarity.

---

##  Project Roadmap

1. GitHub repo setup, Python virtual environment, Project scaffold (folders, ‘.gitignore’, ‘requirements.txt’)

2. Data preparation & tokenization, Character tokenizer, Simple BPE tokenizer (didactic), Train a trivial “bigram baseline” for sanity check

3. 

---

## Data 

- Source -: Kaggle - Multilingual Lyrics for Genre Classification
- Link -: https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification
- Focus Genres -: ‘pop’, ‘r&b’, ‘indie’, ‘jazz’, ‘hip hop’, ‘rock’  
- Files in repo -:  
  - ‘cleaned_Set/train_clean.csv’  
  - ‘cleaned_Set/test_clean.csv’  
- Note -: Large raw Kaggle CSVs are ‘not committed’ (gitignored). Only the small cleaned versions are saved in the repo.

---

##  Data Preparation & Cleaning

1. Kept train/test separate –: Did not merge `train.csv` and `test.csv`.  

2. Normalized columns & labels -: 
- All column names lowercased.  
   - Standardized genre strings -:  
     - ‘hip-hop → hip hop’
     - ‘rnb → r&b’

3. Filtered target genres -: Retained rows where ‘genre ∈ {pop, r&b, indie, jazz, hip hop, rock}’

4. Dropped invalid rows -:
   - Removed empty lyrics rows.  
   - Removed duplicate lyrics.

5. Applied cleaning helpers -: 
   - Remove HTML tags & URLs  
   - Normalize unicode accents (e.g. “Beyoncé → Beyonce”)  
   - Remove stage markers ‘[Verse 1]’, ‘[Chorus]’  
   - Collapse repeated whitespace  

6. Test-only column drops
   - Removed ‘Song year’, ‘Track_id’ from test.csv only.  
   - Train kept them if present.

---

##  Dataset Snapshot 

- Rows (train): N = 215496
- Rows (test): N = 5160
- Unique artists (train): 8439
- Unique artists ( test):  2028
- Avg. characters per lyric (train): 1195.40
- Avg. characters per lyric (test): 1338.75

- Class counts (train):  
     hip hop: 2080 
     Indie: 7493 
     jazz: 7636 
     pop: 89603 
     r&b: 2531 
     rock: 106153

-Class counts (test):
     hip hop: 960
     indie: 510 
     jazz: 660 
     pop: 1110 
     r&b: 510 
     rock: 1410 

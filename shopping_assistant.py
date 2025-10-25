import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # fallback handled later
import re
import sys
import os
from typing import Dict, List, Tuple, Optional

try:
    import openai
except ImportError:
    openai = None

# Make stdout/stderr robust to Unicode on Windows consoles
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

class IntentClassifier:
    """Multi-intent classifier for search, recommend, and compare intents"""

    def __init__(self, confidence_threshold: float = 0.65):
        self.embedder = None
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer(os.getenv('INTENT_EMBEDDER', 'all-MiniLM-L6-v2'))
            except Exception:
                self.embedder = None

        self.vectorizer = None if self.embedder is not None else TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.label_encoder = LabelEncoder()
        self.confidence_threshold = confidence_threshold
        self.trained = False
        self.train()

    def train(self):
        examples: List[Tuple[str, str]] = []

        search_examples = [
            "find a phone", "search for smartphones", "show me phones", "list all phones",
            "what phones are available", "looking for a device", "find me a mobile",
            "phones under £500", "search phones under 800 pounds", "phones below 600",
            "phones less than 1000", "between £600 and £1000 phones", "around £700 phones",
            "show all apple phones", "find samsung devices", "google pixel list", "xiaomi phones",
            "find 8gb ram phones", "phones with 5000mah battery", "6.7 inch screen phones",
            "5g phones", "foldable phones", "compact phones", "lightweight phone",
            "latest phones", "new phones", "cheap phones", "budget phones", "premium phones",
            "android phones", "ios phones", "camera phones", "gaming phones",
            "phones under £200", "search phones under £400", "find phones below £750", "phones less than £1200", 
            "phones between £300 and £600", "phones around £900", "show phones under £250", "list phones under £1000",
            "find budget samsung phones", "show all motorola devices", "list google pixel models", "find oneplus phones",
            "phones with 4 star reviews", "search high rated phones", "phones with 4.5+ reviews", "best rated xiaomi phones",
            "find iPhone 14", "search Galaxy S23", "show Pixel 7", "list Xperia 1 IV", "find OnePlus 11",
            "phones under £500 with good reviews", "samsung phones over £600", "apple phones under £800", 
            "google phones with high ratings", "xiaomi phones below £400", "motorola phones above 4 stars",
            "find phones with great cameras", "search budget android phones", "list premium ios phones", 
            "phones around £500", "find lightweight samsung phones", "show compact google phones", 
            "list foldable samsung devices", "search 5g apple phones", "find high review motorola", 
            "phones between £200 and £400", "search xiaomi under £300", "show oneplus over 4.2 reviews",
            "list phones with strong battery", "find cheap 5g phones", "search premium google phones", 
            "show all budget phones", "find phones under £700 with 4+ stars", "list samsung flagships", 
            "search apple phones under £900", "find motorola budget models", "show xiaomi with good reviews", 
            "list phones under £350", "search high rated oneplus", "find phones around £450", 
            "show all 5g budget phones", "list compact ios devices",
        ]
        recommend_examples = [
            "recommend a phone", "suggest a good smartphone", "what should I buy",
            "which phone is best", "help me choose a phone", "what do you recommend",
            "best phone for me", "help me pick a phone", "which one should I get",
            "best phone for gaming", "recommend gaming phone", "good camera phone",
            "best phone for photography", "recommend phone for students", "best value phone",
            "recommend a flagship", "which phone has best battery", "recommend durable phone",
            "best phone for work", "recommend a premium phone", "what's the best deal phone",
            "recommend a lightweight phone", "recommend small phone", "recommend big screen phone",
            "recommend fast phone", "recommend long battery life phone",
            "recommend a phone under £300", "suggest a budget smartphone", "best phone under £700", 
            "recommend a phone for photography", "suggest a durable samsung", "best phone for video calls", 
            "recommend a high rated google phone", "suggest an apple phone under £800", "best xiaomi for value", 
            "recommend a motorola phone", "suggest a oneplus flagship", "best phone for multitasking", 
            "recommend a phone with great reviews", "suggest a phone under £500", "best phone for seniors", 
            "recommend a compact phone", "suggest a 5g budget phone", "best phone under £400", 
            "recommend a phone for travel", "suggest a high review samsung", "best apple for students", 
            "recommend a phone around £600", "suggest a lightweight oneplus", "best phone for social media", 
            "recommend a premium xiaomi", "suggest a phone with 4+ stars", "best motorola for work", 
            "recommend a phone under £250", "suggest a google phone for gaming", "best phone for long battery", 
            "recommend a samsung under £700", "suggest an apple flagship", "best phone for budget buyers", 
            "recommend a phone with high ratings", "suggest a xiaomi under £400", "best oneplus for photography", 
            "recommend a compact samsung", "suggest a phone around £800", "best phone for productivity", 
            "recommend a motorola under £500", "suggest a high rated google pixel", "best phone for music", 
            "recommend a budget ios phone", "suggest a phone with strong reviews", "best xiaomi for gaming", 
            "recommend a phone under £900", "suggest a oneplus with 4+ stars", "best phone for video streaming", 
            "recommend a samsung for students", "suggest a lightweight apple phone",
        ]
        compare_examples = [
            "compare phones", "which is better", "difference between phones",
            "compare iphone and samsung", "what's better", "compare these devices",
            "iphone vs android", "compare prices", "which one is superior",
            "compare pixel and galaxy", "difference between models", "compare specs",
            "which has better camera", "compare battery life", "price comparison",
            "iphone 17 vs galaxy s25", "compare flagship phones", "oneplus vs xiaomi",
            "what's the difference", "compare budget phones", "which is cheaper",
            "compare performance", "which lasts longer", "compare features",
            "iPhone 15 vs Google Pixel 8", "Compare iPhone 16 and Samsung Galaxy S24",
            "Galaxy S24 vs Galaxy S23", "Pixel 8 versus iPhone 15",
            "compare samsung and oneplus", "difference between google and xiaomi", "compare motorola and apple", 
            "which is better budget or flagship", "compare prices of ios phones", "samsung vs google reviews", 
            "compare iPhone 15 and Pixel 8 Pro", "difference between Galaxy S23 and S24", "compare xiaomi and oneplus", 
            "which has better reviews", "compare budget samsung and motorola", "iPhone 14 vs Galaxy S23", 
            "compare oneplus 11 and xiaomi 13", "difference between pixel 7 and pixel 8", "compare apple and samsung prices", 
            "which phone is cheaper under £500", "compare motorola and xiaomi reviews", "iPhone 15 vs OnePlus 12", 
            "compare samsung flagships", "difference between budget and premium phones", "compare google and apple cameras", 
            "which has better battery samsung or xiaomi", "compare oneplus and motorola prices", "Pixel 7a vs Galaxy A54", 
            "compare reviews of ios and android", "difference between iPhone 14 and 15", "compare budget google phones", 
            "which is better for gaming samsung or oneplus", "compare xiaomi 13 and pixel 8", "iPhone 16 vs Galaxy S24", 
            "compare motorola and samsung cameras", "difference between oneplus 10 and 11", "compare prices under £600", 
            "which has higher reviews apple or google", "compare samsung and xiaomi flagships", "Pixel 8 vs iPhone 14 Pro", 
            "compare budget oneplus and motorola", "difference between galaxy s23 and pixel 7", "compare apple and oneplus reviews", 
            "which is cheaper xiaomi or motorola", "compare samsung and google battery life", "iPhone 15 Pro vs Galaxy S24 Ultra", 
            "compare high rated budget phones", "difference between xiaomi 12 and 13", "compare oneplus and pixel prices", 
            "which has better camera apple or samsung", "compare motorola and google flagships", "difference between pixel 7a and 8", 
            "compare samsung and xiaomi under £500", "which is better for reviews oneplus or xiaomi",
        ]
        general_examples = [
            "hello", "hi there", "how are you", "thanks", "goodbye", "Okay", "okay", "ok", "k", "Okay thanks",
            "what can you do", "tell me a joke", "what's the weather",
            "who are you", "help me", "hey", "yo", "hola", "morning",
            "can you assist", "what is this", "explain your features",
            "hey there", "good evening", "what's up", "how's it going", "see you later", 
            "thank you", "alright", "hi", "bonjour", "evening greeting", 
            "what else can you do", "tell me something funny", "current weather", 
            "who made you", "need help", "yo what's good", "hiya", "good day", 
            "can you help me out", "what's this about", "explain what you can do", 
            "morning greeting", "adios", "later", "howdy", "what's new", 
            "tell me about yourself", "give me a fun fact", "what's the time", 
            "how can you assist me", "say something cool", "what's your purpose", 
            "hello again", "ciao", "good vibes", "what do you know", "spill the beans", 
            "tell me a story", "what's the deal", "how's your day", "greetings", 
            "can you make me laugh", "what's the forecast", "who's behind you", 
            "help me out here", "sup dude", "hola amigo", "good night", 
            "what's your story", "share a joke",
        ]

        examples += [(t, 'search') for t in search_examples]
        examples += [(t, 'recommend') for t in recommend_examples]
        examples += [(t, 'compare') for t in compare_examples]
        examples += [(t, 'general') for t in general_examples]

        texts = [t for t, _ in examples]
        labels = [l for _, l in examples]

        self.train_texts_ = texts
        self.train_labels_ = labels

        if self.embedder is not None:
            X = self.embedder.encode(texts, normalize_embeddings=True)
        else:
            X = self.vectorizer.fit_transform(texts)

        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        self.trained = True

    def rule_predict(self, q: str) -> Optional[Tuple[str, float]]:
        ql = q.lower()
        qn = ql.replace('£', '£').replace('€', '£').replace('eur', '£')
        if re.search(r"\b(compare|vs\.?|versus|difference between|which is better|what's better)\b", qn):
            return 'compare', 0.99
        if re.search(r"\b(recommend|suggest|best|which (phone|one) should i buy|help me choose|what should i buy|which is best)\b", qn):
            return 'recommend', 0.95
        price_pat = r"(under|below|less than|over|above|between|around)\s*\£?\s*\d+"
        has_price = re.search(price_pat, qn) or re.search(r"\£\s*\d+", qn)
        has_search_verb = re.search(r"\b(find|search|show|list|available|look|get)\b", qn)
        has_domain_terms = re.search(r"\b(phone|smartphone|mobile|device|gaming|camera|battery|ram|ios|android|foldable|compact|premium|budget)\b", qn)
        if has_price or has_search_verb or has_domain_terms:
            return 'search', 0.9 if (has_price or has_search_verb) else 0.8
        if re.search(r"\b(what about|how about)\b", qn):
            return 'search', 0.8
        return None

    def predict(self, query: str) -> Tuple[str, float]:
        if not self.trained:
            return 'unknown', 0.0
        rule_hit = self.rule_predict(query)
        if rule_hit:
            return rule_hit
        if self.embedder is not None:
            X = self.embedder.encode([query.lower()], normalize_embeddings=True)
        else:
            X = self.vectorizer.transform([query.lower()])
        probabilities = self.classifier.predict_proba(X)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        intent = self.label_encoder.inverse_transform([predicted_class])[0]
        return intent, confidence

    def evaluate_kfold(self, k: int = 5) -> float:
        """Return mean k-fold accuracy (%) using current encoder/classifier set.
        Trains a fresh LR per fold for a fair estimate.
        """
        texts = getattr(self, 'train_texts_', None)
        labels = getattr(self, 'train_labels_', None)
        if not texts or not labels:
            return 0.0
        skf = StratifiedKFold(n_splits=max(2, min(k, len(set(labels)))), shuffle=True, random_state=42)
        y = np.array(labels)
        accs = []
        for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
            X_train_texts = [texts[i] for i in train_idx]
            X_val_texts = [texts[i] for i in val_idx]
            y_train = LabelEncoder().fit_transform([labels[i] for i in train_idx])
            le = LabelEncoder()
            y_all = le.fit_transform(labels)
            y_val = y_all[val_idx]

            if self.embedder is not None:
                X_train = self.embedder.encode(X_train_texts, normalize_embeddings=True)
                X_val = self.embedder.encode(X_val_texts, normalize_embeddings=True)
            else:
                vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
                X_train = vec.fit_transform(X_train_texts)
                X_val = vec.transform(X_val_texts)

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            acc = (preds == y_val).mean()
            accs.append(acc)
        return float(np.mean(accs) * 100.0)


class SentimentAnalyzer:
    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline 
            model_id = os.getenv('SENTIMENT_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english')
            self.pipe = pipeline('sentiment-analysis', model=model_id)
        except Exception:
            self.pipe = None

        self.positive_words = {
            'best', 'good', 'great', 'excellent', 'amazing', 'love', 'perfect',
            'awesome', 'fantastic', 'wonderful', 'brilliant', 'outstanding',
            'superb', 'quality', 'premium', 'top', 'high-end', 'flagship'
        }
        self.negative_words = {
            'bad', 'worst', 'poor', 'terrible', 'awful', 'hate', 'disappointing',
            'cheap', 'low-quality', 'inferior', 'weak', 'slow', 'expensive',
            'overpriced', 'useless', 'problem', 'issue', 'buggy'
        }
        self.budget_words = {
            'cheap', 'affordable', 'budget', 'inexpensive', 'economical',
            'low-cost', 'under', 'less than', 'below'
        }

    def analyze(self, text: str) -> Dict[str, float]:
        ql = text.lower()
        bud = any(w in ql for w in ['cheap', 'affordable', 'budget', 'under', 'below', 'less than'])
        if self.pipe is not None:
            try:
                res = self.pipe(ql[:512])[0]
                label = (res.get('label') or '').upper()
                score = float(res.get('score') or 0.5)
                if 'NEG' in label:
                    sentiment = 'negative'; norm_score = 0.5 - (score / 2)
                elif 'POS' in label:
                    sentiment = 'positive'; norm_score = 0.5 + (score / 2)
                else:
                    sentiment = 'neutral'; norm_score = 0.5
                return {'sentiment': sentiment, 'score': norm_score, 'is_budget_query': bud}
            except Exception:
                pass

        # Fallback lexicon method
        words = set(ql.split())
        pos = len(words & self.positive_words)
        neg = len(words & self.negative_words)
        total = max(pos + neg, 1)
        if pos == 0 and neg == 0:
            sentiment = 'neutral'; score = 0.5
        elif pos > neg:
            sentiment = 'positive'; score = 0.5 + (pos / (total * 2))
        elif neg > pos:
            sentiment = 'negative'; score = 0.5 - (neg / (total * 2))
        else:
            sentiment = 'neutral'; score = 0.5
        return {'sentiment': sentiment, 'score': score, 'is_budget_query': bud}


class PhoneDatabase:
    def __init__(self, csv_path: str = 'mobile_phones.csv'):
        self.df = pd.read_csv(csv_path)
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df['review'] = pd.to_numeric(self.df['review'], errors='coerce')
        self.df['search_text'] = (
            self.df['name'].fillna('') + ' ' + self.df['brand'].fillna('')
        ).str.lower()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        ql = query.lower()
        if any(w in ql for w in ['gaming', 'gamer', 'game']):
            mask = self.df['name'].str.contains(r"rog|legion|redmagic|black\s*shark|gaming", case=False, regex=True)
            gaming_df = self.df[mask].copy()
            if not gaming_df.empty:
                return gaming_df.nlargest(top_k, 'review')
        query_vec = self.vectorizer.transform([ql])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        if len(similarities) == 0 or np.max(similarities) <= 1e-12:
            return self.df.iloc[0:0].copy()
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]
        return results

    def filter_by_price_max(self, df: pd.DataFrame, max_price: float) -> pd.DataFrame:
        return df[df['price'] <= max_price].copy()

    def filter_by_price_min(self, df: pd.DataFrame, min_price: float) -> pd.DataFrame:
        return df[df['price'] >= min_price].copy()

    def get_top_rated(self, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        return df.nlargest(top_k, 'review')

class DialogueManager:
    def __init__(self):
        self.context = {
            'last_intent': None,
            'last_query': None,
            'last_results': None,     # DataFrame of last results
            'last_brand': None,
            'last_min_price': None,
            'last_max_price': None,
            'last_page_offset': 0,    # pagination offset for “show me more”
            'page_size': 10,
            'conversation_history': [],
            'messages': []
        }

    def update_context(self, intent: str, query: str, results: Optional[pd.DataFrame] = None,
                       min_price: Optional[float] = None, max_price: Optional[float] = None,
                       brand_override: Optional[List[str]] = None, reset_pagination: bool = True):
        self.context['last_intent'] = intent
        self.context['last_query'] = query
        self.context['last_results'] = results
        if reset_pagination:
            self.context['last_page_offset'] = 0
        if min_price is not None or max_price is not None:
            self.context['last_min_price'] = min_price
            self.context['last_max_price'] = max_price
        if results is not None and len(results) > 0 and 'brand' in results.columns:
            brands = results['brand'].dropna().unique()
            if len(brands) == 1:
                self.context['last_brand'] = brands[0]
        if brand_override:
            self.context['last_brand'] = brand_override[0] if len(brand_override) == 1 else None
        self.context['conversation_history'].append({'query': query, 'intent': intent})

    def add_user_message(self, text: str):
        msgs = self.context.setdefault('messages', [])
        msgs.append({'role': 'user', 'content': text})
        # Keep full conversation (no truncation) so GPT can use full context

    def add_assistant_message(self, text: str):
        msgs = self.context.setdefault('messages', [])
        msgs.append({'role': 'assistant', 'content': text})
        # Keep full conversation (no truncation) so GPT can use full context

    def get_paged_slice(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
        """Return a page slice and update offset."""
        if df is None or df.empty:
            return df, 0, 0
        ps = self.context['page_size']
        start = self.context['last_page_offset']
        end = min(start + ps, len(df))
        page = df.iloc[start:end]
        self.context['last_page_offset'] = end
        return page, start + 1, end

    def is_follow_up(self, query: str) -> bool:
        indicators = [
            'what about', 'how about', 'show me more', 'any other', 'else',
            'similar', 'like that', 'those', 'them', 'it', 'that one',
            'this one', 'the same', 'also', 'too', 'as well', 'more results', 'next', 'cheaper'
        ]
        ql = query.lower()
        return any(ind in ql for ind in indicators)

class ShoppingCopilot:
    def __init__(self, csv_path: str = 'mobile_phones.csv', use_openai: bool = True):
        self.intent_classifier = IntentClassifier(confidence_threshold=0.65)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.db = PhoneDatabase(csv_path)
        self.dialogue_manager = DialogueManager()

        self.brand_aliases = {
            'iphone': 'Apple', 'ios': 'Apple', 'apple': 'Apple',
            'galaxy': 'Samsung', 'samsung': 'Samsung',
            'pixel': 'Google', 'google': 'Google',
            'oneplus': 'OnePlus',
            'xiaomi': 'Xiaomi', 'mi ': 'Xiaomi', ' redmi': 'Xiaomi', 'redmi': 'Xiaomi',
            'motorola': 'Motorola', 'moto': 'Motorola', 'razr': 'Motorola',
            'sony': 'Sony', 'xperia': 'Sony',
            'asus': 'Asus', 'rog': 'Asus',
            'nothing': 'Nothing',
            'nokia': 'Nokia',
            'honor': 'Honor',
            'oppo': 'Oppo',
            'vivo': 'Vivo',
            'realme': 'Realme',
            'black shark': 'Black Shark', 'blackshark': 'Black Shark',
            'lenovo': 'Lenovo', 'legion': 'Lenovo',
            'huawei': 'Huawei',
        }

        self.use_openai = use_openai and (openai is not None) and bool(os.getenv('OPENAI_API_KEY'))
        self.openai_client = None
        if self.use_openai:
            try:
                from openai import OpenAI  # type: ignore
                self.openai_client = OpenAI()
            except Exception:
                try:
                    openai.api_key = os.getenv('OPENAI_API_KEY')
                    self.openai_client = openai  # type: ignore
                except Exception:
                    self.use_openai = False

    def reset_context(self):
        """Clear dialogue state to avoid leaking prior session context."""
        self.dialogue_manager = DialogueManager()
        
    def extract_price_limits(self, query: str) -> Tuple[Optional[float], Optional[float]]:
        q = query.lower().replace(',', '')
        m = re.search(r'between\s+£?\s*(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*£?\s*(\d+(?:\.\d+)?)', q)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return (min(lo, hi), max(lo, hi))
        m = re.search(r'(?:around|about|~)\s*£?\s*(\d+(?:\.\d+)?)', q)
        if m:
            x = float(m.group(1)); pad = x * 0.15
            return (x - pad, x + pad)
        m = re.search(r'(?:over|above|more than|min(?:imum)?)\s+£?\s*(\d+(?:\.\d+)?)', q)
        if m:
            return (float(m.group(1)), None)
        m = re.search(r'(?:under|below|less than|cheaper than|max(?:imum)?)\s+£?\s*(\d+(?:\.\d+)?)', q)
        if m:
            return (None, float(m.group(1)))
        m = re.search(r'£\s*(\d+(?:\.\d+)?)', query)
        if m and re.search(r'(under|below|less than|budget|max)', q):
            val = float(m.group(1)); return (None, val)
        return (None, None)

    def aliases_to_brands(self, query: str) -> List[str]:
        """Return brands in the order they appear in the query (not as a set). Skips alias+number (model mentions)."""
        q = query.lower()
        hits = {}

        def add_hit(brand: str, pos: int):
            if brand not in hits or pos < hits[brand]:
                hits[brand] = pos

        # alias matches
        for alias, brand in self.brand_aliases.items():
            for m in re.finditer(rf'\b{re.escape(alias)}\b', q):
                # If alias followed by a number soon after (e.g., "iphone 17"), it's model-ish → skip adding brand.
                tail = q[m.end(): m.end()+12]
                if re.search(r'\d', tail):
                    continue
                add_hit(brand, m.start())

        # exact dataset brand matches
        for b in self.db.df['brand'].dropna().unique():
            for m in re.finditer(rf'\b{re.escape(b.lower())}\b', q):
                add_hit(b, m.start())

        return [b for b, _ in sorted(hits.items(), key=lambda kv: kv[1])]

    def extract_model_mentions(self, query: str) -> List[pd.Series]:
        q = query.lower()
        parts = re.split(r'\b(?:vs\.?|versus|or|and|,|/)\b', q)
        parts = [p.strip() for p in parts if p.strip()]
        brand_tokens = [
            'iphone','galaxy','pixel','oneplus','xiaomi','redmi','mi','moto','motorola',
            'xperia','sony','rog','asus','nothing','nokia','honor','oppo','vivo','huawei','realme','lenovo','razr'
        ]
        # Common non-numeric model keywords across brands
        model_keywords = r"\b(pro|max|plus|ultra|mini|air|se|fold|flip|note|edge|nord|mate|magic|razr|fe|t|a)\b"
        modelish = []
        for p in parts:
            has_brand = any(bt in p for bt in brand_tokens)
            has_digit = bool(re.search(r'\d', p))
            has_kw = bool(re.search(model_keywords, p))
            two_words = len(p.split()) >= 2
            # Accept if brand is present AND (digits or known model keyword or at least two tokens)
            if has_brand and (has_digit or has_kw or two_words):
                modelish.append(p)
        resolved = []
        seen = set()
        for m in modelish:
            res = self.db.search(m, top_k=1)
            if not res.empty:
                row = res.iloc[0]
                uid = f"{row.get('name','')}|{row.get('brand','')}"
                if uid not in seen:
                    seen.add(uid)
                    resolved.append(row)
        return resolved

    # ---------- Formatting ----------
    def format_phone_list(self, df: pd.DataFrame, max_items: int = 10,
                           range_note: Optional[str] = None,
                           page_info: Optional[Tuple[int, int, int]] = None) -> str:
        if df is None or df.empty:
            return "No phones found matching your criteria."
        df_display = df.head(max_items)
        lines = []
        for _, row in df_display.iterrows():
            lines.append(f"- {row['name']} ({row['brand']}) - £{row['price']:.0f} | Rating: {row['review']:.1f}/5.0")
        out = "\n".join(lines)
        if page_info:
            start, end, total = page_info
            out += f"\n\n(Showing {start}–{end} of {total})"
        if range_note:
            out += f"\n\n{range_note}"
        return out

    # ---------- Handlers ----------
    def handle_search(self, query: str, sentiment: Dict,
                       min_price: Optional[float] = None, max_price: Optional[float] = None,
                       brand_override: Optional[List[str]] = None, followup: bool = False) -> Dict:
        if min_price is None and max_price is None:
            min_price, max_price = self.extract_price_limits(query)
        brands = brand_override if brand_override is not None else self.aliases_to_brands(query)
        is_budget = sentiment['is_budget_query']

        # Decide base results: use full catalog for brand-only or price filters; semantic search otherwise
        ql = query.lower()
        filler_words = {
            'phone', 'phones', 'device', 'devices', 'smartphone', 'smartphones',
            'show', 'me', 'all', 'list', 'find', 'search', 'for', 'a', 'an', 'the',
            'under', 'below', 'less', 'than', 'over', 'above', 'between', 'and'
        }
        brand_tokens_set = set(self.brand_aliases.keys()) | set(b.lower() for b in self.db.df['brand'].dropna().unique())
        tokens = re.findall(r"[a-z0-9+]+", ql)
        non_brand_terms = [t for t in tokens if t not in brand_tokens_set and t not in filler_words]
        is_brand_only_query = bool(brands) and (len(non_brand_terms) == 0) and (min_price is None and max_price is None)

        if (min_price is not None) or (max_price is not None) or is_brand_only_query:
            results = self.db.df.copy()
        else:
            results = self.db.search(query, top_k=200)

        # Implicit budget
        if min_price is None and max_price is None and is_budget:
            max_price = 500

        if min_price is not None:
            results = self.db.filter_by_price_min(results, min_price)
        if max_price is not None:
            results = self.db.filter_by_price_max(results, max_price)
        if brands:
            results = results[results['brand'].isin(brands)].copy()

        # If query has specific non-brand terms (e.g., 'air' in 'iphone air'),
        # narrow to names containing all those terms when possible.
        if results is not None and not results.empty and non_brand_terms and not is_brand_only_query:
            mask = pd.Series(True, index=results.index)
            for t in non_brand_terms:
                mask &= results['name'].str.contains(re.escape(t), case=False, na=False)
            narrowed = results[mask]
            if not narrowed.empty:
                results = narrowed

        if not results.empty:
            # Keep TF-IDF search ordering when available; otherwise sort by rating then price
            if 'similarity' in results.columns:
                ql = query.lower()
                has_brand_token = any(tok in ql for tok in self.brand_aliases.keys())
                has_digits = bool(re.search(r'\d', ql))
                # For model-like queries (brand + digits), prune weak matches
                if has_brand_token and has_digits:
                    max_sim = float(results['similarity'].max()) if not results['similarity'].empty else 0.0
                    cutoff = max(0.15, max_sim * 0.5)
                    results = results[results['similarity'] >= cutoff].copy()
                results = results.sort_values(['similarity', 'review', 'price'], ascending=[False, False, True])
            else:
                results = results.sort_values(['review', 'price'], ascending=[False, True])

        # Follow-up pagination ("show me more")
        if followup and ('show me more' in query.lower() or 'more results' in query.lower() or 'next' in query.lower()):
            last = self.dialogue_manager.context.get('last_results')
            if last is not None and not last.empty:
                page, start, end = self.dialogue_manager.get_paged_slice(last)
                response = "Here are more results:\n\n" + self.format_phone_list(
                    page, max_items=len(page), page_info=(start, end, len(last))
                )
                return {'response': response, 'intent': 'search', 'confidence': 1.0, 'results_count': len(page)}

        # Save & page first slice
        self.dialogue_manager.update_context(
            'search', query, results if not results.empty else None,
            min_price=min_price, max_price=max_price,
            brand_override=brands if brands else None,
            reset_pagination=True
        )

        if results.empty:
            if brands and (max_price is not None):
                # Graceful fallback: show the cheapest few from those brands
                brand_df = self.db.df[self.db.df['brand'].isin(brands)].copy()
                if not brand_df.empty:
                    cheapest = brand_df.nsmallest(3, 'price')
                    response = ("I couldn't find matches under your cap; here are the cheapest options from "
                                f"{', '.join(brands)} instead:\n\n" + self.format_phone_list(cheapest, max_items=3))
                    return {'response': response, 'intent': 'search', 'confidence': 1.0, 'results_count': len(cheapest)}
            if brands:
                response = f"I couldn't find any {', '.join(brands)} phones in the catalogue. Try adjusting your query."
            else:
                response = "I couldn't find any phones matching your search criteria. Try adjusting your requirements."
            count = 0
        else:
            page, start, end = self.dialogue_manager.get_paged_slice(results)
            count = len(results)
            note = None
            if min_price is not None and max_price is not None:
                note = f"(Filtered to phones between £{min_price:.0f} and £{max_price:.0f})"
            elif max_price is not None:
                note = f"(Filtered to phones under £{max_price:.0f})"
            elif min_price is not None:
                note = f"(Filtered to phones over £{min_price:.0f})"
            response = f"I found {count} phone{'s' if count != 1 else ''} for you:\n\n"
            response += self.format_phone_list(page, max_items=len(page), range_note=note,
                                                page_info=(start, end, count))

        return {'response': response, 'intent': 'search', 'confidence': 1.0, 'results_count': count}

    def handle_recommend(self, query: str, sentiment: Dict,
                          min_price: Optional[float] = None, max_price: Optional[float] = None,
                          brand_override: Optional[List[str]] = None, followup: bool = False) -> Dict:
        if min_price is None and max_price is None:
            min_price, max_price = self.extract_price_limits(query)
        brands = brand_override if brand_override is not None else self.aliases_to_brands(query)
        is_budget = sentiment['is_budget_query']

        candidates = self.db.df.copy()
        ql = query.lower()
        if min_price is not None:
            candidates = self.db.filter_by_price_min(candidates, min_price)
        if max_price is not None:
            candidates = self.db.filter_by_price_max(candidates, max_price)
        if min_price is None and max_price is None and is_budget:
            candidates = self.db.filter_by_price_max(candidates, 500)
        if brands:
            candidates = candidates[candidates['brand'].isin(brands)].copy()
        # Gaming-specific recommendations when the user asks for gaming phones
        if any(w in ql for w in ['gaming', 'gamer', 'rog', 'legion', 'redmagic', 'black shark', 'blackshark']):
            mask = candidates['name'].str.contains(r"rog|legion|redmagic|black\s*shark|gaming", case=False, regex=True)
            gaming_candidates = candidates[mask].copy()
            if not gaming_candidates.empty:
                candidates = gaming_candidates

        recommendations = self.db.get_top_rated(candidates, top_k=5)

        self.dialogue_manager.update_context(
            'recommend', query, recommendations if not recommendations.empty else None,
            min_price=min_price, max_price=max_price,
            brand_override=brands if brands else None,
            reset_pagination=True
        )

        if recommendations.empty:
            response = "I couldn't find suitable recommendations. Could you provide more details about your preferences?"
        else:
            response = "Based on your requirements, I recommend these excellent phones:\n\n"
            response += self.format_phone_list(recommendations, max_items=5)
            if is_budget and min_price is None and max_price is None:
                response += "\n\n💡 These are great budget-friendly options with excellent value!"
            elif max_price is not None and min_price is None:
                response += f"\n\n💡 All recommendations are within your £{max_price:.0f} budget."
            elif min_price is not None and max_price is None:
                response += f"\n\n💡 These are top options above £{min_price:.0f}."
            elif min_price is not None and max_price is not None:
                response += f"\n\n💡 All recommendations are between £{min_price:.0f} and £{max_price:.0f}."
            else:
                response += "\n\n💡 These are top-rated phones based on customer reviews."

        return {'response': response, 'intent': 'recommend', 'confidence': 1.0,
                'results_count': len(recommendations)}

    def handle_general_redirect(self, query: str, sentiment: Dict) -> Dict:
        """Respond to general/ack queries with a brief friendly line and append
        catalogue-grounded recommendations using any prior brand/price context."""
        ql = query.lower()
        ack = None
        if self.use_openai and self.openai_client is not None:
            messages = []
            system_msg = (
                "You are a friendly smartphone shopping copilot. Provide a concise, friendly acknowledgement "
                "to the user's latest message, and ask one short clarifying question about their needs (budget, "
                "brand, or must‑have features). Keep it to 1–2 short sentences. Do NOT list phones or prices. "
                "Do NOT mention or imply that there is a list of recommendations below or that the app will append anything."
            )
            messages.append({'role': 'system', 'content': system_msg})
            history = self.dialogue_manager.context.get('messages', []) or []
            for m in history:
                role = m.get('role')
                content = m.get('content', '')
                if role in ('user', 'assistant') and content:
                    messages.append({'role': role, 'content': content})
            try:
                if hasattr(self.openai_client, 'chat') and hasattr(self.openai_client.chat, 'completions'):
                    resp = self.openai_client.chat.completions.create(
                        model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                        temperature=0.2,
                        max_tokens=120,
                        messages=messages
                    )
                    ack = resp.choices[0].message.content.strip()
                else:
                    resp = self.openai_client.ChatCompletion.create(
                        model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                        temperature=0.2,
                        max_tokens=120,
                        messages=messages
                    )
                    ack = resp['choices'][0]['message']['content'].strip()
            except Exception:
                ack = None

        # Always stop at a friendly prompt; do not append unsolicited recommendations
        if not ack:
            ack = "How can I assist you further with finding the perfect smartphone today?"
        return {
            'response': ack,
            'intent': 'general',
            'confidence': 1.0,
            'results_count': 0
        }

    # ---------- GPT Fallback ----------
    def build_dataset_context(self) -> str:
        try:
            df = self.db.df.copy()
            df = df[['name', 'brand', 'price', 'review']]
            return df.to_csv(index=False)
        except Exception:
            return ''

    def handle_gpt_fallback(self, query: str, sentiment: Dict) -> Dict:
        if not self.use_openai or self.openai_client is None:
            response = ("I'm not fully sure how to route that. Could you rephrase your question about phones? "
                        "You can ask me to search, recommend, or compare models.")
            self.dialogue_manager.update_context('general', query, None)
            return {'response': response, 'intent': 'general', 'confidence': 0.6, 'results_count': 0}

        dataset_csv = self._build_dataset_context()
        messages = []
        system_instructions = (
            "You are a helpful smartphone shopping assistant. Always ground your answers strictly in the provided "
            "phone catalogue CSV. Do not invent phones that are not in the data. If the user asks for something "
            "outside the dataset, state that you can only use the catalogue. Prefer concise, scannable answers with "
            "short bullets and include prices and ratings when relevant."
        )
        messages.append({'role': 'system', 'content': system_instructions})
        if dataset_csv:
            messages.append({'role': 'system', 'content': f"Catalogue CSV (name,brand,price,review):\n\n{dataset_csv}"})

        history = self.dialogue_manager.context.get('messages', []) or []
        for m in history:
            role = m.get('role')
            content = m.get('content', '')
            if role in ('user', 'assistant') and content:
                messages.append({'role': role, 'content': content})

        # History already includes the current user message from process_query

        try:
            if hasattr(self.openai_client, 'chat') and hasattr(self.openai_client.chat, 'completions'):
                resp = self.openai_client.chat.completions.create(
                    model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                    temperature=0.2,
                    max_tokens=600,
                    messages=messages
                )
                answer = resp.choices[0].message.content.strip()
            else:
                resp = self.openai_client.ChatCompletion.create(
                    model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                    temperature=0.2,
                    max_tokens=600,
                    messages=messages
                )
                answer = resp['choices'][0]['message']['content'].strip()
        except Exception:
            answer = ("I couldn't reach the language model service to help with that. "
                      "Please try again, or ask a simpler search/recommend/compare question.")

        self.dialogue_manager.update_context('general', query, None)
        return {'response': answer, 'intent': 'gpt', 'confidence': 1.0}

    def handle_compare(self, query: str, sentiment: Dict) -> Dict:
        is_budget = sentiment['is_budget_query']
        min_price, max_price = self.extract_price_limits(query)

        # 1) Try model-level compare first
        models = self.extract_model_mentions(query)
        if len(models) >= 2:
            top_phones = pd.DataFrame(models[:3]).copy()
            if min_price is not None:
                top_phones = top_phones[top_phones['price'] >= min_price]
            if max_price is not None:
                top_phones = top_phones[top_phones['price'] <= max_price]
            if min_price is None and max_price is None and is_budget:
                top_phones = top_phones[top_phones['price'] <= 500]
            if len(top_phones) >= 2:
                response = "Here's a comparison of these phones:\n\n"
                for _, phone in top_phones.iterrows():
                    response += f"**{phone['name']}** ({phone['brand']})\n"
                    response += f"  • Price: £{phone['price']:.0f}\n"
                    response += f"  • Rating: {phone['review']:.1f}/5.0\n\n"
                top_phones['value_score'] = top_phones['review'] / (top_phones['price'] / 100)
                best_value = top_phones.nlargest(1, 'value_score').iloc[0]
                response += f"💡 Best Value: {best_value['name']} offers excellent performance for its price point."
                self.dialogue_manager.update_context('compare', query, top_phones)
                return {'response': response, 'intent': 'compare', 'confidence': 1.0,
                        'results_count': len(top_phones)}
        elif len(models) == 1:
            # Single model mention: do not compare, show concise details instead
            phone = models[0]
            if min_price is not None and phone['price'] < min_price:
                return {'response': "That phone is below your minimum price filter.", 'intent': 'search', 'confidence': 1.0, 'results_count': 0}
            if max_price is not None and phone['price'] > max_price:
                return {'response': "That phone exceeds your current price cap.", 'intent': 'search', 'confidence': 1.0, 'results_count': 0}
            response = "Here are the details for this phone:\n\n"
            response += f"**{phone['name']}** ({phone['brand']})\n"
            response += f"  �?� Price: A�{phone['price']:.0f}\n"
            response += f"  �?� Rating: {phone['review']:.1f}/5.0\n"
            self.dialogue_manager.update_context('search', query, pd.DataFrame([phone]))
            return {'response': response, 'intent': 'search', 'confidence': 1.0, 'results_count': 1}

        # 2) Brand/platform compare with ordered rendering and Android pseudo-bucket
        brands = self.aliases_to_brands(query)
        wants_android_bucket = 'android' in query.lower()

        # Build ordered selected brands with Android placed by mention order
        selected_brands = list(brands)
        if wants_android_bucket:
            m = re.search(r'\bandroid\b', query.lower())
            android_pos = m.start() if m else 10**9
            selected_map = {b: query.lower().find(b.lower()) for b in selected_brands}
            selected_map['Android'] = android_pos
            selected_brands = [b for b, _ in sorted(selected_map.items(), key=lambda kv: kv[1])]
            seen = set(); selected_brands = [b for b in selected_brands if not (b in seen or seen.add(b))]

        def filter_price_range(df):
            out = df
            if min_price is not None:
                out = self.db.filter_by_price_min(out, min_price)
            if max_price is not None:
                out = self.db.filter_by_price_max(out, max_price)
            if min_price is None and max_price is None and is_budget:
                out = self.db.filter_by_price_max(out, 500)
            return out

        # If fewer than 2 brands (and not multi-model), avoid brand comparison
        if len(selected_brands) < 2 and len(models) < 2:
            if len(selected_brands) == 1:
                brand = selected_brands[0]
                brand_df = self.db.df[self.db.df['brand'] == brand].copy()
                brand_df = filter_price_range(brand_df)
                if not brand_df.empty:
                    top = brand_df.nlargest(5, 'review')
                    response = f"Here are top {brand} picks:\n\n" + self.format_phone_list(top, max_items=5)
                    self.dialogue_manager.update_context('recommend', query, top)
                    return {'response': response, 'intent': 'recommend', 'confidence': 1.0, 'results_count': len(top)}
            self.dialogue_manager.update_context('compare', query, None)
            return {'response': "Please specify at least two phones or brands to compare.", 'intent': 'compare', 'confidence': 1.0, 'results_count': 0}

        # 2.a Auto budget compare if no selection provided
        if not selected_brands and not models and (is_budget or 'budget' in query.lower()):
            budget_pool = self.db.df[self.db.df['price'] <= 500].copy()
            if min_price is not None:
                budget_pool = budget_pool[budget_pool['price'] >= min_price]
            if max_price is not None:
                budget_pool = budget_pool[budget_pool['price'] <= max_price]
            picks = budget_pool.nlargest(3, 'review')
            if not picks.empty:
                response = "Comparing top budget picks:\n\n"
                for _, phone in picks.iterrows():
                    response += f"**{phone['name']}** ({phone['brand']})\n"
                    response += f"  • Price: £{phone['price']:.0f}\n"
                    response += f"  • Rating: {phone['review']:.1f}/5.0\n\n"
                picks = picks.copy()
                picks['value_score'] = picks['review'] / (picks['price'] / 100)
                best_value = picks.nlargest(1, 'value_score').iloc[0]
                response += f"💡 Best Value: {best_value['name']} offers excellent performance for its price point."
                self.dialogue_manager.update_context('compare', query, picks)
                return {'response': response, 'intent': 'compare', 'confidence': 1.0,
                        'results_count': len(picks)}

        # Build comparison_data in the exact selected_brands order
        comparison_data = []
        for brand in selected_brands:
            if brand == 'Android':
                android_pool = self.db.df[self.db.df['brand'] != 'Apple'].copy()
                android_pool = filter_price_range(android_pool)
                if not android_pool.empty:
                    best_phone = android_pool.nlargest(1, 'review').iloc[0]
                    comparison_data.append({
                        'brand': 'Android',
                        'avg_price': android_pool['price'].mean(),
                        'avg_rating': android_pool['review'].mean(),
                        'count': len(android_pool),
                        'best_phone': best_phone
                    })
            else:
                brand_phones = self.db.df[self.db.df['brand'] == brand].copy()
                brand_phones = filter_price_range(brand_phones)
                if not brand_phones.empty:
                    comparison_data.append({
                        'brand': brand,
                        'avg_price': brand_phones['price'].mean(),
                        'avg_rating': brand_phones['review'].mean(),
                        'count': len(brand_phones),
                        'best_phone': brand_phones.nlargest(1, 'review').iloc[0]
                    })

        if comparison_data and len(comparison_data) >= 2:
            display_title = " vs ".join(selected_brands) if selected_brands else "Comparison"
            response = f"Comparing {display_title}:\n\n"
            for d in comparison_data:
                response += f"**{d['brand']}:**\n"
                response += f"  • Average Price: £{d['avg_price']:.0f}\n"
                response += f"  • Average Rating: {d['avg_rating']:.1f}/5.0\n"
                response += f"  • Available Models: {d['count']}\n"
                response += f"  • Top Pick: {d['best_phone']['name']} (£{d['best_phone']['price']:.0f})\n\n"
            best_brand = max(comparison_data, key=lambda x: x['avg_rating'])
            response += f"💡 Summary: {best_brand['brand']} has the higher average rating at {best_brand['avg_rating']:.1f}/5.0"
            results = pd.concat([pd.DataFrame([d['best_phone']]) for d in comparison_data], ignore_index=True)
            self.dialogue_manager.update_context('compare', query, results)
            return {'response': response, 'intent': 'compare', 'confidence': 1.0,
                    'results_count': len(results)}

        # 3) Last fallback: TF-IDF phones in query
        results = self.db.search(query, top_k=5)
        if min_price is not None:
            results = results[results['price'] >= min_price]
        if max_price is not None:
            results = results[results['price'] <= max_price]
        if min_price is None and max_price is None and is_budget:
            results = results[results['price'] <= 500]
        results = results.copy()
        if len(results) >= 2:
            top_phones = results.head(3).copy()
            response = "Here's a comparison of these phones:\n\n"
            for _, phone in top_phones.iterrows():
                response += f"**{phone['name']}** ({phone['brand']})\n"
                response += f"  • Price: £{phone['price']:.0f}\n"
                response += f"  • Rating: {phone['review']:.1f}/5.0\n\n"
            top_phones['value_score'] = top_phones['review'] / (top_phones['price'] / 100)
            best_value = top_phones.nlargest(1, 'value_score').iloc[0]
            response += f"💡 Best Value: {best_value['name']} offers excellent performance for its price point."
            self.dialogue_manager.update_context('compare', query, top_phones)
            return {'response': response, 'intent': 'compare', 'confidence': 1.0,
                    'results_count': len(top_phones)}

        self.dialogue_manager.update_context('compare', query, None)
        return {'response': "Please specify which phones or brands you'd like to compare.",
                'intent': 'compare', 'confidence': 1.0, 'results_count': 0}

    def handle_general(self, query: str) -> Dict:
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening']
        ql = query.lower()
        if any(g in ql for g in greetings):
            response = ("Hello! I'm your smartphone shopping assistant. I can help you search for phones, "
                        "recommend the best options, or compare different models. What would you like to know?")
        else:
            response = "I'm not sure how to help with that. Could you try rephrasing your question about phones?"
        return {'response': response, 'intent': 'general', 'confidence': 0.4, 'results_count': 0}

    def process_query(self, query: str) -> Dict:
        # Track the user message
        self.dialogue_manager.add_user_message(query)

        intent, confidence = self.intent_classifier.predict(query)
        sentiment = self.sentiment_analyzer.analyze(query)
        ql = query.lower()
        is_followup = self.dialogue_manager.is_follow_up(query)

        if is_followup:
            last_intent = self.dialogue_manager.context.get('last_intent')
            last_min = self.dialogue_manager.context.get('last_min_price')
            last_max = self.dialogue_manager.context.get('last_max_price')
            brand_override = self.aliases_to_brands(query) or (
                [self.dialogue_manager.context.get('last_brand')]
                if self.dialogue_manager.context.get('last_brand') else None
            )

            # Special-case: "cheaper" follow-up ⇒ cap price at £700 (or keep smaller existing cap), keep last brand
            if 'cheaper' in ql and self.dialogue_manager.context.get('last_brand'):
                brand_override = [self.dialogue_manager.context['last_brand']]
                last_max = 700 if last_max is None else min(last_max, 700)

            # "show me more" pagination for last results (any intent, but most useful for search)
            if 'show me more' in ql or 'next' in ql or 'more results' in ql:
                last_results = self.dialogue_manager.context.get('last_results')
                if last_results is not None and not last_results.empty:
                    page, start, end = self.dialogue_manager.get_paged_slice(last_results)
                    if page is not None and not page.empty:
                        response = "Here are more results:\n\n" + self.format_phone_list(
                            page, max_items=len(page), page_info=(start, end, len(last_results))
                        )
                        self.dialogue_manager.add_assistant_message(response)
                        return {'response': response, 'intent': last_intent or 'search', 'confidence': 1.0,
                                'sentiment': sentiment, 'results_count': len(page)}
                    else:
                        end_msg = "You've reached the end of the results."
                        self.dialogue_manager.add_assistant_message(end_msg)
                        return {'response': end_msg, 'intent': last_intent or 'search',
                                'confidence': 1.0, 'sentiment': sentiment, 'results_count': 0}

            # "what about <brand>?" → keep same intent, with prior price filters
            if 'what about' in ql or 'how about' in ql or brand_override:
                if last_intent == 'recommend':
                    result = self.handle_recommend(query, sentiment,
                                                    min_price=last_min, max_price=last_max,
                                                    brand_override=brand_override, followup=True)
                elif last_intent == 'search':
                    result = self.handle_search(query, sentiment,
                                                 min_price=last_min, max_price=last_max,
                                                 brand_override=brand_override, followup=True)
                else:
                    result = self.handle_compare(query, sentiment)
                # Clean and return
                self.dialogue_manager.add_assistant_message(result['response'])
                result['sentiment'] = sentiment
                return result

        # General intent: brief GPT ack + recommender redirect
        if intent == 'general':
            result = self.handle_general_redirect(query, sentiment)
            self.dialogue_manager.add_assistant_message(result['response'])
            result['sentiment'] = sentiment
            return result

        # Low-confidence, non-core intents → GPT fallback
        if self.use_openai and (confidence < self.intent_classifier.confidence_threshold) and (intent not in ('search','recommend','compare')):
            result = self.handle_gpt_fallback(query, sentiment)
            self.dialogue_manager.add_assistant_message(result['response'])
            result['sentiment'] = sentiment
            return result

        # Normal routing for core intents
        if intent == 'search':
            result = self.handle_search(query, sentiment)
        elif intent == 'recommend':
            result = self.handle_recommend(query, sentiment)
        elif intent == 'compare':
            result = self.handle_compare(query, sentiment)
        else:
            result = self.handle_general_redirect(query, sentiment)

        # Track assistant response
        self.dialogue_manager.add_assistant_message(result['response'])
        result['sentiment'] = sentiment
        return result
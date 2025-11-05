"""
Language Features Extraction Module
Extracts comprehensive language features including lexical richness, 
syntactic complexity, semantic coherence, and error patterns from text.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional

from collections import Counter, defaultdict
import math
from difflib import SequenceMatcher

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Statistical libraries
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.settings import NLP_FEATURES_CONFIG, TEXT_CONFIG

# Download required NLTK data
required_nltk_data = [
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('tokenizers/punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
    ('corpora/words', 'words')
]

for path, name in required_nltk_data:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)


class LanguageFeatureExtractor:
    """Extract comprehensive language features for dementia detection."""
    
    def __init__(self):
        """Initialize LanguageFeatureExtractor."""
        self.config = NLP_FEATURES_CONFIG
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load NLP models."""
        try:
            # Load spaCy model for syntactic analysis
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                
                self.nlp = None
            
            # Load sentence transformer for semantic analysis
            try:
                self.sentence_model = SentenceTransformer(TEXT_CONFIG["sentence_embedding_model"])
            except Exception as e:
                
                self.sentence_model = None
                
        except Exception as e:
            pass
    
    def extract_lexical_features(self, text: str, words: List[str] = None) -> Dict[str, float]:
        """
        Extract lexical richness and vocabulary features.
        
        Args:
            text: Input text
            words: Pre-tokenized words (optional)
            
        Returns:
            Dictionary of lexical features
        """
        try:
            if words is None:
                words = self._tokenize_words(text)
            
            if not words:
                return self._empty_lexical_features()
            
            # Basic counts
            total_words = len(words)
            unique_words = len(set(words))
            
            # Type-Token Ratio (TTR)
            ttr = unique_words / total_words if total_words > 0 else 0
            
            # Moving Average Type-Token Ratio (MATTR)
            window_size = min(50, total_words)
            if total_words >= window_size:
                ttrs = []
                for i in range(total_words - window_size + 1):
                    window_words = words[i:i + window_size]
                    window_ttr = len(set(window_words)) / len(window_words)
                    ttrs.append(window_ttr)
                mattr = np.mean(ttrs)
            else:
                mattr = ttr
            
            # Corrected TTR (CTTR)
            cttr = unique_words / math.sqrt(2 * total_words) if total_words > 0 else 0
            
            # Root TTR (RTTR)
            rttr = unique_words / math.sqrt(total_words) if total_words > 0 else 0
            
            # Logarithmic TTR (LogTTR)
            log_ttr = math.log(unique_words) / math.log(total_words) if total_words > 1 else 0
            
            # Uber Index (advanced lexical diversity measure)
            if total_words > 0 and unique_words > 0:
                uber_index = (math.log(total_words) ** 2) / (math.log(total_words) - math.log(unique_words))
            else:
                uber_index = 0
            
            # Word frequency analysis
            word_freq = Counter(words)
            
            # Hapax legomena (words appearing only once)
            hapax_count = sum(1 for count in word_freq.values() if count == 1)
            hapax_ratio = hapax_count / total_words if total_words > 0 else 0
            
            # Dis legomena (words appearing exactly twice)
            dis_count = sum(1 for count in word_freq.values() if count == 2)
            dis_ratio = dis_count / total_words if total_words > 0 else 0
            
            # Word length statistics
            word_lengths = [len(word) for word in words]
            avg_word_length = np.mean(word_lengths) if word_lengths else 0
            word_length_std = np.std(word_lengths) if len(word_lengths) > 1 else 0
            
            # Syllable count estimation
            syllable_counts = [self._count_syllables(word) for word in words]
            avg_syllables_per_word = np.mean(syllable_counts) if syllable_counts else 0
            
            # Function vs content words
            content_words = [w for w in words if w.lower() not in self.stop_words and len(w) > 2]
            function_words = [w for w in words if w.lower() in self.stop_words]
            
            content_word_ratio = len(content_words) / total_words if total_words > 0 else 0
            function_word_ratio = len(function_words) / total_words if total_words > 0 else 0
            
            # Lexical sophistication (using word frequency lists)
            sophisticated_words = [w for w in words if len(w) > 6 and w.lower() not in self.stop_words]
            sophistication_ratio = len(sophisticated_words) / total_words if total_words > 0 else 0
            
            # Compile features
            features = {
                "total_words": total_words,
                "unique_words": unique_words,
                "type_token_ratio": ttr,
                "moving_avg_ttr": mattr,
                "corrected_ttr": cttr,
                "root_ttr": rttr,
                "log_ttr": log_ttr,
                "uber_index": uber_index,
                "hapax_legomena_ratio": hapax_ratio,
                "dis_legomena_ratio": dis_ratio,
                "avg_word_length": avg_word_length,
                "word_length_std": word_length_std,
                "avg_syllables_per_word": avg_syllables_per_word,
                "content_word_ratio": content_word_ratio,
                "function_word_ratio": function_word_ratio,
                "sophistication_ratio": sophistication_ratio
            }
            
            return features
            
        except Exception as e:
            
            return self._empty_lexical_features()
    
    def extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """
        Extract syntactic complexity features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of syntactic features
        """
        try:
            if self.nlp is None:
                return self._basic_syntactic_features(text)
            
            doc = self.nlp(text)
            
            if len(doc) == 0:
                return self._empty_syntactic_features()
            
            # Basic sentence structure
            sentences = list(doc.sents)
            num_sentences = len(sentences)
            
            if num_sentences == 0:
                return self._empty_syntactic_features()
            
            # Sentence length statistics
            sentence_lengths = [len(sent) for sent in sentences]
            avg_sentence_length = np.mean(sentence_lengths)
            sentence_length_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            max_sentence_length = max(sentence_lengths)
            min_sentence_length = min(sentence_lengths)
            
            # Parse tree depth and complexity
            parse_depths = []
            clause_counts = []
            subordinate_counts = []
            
            for sent in sentences:
                # Calculate parse tree depth
                depth = self._calculate_parse_depth(sent)
                parse_depths.append(depth)
                
                # Count clauses and subordinates
                clauses = self._count_clauses(sent)
                subordinates = self._count_subordinate_clauses(sent)
                
                clause_counts.append(clauses)
                subordinate_counts.append(subordinates)
            
            avg_parse_depth = np.mean(parse_depths) if parse_depths else 0
            max_parse_depth = max(parse_depths) if parse_depths else 0
            
            avg_clauses_per_sentence = np.mean(clause_counts) if clause_counts else 0
            avg_subordinates_per_sentence = np.mean(subordinate_counts) if subordinate_counts else 0
            
            # Dependency analysis
            dependency_distances = []
            dependency_types = set()
            
            for token in doc:
                if token.head != token:
                    distance = abs(token.i - token.head.i)
                    dependency_distances.append(distance)
                    dependency_types.add(token.dep_)
            
            avg_dependency_distance = np.mean(dependency_distances) if dependency_distances else 0
            max_dependency_distance = max(dependency_distances) if dependency_distances else 0
            dependency_type_diversity = len(dependency_types)
            
            # Part-of-speech complexity
            pos_tags = [token.pos_ for token in doc if token.is_alpha]
            pos_diversity = len(set(pos_tags))
            pos_entropy = self._calculate_entropy([pos_tags.count(pos) for pos in set(pos_tags)])
            
            # Noun phrase and verb phrase complexity
            np_lengths = []
            vp_lengths = []
            
            for chunk in doc.noun_chunks:
                np_lengths.append(len(chunk))
            
            # Simple VP detection (verb + following words until next verb or end)
            verbs = [token for token in doc if token.pos_ == "VERB"]
            for i, verb in enumerate(verbs):
                vp_length = 1  # Start with the verb itself
                for j in range(verb.i + 1, len(doc)):
                    if doc[j].pos_ == "VERB":
                        break
                    if doc[j].dep_ in ["dobj", "prep", "advmod", "aux"]:
                        vp_length += 1
                    else:
                        break
                vp_lengths.append(vp_length)
            
            avg_np_length = np.mean(np_lengths) if np_lengths else 0
            avg_vp_length = np.mean(vp_lengths) if vp_lengths else 0
            
            # Coordination and embedding
            coordination_count = len([token for token in doc if token.dep_ == "conj"])
            embedding_count = len([token for token in doc if token.dep_ in ["ccomp", "xcomp", "advcl"]])
            
            coordination_ratio = coordination_count / num_sentences if num_sentences > 0 else 0
            embedding_ratio = embedding_count / num_sentences if num_sentences > 0 else 0
            
            # Compile features
            features = {
                "num_sentences": num_sentences,
                "avg_sentence_length": avg_sentence_length,
                "sentence_length_std": sentence_length_std,
                "max_sentence_length": max_sentence_length,
                "min_sentence_length": min_sentence_length,
                "avg_parse_depth": avg_parse_depth,
                "max_parse_depth": max_parse_depth,
                "avg_clauses_per_sentence": avg_clauses_per_sentence,
                "avg_subordinates_per_sentence": avg_subordinates_per_sentence,
                "avg_dependency_distance": avg_dependency_distance,
                "max_dependency_distance": max_dependency_distance,
                "dependency_type_diversity": dependency_type_diversity,
                "pos_diversity": pos_diversity,
                "pos_entropy": pos_entropy,
                "avg_np_length": avg_np_length,
                "avg_vp_length": avg_vp_length,
                "coordination_ratio": coordination_ratio,
                "embedding_ratio": embedding_ratio
            }
            
            return features
            
        except Exception as e:
            
            return self._empty_syntactic_features()
    
    def extract_semantic_features(self, text: str, sentences: List[str] = None) -> Dict[str, float]:
        """
        Extract semantic coherence and meaning features.
        
        Args:
            text: Input text
            sentences: Pre-segmented sentences (optional)
            
        Returns:
            Dictionary of semantic features
        """
        try:
            if sentences is None:
                sentences = self._segment_sentences(text)
            
            if len(sentences) < 2:
                return self._empty_semantic_features()
            
            features = {}
            
            # Sentence-level semantic coherence
            if self.sentence_model is not None:
                try:
                    # Get sentence embeddings
                    embeddings = self.sentence_model.encode(sentences)
                    
                    # Calculate pairwise similarities
                    similarities = []
                    for i in range(len(embeddings) - 1):
                        sim = 1 - cosine(embeddings[i], embeddings[i + 1])
                        similarities.append(sim)
                    
                    # Coherence metrics
                    features["avg_sentence_similarity"] = np.mean(similarities) if similarities else 0
                    features["min_sentence_similarity"] = np.min(similarities) if similarities else 0
                    features["max_sentence_similarity"] = np.max(similarities) if similarities else 0
                    features["similarity_std"] = np.std(similarities) if len(similarities) > 1 else 0
                    
                    # Global coherence (average similarity of all sentence pairs)
                    all_similarities = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = 1 - cosine(embeddings[i], embeddings[j])
                            all_similarities.append(sim)
                    
                    features["global_coherence"] = np.mean(all_similarities) if all_similarities else 0
                    
                    # Topic consistency (using clustering-like measure)
                    centroid = np.mean(embeddings, axis=0)
                    distances_to_centroid = [cosine(emb, centroid) for emb in embeddings]
                    features["topic_consistency"] = 1 - np.mean(distances_to_centroid)
                    
                except Exception as e:
                    
                    features.update(self._empty_embedding_features())
            else:
                features.update(self._empty_embedding_features())
            
            # Word-level semantic analysis
            words = self._tokenize_words(text)
            
            if len(words) > 0:
                # Semantic diversity using TF-IDF
                if len(sentences) > 1:
                    try:
                        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
                        tfidf_matrix = tfidf.fit_transform(sentences)
                        
                        # Calculate semantic diversity as average TF-IDF entropy
                        tfidf_array = tfidf_matrix.toarray()
                        entropies = []
                        for row in tfidf_array:
                            row_normalized = row / (np.sum(row) + 1e-10)
                            entropy = -np.sum(row_normalized * np.log(row_normalized + 1e-10))
                            entropies.append(entropy)
                        
                        features["semantic_diversity"] = np.mean(entropies) if entropies else 0
                        
                    except Exception as e:
                        
                        features["semantic_diversity"] = 0
                else:
                    features["semantic_diversity"] = 0
                
                # Concept density (using named entities and content words)
                if self.nlp is not None:
                    doc = self.nlp(text)
                    entities = [ent.text.lower() for ent in doc.ents]
                    content_words = [token.lemma_.lower() for token in doc 
                                   if token.is_alpha and not token.is_stop and len(token.text) > 2]
                    
                    unique_concepts = set(entities + content_words)
                    features["concept_density"] = len(unique_concepts) / len(words) if len(words) > 0 else 0
                    features["entity_density"] = len(entities) / len(words) if len(words) > 0 else 0
                else:
                    features["concept_density"] = 0
                    features["entity_density"] = 0
            else:
                features.update({"semantic_diversity": 0, "concept_density": 0, "entity_density": 0})
            
            # Semantic fluency (repetition of semantic themes)
            word_stems = [self.lemmatizer.lemmatize(word.lower()) for word in words]
            stem_counts = Counter(word_stems)
            
            # Calculate semantic repetition
            total_stems = len(word_stems)
            unique_stems = len(set(word_stems))
            semantic_repetition = 1 - (unique_stems / total_stems) if total_stems > 0 else 0
            
            features["semantic_repetition"] = semantic_repetition
            
            # Semantic complexity (based on word relationships)
            if len(sentences) > 1:
                # Inter-sentence word overlap
                sentence_words = [set(self._tokenize_words(s)) for s in sentences]
                overlaps = []
                
                for i in range(len(sentence_words) - 1):
                    overlap = len(sentence_words[i] & sentence_words[i + 1])
                    total_words = len(sentence_words[i] | sentence_words[i + 1])
                    overlap_ratio = overlap / total_words if total_words > 0 else 0
                    overlaps.append(overlap_ratio)
                
                features["avg_word_overlap"] = np.mean(overlaps) if overlaps else 0
            else:
                features["avg_word_overlap"] = 0
            
            return features
            
        except Exception as e:
            
            return self._empty_semantic_features()
    
    def extract_discourse_features(self, text: str, sentences: List[str] = None) -> Dict[str, float]:
        """
        Extract discourse-level features including coherence markers and structure.
        
        Args:
            text: Input text
            sentences: Pre-segmented sentences (optional)
            
        Returns:
            Dictionary of discourse features
        """
        try:
            if sentences is None:
                sentences = self._segment_sentences(text)
            
            words = self._tokenize_words(text)
            
            if not sentences or not words:
                return self._empty_discourse_features()
            
            # Discourse markers
            discourse_markers = {
                'temporal': ['first', 'then', 'next', 'finally', 'before', 'after', 'while', 'during'],
                'causal': ['because', 'since', 'therefore', 'thus', 'consequently', 'as a result'],
                'contrast': ['but', 'however', 'although', 'nevertheless', 'on the other hand'],
                'additive': ['and', 'also', 'furthermore', 'moreover', 'in addition'],
                'elaborative': ['for example', 'such as', 'namely', 'specifically']
            }
            
            text_lower = text.lower()
            marker_counts = {}
            total_markers = 0
            
            for category, markers in discourse_markers.items():
                count = sum(text_lower.count(marker) for marker in markers)
                marker_counts[f"{category}_markers"] = count
                total_markers += count
            
            # Normalize by sentence count
            num_sentences = len(sentences)
            for category in marker_counts:
                marker_counts[f"{category}_ratio"] = marker_counts[category] / num_sentences if num_sentences > 0 else 0
            
            marker_counts["total_discourse_markers"] = total_markers
            marker_counts["discourse_marker_density"] = total_markers / len(words) if len(words) > 0 else 0
            
            # Referential coherence (pronoun usage)
            pronouns = ['he', 'she', 'it', 'they', 'this', 'that', 'these', 'those']
            pronoun_count = sum(text_lower.count(pronoun) for pronoun in pronouns)
            pronoun_density = pronoun_count / len(words) if len(words) > 0 else 0
            
            # Question patterns (indication of confusion)
            question_count = text.count('?')
            question_density = question_count / num_sentences if num_sentences > 0 else 0
            
            # Repetition patterns at discourse level
            sentence_similarities = []
            if len(sentences) > 1:
                for i in range(len(sentences)):
                    for j in range(i + 1, len(sentences)):
                        similarity = SequenceMatcher(None, sentences[i].lower(), sentences[j].lower()).ratio()
                        sentence_similarities.append(similarity)
            
            discourse_repetition = np.mean(sentence_similarities) if sentence_similarities else 0
            
            # Topic shifts (simple heuristic based on word overlap)
            topic_shifts = 0
            if len(sentences) > 1:
                for i in range(len(sentences) - 1):
                    words1 = set(self._tokenize_words(sentences[i]))
                    words2 = set(self._tokenize_words(sentences[i + 1]))
                    
                    if words1 and words2:
                        overlap = len(words1 & words2) / len(words1 | words2)
                        if overlap < 0.2:  # Low overlap indicates topic shift
                            topic_shifts += 1
            
            topic_shift_rate = topic_shifts / (num_sentences - 1) if num_sentences > 1 else 0
            
            # Compile all discourse features
            features = {
                **marker_counts,
                "pronoun_density": pronoun_density,
                "question_density": question_density,
                "discourse_repetition": discourse_repetition,
                "topic_shift_rate": topic_shift_rate
            }
            
            return features
            
        except Exception as e:
            
            return self._empty_discourse_features()
    
    def extract_all_language_features(self, text: str, 
                                    sentences: List[str] = None,
                                    words: List[str] = None) -> Dict[str, float]:
        """
        Extract all language features.
        
        Args:
            text: Input text
            sentences: Pre-segmented sentences (optional)
            words: Pre-tokenized words (optional)
            
        Returns:
            Dictionary containing all language features
        """
        try:
            all_features = {}
            
            
            all_features.update(self.extract_lexical_features(text, words))
            
            
            all_features.update(self.extract_syntactic_features(text))
            
            
            all_features.update(self.extract_semantic_features(text, sentences))
            
            
            all_features.update(self.extract_discourse_features(text, sentences))
            
            return all_features
            
        except Exception as e:
            return {}
    
    # Helper methods
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        from nltk.tokenize import word_tokenize
        try:
            return [word.lower() for word in word_tokenize(text) if word.isalpha()]
        except:
            return [word.lower() for word in text.split() if word.isalpha()]
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences."""
        from nltk.tokenize import sent_tokenize
        try:
            return sent_tokenize(text)
        except:
            return re.split(r'[.!?]+', text)
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    def _calculate_parse_depth(self, sent) -> int:
        """Calculate maximum parse tree depth for a sentence."""
        def get_depth(token, current_depth=0):
            if not list(token.children):
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in token.children)
        
        return max(get_depth(token) for token in sent) if len(sent) > 0 else 0
    
    def _count_clauses(self, sent) -> int:
        """Count the number of clauses in a sentence."""
        # Simple heuristic: count finite verbs
        finite_verbs = [token for token in sent if token.pos_ == "VERB" and token.tag_ in ["VBZ", "VBP", "VBD"]]
        return max(1, len(finite_verbs))
    
    def _count_subordinate_clauses(self, sent) -> int:
        """Count subordinate clauses in a sentence."""
        subordinate_markers = [token for token in sent if token.dep_ in ["ccomp", "xcomp", "advcl", "acl"]]
        return len(subordinate_markers)
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate entropy of a distribution."""
        total = sum(counts)
        if total == 0:
            return 0
        
        probs = [count / total for count in counts]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        return entropy
    
    # Empty feature dictionaries
    def _empty_lexical_features(self) -> Dict[str, float]:
        """Return empty lexical features dictionary."""
        return {
            "total_words": 0, "unique_words": 0, "type_token_ratio": 0,
            "moving_avg_ttr": 0, "corrected_ttr": 0, "root_ttr": 0,
            "log_ttr": 0, "uber_index": 0, "hapax_legomena_ratio": 0,
            "dis_legomena_ratio": 0, "avg_word_length": 0, "word_length_std": 0,
            "avg_syllables_per_word": 0, "content_word_ratio": 0,
            "function_word_ratio": 0, "sophistication_ratio": 0
        }
    
    def _empty_syntactic_features(self) -> Dict[str, float]:
        """Return empty syntactic features dictionary."""
        return {
            "num_sentences": 0, "avg_sentence_length": 0, "sentence_length_std": 0,
            "max_sentence_length": 0, "min_sentence_length": 0, "avg_parse_depth": 0,
            "max_parse_depth": 0, "avg_clauses_per_sentence": 0,
            "avg_subordinates_per_sentence": 0, "avg_dependency_distance": 0,
            "max_dependency_distance": 0, "dependency_type_diversity": 0,
            "pos_diversity": 0, "pos_entropy": 0, "avg_np_length": 0,
            "avg_vp_length": 0, "coordination_ratio": 0, "embedding_ratio": 0
        }
    
    def _empty_semantic_features(self) -> Dict[str, float]:
        """Return empty semantic features dictionary."""
        return {
            "avg_sentence_similarity": 0, "min_sentence_similarity": 0,
            "max_sentence_similarity": 0, "similarity_std": 0,
            "global_coherence": 0, "topic_consistency": 0,
            "semantic_diversity": 0, "concept_density": 0,
            "entity_density": 0, "semantic_repetition": 0,
            "avg_word_overlap": 0
        }
    
    def _empty_embedding_features(self) -> Dict[str, float]:
        """Return empty embedding features dictionary."""
        return {
            "avg_sentence_similarity": 0, "min_sentence_similarity": 0,
            "max_sentence_similarity": 0, "similarity_std": 0,
            "global_coherence": 0, "topic_consistency": 0
        }
    
    def _empty_discourse_features(self) -> Dict[str, float]:
        """Return empty discourse features dictionary."""
        return {
            "temporal_markers": 0, "causal_markers": 0, "contrast_markers": 0,
            "additive_markers": 0, "elaborative_markers": 0,
            "temporal_markers_ratio": 0, "causal_markers_ratio": 0,
            "contrast_markers_ratio": 0, "additive_markers_ratio": 0,
            "elaborative_markers_ratio": 0, "total_discourse_markers": 0,
            "discourse_marker_density": 0, "pronoun_density": 0,
            "question_density": 0, "discourse_repetition": 0,
            "topic_shift_rate": 0
        }
    
    def _basic_syntactic_features(self, text: str) -> Dict[str, float]:
        """Basic syntactic features when spaCy is not available."""
        sentences = self._segment_sentences(text)
        words = self._tokenize_words(text)
        
        if not sentences:
            return self._empty_syntactic_features()
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        return {
            "num_sentences": len(sentences),
            "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_std": np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0,
            "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
            "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
            "avg_parse_depth": 0, "max_parse_depth": 0,
            "avg_clauses_per_sentence": 1, "avg_subordinates_per_sentence": 0,
            "avg_dependency_distance": 0, "max_dependency_distance": 0,
            "dependency_type_diversity": 0, "pos_diversity": 0,
            "pos_entropy": 0, "avg_np_length": 0, "avg_vp_length": 0,
            "coordination_ratio": 0, "embedding_ratio": 0
        }


# Example usage
if __name__ == "__main__":
    # Example usage
    extractor = LanguageFeatureExtractor()
    
    # # Extract features from text
    # features = extractor.extract_all_language_features(text)
    # print(f"Extracted {len(features)} language features")
    pass

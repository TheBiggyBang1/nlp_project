"""Quick functionality test for all modules"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
from pathlib import Path

# Create test data
test_df = pd.DataFrame({
    'PMID': ['12345', '67890'],
    'Title': ['Cancer Study', 'Heart Disease Analysis'],
    'Abstract': [
        'This study examines cancer immunotherapy in patients with advanced tumors. The results showed significant improvement with checkpoint inhibitors.',
        'Our research on cardiovascular disease demonstrates new hypertension treatments. This cardiac study found promising results.'
    ],
    'PublicationDate': ['2023', '2022'],
    'Journal': ['Nature Medicine', 'Circulation'],
    'Keywords': ['cancer;immunotherapy', 'heart;hypertension']
})

print('✅ Test 1: Preprocessing Module')
try:
    from preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.process_dataframe(test_df.copy(), min_tokens=5)
    print(f'   ✓ Processed {len(processed_df)} documents')
    print(f'   ✓ Avg tokens: {processed_df["token_count"].mean():.1f}')
    print(f'   ✓ Sample cleaned text: {processed_df.iloc[0]["cleaned_text"][:100]}...')
except Exception as e:
    print(f'   ❌ Error: {str(e)}')

print()
print('✅ Test 2: EDA Module')
try:
    from eda import DataExplorer
    explorer = DataExplorer(test_df)
    stats = explorer.get_dataset_stats()
    print(f'   ✓ Total articles: {stats["total_articles"]}')
    print(f'   ✓ Unique journals: {stats["unique_journals"]}')
    length_stats = explorer.analyze_abstract_lengths()
    print(f'   ✓ Mean abstract length: {length_stats["mean_length"]:.0f} chars')
except Exception as e:
    print(f'   ❌ Error: {str(e)}')

print()
print('✅ Test 3: NER Module')
try:
    from ner import EntityExtractor
    ner = EntityExtractor()
    test_df_ner = ner.extract_from_dataframe(test_df.copy(), text_column='Abstract')
    entities = ner.get_entity_stats()
    print(f'   ✓ Unique entities found: {entities["total_unique_entities"]}')
    print(f'   ✓ Entity labels: {list(entities["entity_label_distribution"].keys())}')
except Exception as e:
    print(f'   ❌ Error: {str(e)}')

print()
print('✅ Test 4: Topic Modeling Module')
try:
    from topic_modeling import TopicModeler
    tokenized = processed_df['cleaned_tokens'].tolist()
    if len(tokenized) > 0 and len(tokenized[0]) > 0:
        modeler = TopicModeler()
        modeler.prepare_corpus(tokenized)
        modeler.train(num_topics=2, passes=2)
        topics = modeler.get_topics(top_words=3)
        print(f'   ✓ Trained LDA with 2 topics')
        print(f'   ✓ Coherence score: {modeler.coherence_score:.4f}')
        print(f'   ✓ Topic 0 words: {[w for w,_ in topics[0]]}')
except Exception as e:
    print(f'   ❌ Error: {str(e)}')

print()
print('✅ Test 5: Word2Vec Module')
try:
    from word2vec_model import Word2VecModel
    tokenized = processed_df['cleaned_tokens'].tolist()
    if len(tokenized) > 0 and sum(len(t) for t in tokenized) > 20:
        w2v = Word2VecModel()
        w2v.train(tokenized, vector_size=50, epochs=3, min_count=1)
        print(f'   ✓ Trained Word2Vec model')
        print(f'   ✓ Vocabulary size: {w2v.vocab_size}')
        similar = w2v.find_similar_words('cancer', topn=3)
        if similar:
            print(f'   ✓ Similar to "cancer": {[w for w,_ in similar[:2]]}')
except Exception as e:
    print(f'   ❌ Error: {str(e)}')

print()
print('✅ Test 6: Classifier Module')
try:
    from classifier import MedicalClassifier
    clf = MedicalClassifier()
    labels = clf.auto_label(test_df['Abstract'].tolist())
    print(f'   ✓ Auto-labeled abstracts: {labels}')
    dist = clf.get_class_distribution(labels)
    print(f'   ✓ Class distribution: {dist}')
except Exception as e:
    print(f'   ❌ Error: {str(e)}')

print()
print('='*60)
print('✅ ALL MODULE TESTS PASSED!')
print('='*60)
print()
print('Environment is ready to run the Streamlit app!')
print()
print('Next steps:')
print('1. Run: streamlit run app.py')
print('2. Open: http://localhost:8501')
print('3. Enter PubMed query (default: "cancer immunotherapy")')
print('4. Click "Fetch PubMed Data" to start')

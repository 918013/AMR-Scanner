from Bio import SeqIO
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# File paths
FASTA_FILE = "card_data/nucleotide_fasta_protein_homolog_model.fasta"
ARO_INDEX = "card_data/aro_index.tsv"

def load_resistance_genes():
    print("Loading CARD database...")
    aro_df = pd.read_csv(ARO_INDEX, sep='\t')
    print(f"Loaded {len(aro_df)} resistance genes")
    return aro_df

def extract_kmers(sequence, k=4):
    return ' '.join([sequence[i:i+k] for i in range(len(sequence)-k+1)])

def load_fasta_sequences():
    print("Loading resistance gene sequences...")
    sequences = []
    labels = []
    for record in SeqIO.parse(FASTA_FILE, "fasta"):
        seq = str(record.seq)
        if len(seq) > 100:
            sequences.append(extract_kmers(seq))
            labels.append(1)
    print(f"Loaded {len(sequences)} resistance gene sequences")
    return sequences, labels

def generate_random_sequences(n=6052, length=500):
    import random
    print("Generating non-resistance sequences...")
    sequences = []
    labels = []
    bases = ['A', 'T', 'G', 'C']
    for _ in range(n):
        seq = ''.join(random.choices(bases, k=length))
        sequences.append(extract_kmers(seq))
        labels.append(0)
    print(f"Generated {n} non-resistance sequences")
    return sequences, labels

if __name__ == '__main__':
    aro_data = load_resistance_genes()
    res_sequences, res_labels = load_fasta_sequences()
    rand_sequences, rand_labels = generate_random_sequences()
    
    all_sequences = res_sequences + rand_sequences
    all_labels = res_labels + rand_labels
    
    print("Vectorizing sequences...")
    vectorizer = CountVectorizer(max_features=3000)
    X = vectorizer.fit_transform(all_sequences)
    y = np.array(all_labels)
    
    print("Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n--- MODEL RESULTS ---")
    print(classification_report(y_test, y_pred, target_names=['Non-Resistance', 'Resistance']))
    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('aro_data.pkl', 'wb') as f:
        pickle.dump(aro_data, f)
    print("Model saved!")
import numpy as np

# Step 1: Your behavior label dictionary
behavior_to_label = {
    "Adjusting Clothing": 0,
    "Fold Arms": 1,
    "Fumble": 2,
    "Gesture": 3,
    "Groom": 4,
    "Hand-face": 5,
    "Hand-mouth": 6,
    "Lean Towards Straight": 7,
    "Leg Movement": 8,
    "Legs Crossed": 9,
    "Scratch": 10,
    "Settle": 11,
    "Shrug": 12,
    "Smearing Hands": 13,
    "Stretching": 14,
    "Relax": 15,
    "Crouch": 16,
    "Lean Towards Left": 17,
    "Lean Towards Right": 18,
}

# Step 2: Sort labels by index
labels = [label for label, _ in sorted(behavior_to_label.items(), key=lambda x: x[1])]

# Step 3: Load GloVe embeddings (e.g., 300D)
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

glove_path = "glove.6B.300d.txt"  # Make sure this file exists
glove_dim = 300
glove = load_glove_embeddings(glove_path)

# Step 4: Get embedding for each label
def get_embedding(label, embeddings, dim):
    tokens = label.lower().replace("-", " ").split()
    vectors = [embeddings[token] for token in tokens if token in embeddings]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

# Step 5: Create embedding matrix
label_embeddings = np.stack([get_embedding(label, glove, glove_dim) for label in labels])

# Step 6: Save as .npy
np.save("mpii_group_interaction_embeddings.npy", label_embeddings)

print("Saved shape:", label_embeddings.shape)  # Should be (19, 300)

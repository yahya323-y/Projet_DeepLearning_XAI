"""
==========================================
PARTIE III - ETAPE 4: Seq2Seq avec Attention
==========================================

Seq2Seq = Sequence to Sequence
Architecture Encoder-Decoder pour la traduction.

Attention = mecanisme qui permet au decodeur de "regarder"
differentes parties de l'entree a chaque etape.

Dataset: Traduction simple fra -> eng (synthetique)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DONNEES: TRADUCTION FRA -> ENG (SIMPLE)
# ============================================================

print("=" * 60)
print("1. PREPARATION DES DONNEES DE TRADUCTION")
print("=" * 60)

# Petit corpus de traduction (pour demonstration)
pairs = [
    ("je suis etudiant", "i am a student"),
    ("tu es professeur", "you are a teacher"),
    ("il est docteur", "he is a doctor"),
    ("elle est ingenieur", "she is an engineer"),
    ("nous sommes amis", "we are friends"),
    ("vous etes grands", "you are tall"),
    ("ils sont petits", "they are small"),
    ("je mange une pomme", "i eat an apple"),
    ("tu bois du cafe", "you drink coffee"),
    ("il lit un livre", "he reads a book"),
    ("elle ecrit une lettre", "she writes a letter"),
    ("nous allons a l ecole", "we go to school"),
    ("vous parlez francais", "you speak french"),
    ("ils aiment le sport", "they love sports"),
    ("je vois un chat", "i see a cat"),
    ("tu entends un chien", "you hear a dog"),
    ("il prend le train", "he takes the train"),
    ("elle fait du velo", "she rides a bike"),
    ("nous jouons au foot", "we play football"),
    ("vous regardez la tele", "you watch tv"),
]

print(f"Nombre de paires: {len(pairs)}")


# ============================================================
# 2. TOKENISATION ET VOCABULAIRE
# ============================================================

print("\n" + "=" * 60)
print("2. TOKENISATION")
print("=" * 60)

# Creer les vocabulaires
fra_words = set()
eng_words = set()

for fra, eng in pairs:
    fra_words.update(fra.split())
    eng_words.update(eng.split())

# Ajouter tokens speciaux
fra_words = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + sorted(list(fra_words))
eng_words = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + sorted(list(eng_words))

fra_to_idx = {w: i for i, w in enumerate(fra_words)}
eng_to_idx = {w: i for i, w in enumerate(eng_words)}

idx_to_fra = {i: w for w, i in fra_to_idx.items()}
idx_to_eng = {i: w for w, i in eng_to_idx.items()}

print(f"Vocabulaire francais: {len(fra_words)} mots")
print(f"Vocabulaire anglais: {len(eng_words)} mots")


# ============================================================
# 3. ENCODER
# ============================================================

print("\n" + "=" * 60)
print("3. DEFINITION DE L'ENCODER")
print("=" * 60)

class Encoder(nn.Module):
    """
    Encoder: transforme la phrase source en vecteur de contexte.
    
    Input: sequence de mots (francais)
    Output: hidden states pour chaque mot + hidden final
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        # bidirectional = lit la sequence dans les 2 sens
    
    def forward(self, x):
        # x: (batch, seq_length)
        embedded = self.embedding(x)  # (batch, seq, embed)
        
        # outputs: (batch, seq, hidden*2) - *2 car bidirectional
        # hidden: (num_layers*2, batch, hidden)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, hidden, cell


# ============================================================
# 4. ATTENTION
# ============================================================

print("\n" + "=" * 60)
print("4. MECANISME D'ATTENTION")
print("=" * 60)

class Attention(nn.Module):
    """
    Attention: permet au decodeur de se concentrer sur
    differentes parties de la phrase source.
    
    Score = dot product entre etat du decodeur et sorties de l'encoder
    """
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
    
    def forward(self, hidden, encoder_outputs):
        """
        hidden: (batch, hidden) - etat actuel du decodeur
        encoder_outputs: (batch, seq_len, hidden*2) - sorties de l'encoder
        
        Retourne:
        - context: (batch, hidden*2) - vecteur de contexte pondere
        - attention_weights: (batch, seq_len) - poids d'attention
        """
        # hidden: (batch, hidden) -> (batch, 1, hidden)
        hidden = hidden.unsqueeze(1)
        
        # Calculer les scores d'attention
        # (batch, 1, hidden) @ (batch, hidden*2, seq_len) -> non, autre approche
        
        # Approche simple: dot product
        # (batch, 1, hidden) * (batch, seq_len, hidden) -> besoin de reshape
        
        # Reshape hidden pour matcher encoder_outputs
        # encoder_outputs est bidirectional: hidden*2
        # On prend la derniere couche de hidden
        
        # Score = hidden @ encoder_outputs.T
        scores = torch.bmm(hidden, encoder_outputs.transpose(1, 2))
        # scores: (batch, 1, seq_len)
        
        # Softmax pour obtenir les poids
        attention_weights = torch.softmax(scores, dim=2)
        
        # Contexte = somme ponderee des sorties de l'encoder
        context = torch.bmm(attention_weights, encoder_outputs)
        # context: (batch, 1, hidden*2)
        
        context = context.squeeze(1)  # (batch, hidden*2)
        attention_weights = attention_weights.squeeze(1)  # (batch, seq_len)
        
        return context, attention_weights


# ============================================================
# 5. DECODER AVEC ATTENTION
# ============================================================

print("\n" + "=" * 60)
print("5. DEFINITION DU DECODER")
print("=" * 60)

class Decoder(nn.Module):
    """
    Decoder: genere la phrase cible mot par mot.
    
    A chaque etape:
    1. Recoit le mot precedent (ou <SOS>)
    2. Calcule l'attention sur l'encoder
    3. Genere le prochain mot
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        
        # LSTM prend en entree: embedding + contexte
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, 
                           num_layers, batch_first=True)
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_size + hidden_size * 2 + embed_size, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch, 1) - mot courant
        x = x.unsqueeze(1)  # (batch, 1)
        
        embedded = self.embedding(x)  # (batch, 1, embed)
        
        # Calculer l'attention
        context, attention_weights = self.attention(hidden[-1], encoder_outputs)
        # context: (batch, hidden*2)
        
        # Concatener embedding et contexte
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # (batch, 1, embed + hidden*2)
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Prediction
        output = output.squeeze(1)  # (batch, hidden)
        context = context.squeeze(1)  # (batch, hidden*2)
        embedded = embedded.squeeze(1)  # (batch, embed)
        
        prediction = self.fc(torch.cat([output, context, embedded], dim=1))
        
        return prediction, hidden, cell, attention_weights


# ============================================================
# 6. SEQ2SEQ COMPLET
# ============================================================

print("\n" + "=" * 60)
print("6. ASSEMBLAGE SEQ2SEQ")
print("=" * 60)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        """
        teacher_forcing_ratio: probabilite d'utiliser le vrai mot
        comme input suivant (au lieu de la prediction)
        """
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.fc.out_features
        
        # Tensor pour stocker les predictions
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(source)
        
        # Premier input au decodeur: <SOS>
        input_token = target[:, 0]  # (batch)
        
        for t in range(1, target_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = target[:, t] if teacher_force else top1
        
        return outputs


# ============================================================
# 7. PREPARATION DES DONNEES POUR L'ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("7. PREPARATION DES TENSEURS")
print("=" * 60)

def prepare_data(pairs, fra_to_idx, eng_to_idx, max_len=15):
    source_seqs = []
    target_seqs = []
    
    for fra, eng in pairs:
        # Tokeniser et ajouter <SOS> et <EOS>
        fra_tokens = ['<SOS>'] + fra.split() + ['<EOS>']
        eng_tokens = ['<SOS>'] + eng.split() + ['<EOS>']
        
        # Convertir en indices
        fra_indices = [fra_to_idx.get(w, fra_to_idx['<UNK>']) for w in fra_tokens]
        eng_indices = [eng_to_idx.get(w, eng_to_idx['<UNK>']) for w in eng_tokens]
        
        # Padding
        fra_indices += [fra_to_idx['<PAD>']] * (max_len - len(fra_indices))
        eng_indices += [eng_to_idx['<PAD>']] * (max_len - len(eng_indices))
        
        source_seqs.append(fra_indices[:max_len])
        target_seqs.append(eng_indices[:max_len])
    
    return torch.LongTensor(source_seqs), torch.LongTensor(target_seqs)

source_data, target_data = prepare_data(pairs, fra_to_idx, eng_to_idx)

print(f"Source shape: {source_data.shape}")
print(f"Target shape: {target_data.shape}")


# ============================================================
# 8. ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("8. ENTRAINEMENT SEQ2SEQ")
print("=" * 60)

EMBED_SIZE = 64
HIDDEN_SIZE = 128

encoder = Encoder(len(fra_words), EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(len(eng_words), EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=eng_to_idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

N_EPOCHS = 1000

for epoch in range(N_EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    output = model(source_data, target_data)
    
    # Reshape pour CrossEntropyLoss
    output = output[:, 1:].reshape(-1, output.shape[2])
    target = target_data[:, 1:].reshape(-1)
    
    loss = criterion(output, target)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {loss.item():.4f}")

print("\nEntrainement termine!")


# ============================================================
# 9. TRADUCTION (INFERENCE)
# ============================================================

print("\n" + "=" * 60)
print("9. TEST DE TRADUCTION")
print("=" * 60)

def translate(model, sentence, fra_to_idx, idx_to_eng, max_len=15):
    model.eval()
    
    # Tokeniser
    tokens = ['<SOS>'] + sentence.split() + ['<EOS>']
    indices = [fra_to_idx.get(w, fra_to_idx['<UNK>']) for w in tokens]
    indices += [fra_to_idx['<PAD>']] * (max_len - len(indices))
    source = torch.LongTensor([indices[:max_len]])
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(source)
        
        # Commencer avec <SOS>
        input_token = torch.LongTensor([eng_to_idx['<SOS>']])
        
        translated = []
        attentions = []
        
        for _ in range(max_len):
            output, hidden, cell, attention = model.decoder(input_token, hidden, cell, encoder_outputs)
            
            # Mot predit
            top1 = output.argmax(1).item()
            
            if top1 == eng_to_idx['<EOS>']:
                break
            
            translated.append(idx_to_eng[top1])
            attentions.append(attention.squeeze().numpy())
            
            input_token = torch.LongTensor([top1])
    
    return ' '.join(translated), attentions


# Tester
test_sentences = [
    "je suis etudiant",
    "tu es professeur",
    "il est docteur",
    "nous sommes amis"
]

print("\nTraductions:")
for sentence in test_sentences:
    translation, attentions = translate(model, sentence, fra_to_idx, idx_to_eng)
    print(f"FR: '{sentence}' -> EN: '{translation}'")

# Sauvegarder
torch.save(model.state_dict(), '../models/seq2seq_attention.pth')
print("\nModele Seq2Seq sauvegarde!")


# ============================================================
# 10. VISUALISATION DE L'ATTENTION
# ============================================================

print("\n" + "=" * 60)
print("10. VISUALISATION DE L'ATTENTION")
print("=" * 60)

def plot_attention(sentence, translation, attentions):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Preparer les labels
    source_words = ['<SOS>'] + sentence.split() + ['<EOS>']
    target_words = translation.split() + ['<EOS>']
    
    # Convertir attentions en matrice
    attn_matrix = np.array(attentions[:len(target_words)])
    
    # Afficher
    im = ax.imshow(attn_matrix, cmap='hot', interpolation='nearest')
    
    ax.set_xticks(range(len(source_words)))
    ax.set_yticks(range(len(target_words)))
    ax.set_xticklabels(source_words, rotation=45)
    ax.set_yticklabels(target_words)
    
    ax.set_xlabel('Source (Francais)')
    ax.set_ylabel('Cible (Anglais)')
    ax.set_title('Matrice d\'Attention')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('../outputs/attention_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


# Visualiser pour une phrase
sentence = "je suis etudiant"
translation, attentions = translate(model, sentence, fra_to_idx, idx_to_eng)
plot_attention(sentence, translation, attentions)

print("\nMatrice d'attention sauvegardee!")
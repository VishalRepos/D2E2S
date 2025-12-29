"""
Data augmentation techniques for text
"""
import random
import numpy as np

class TextAugmenter:
    """Simple text augmentation without external dependencies"""
    
    def __init__(self):
        # Common synonym mappings for sentiment analysis
        self.synonyms = {
            'good': ['great', 'excellent', 'nice', 'fine', 'wonderful'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'great': ['good', 'excellent', 'wonderful', 'fantastic'],
            'excellent': ['great', 'wonderful', 'outstanding', 'superb'],
            'nice': ['good', 'pleasant', 'lovely', 'fine'],
            'poor': ['bad', 'terrible', 'inferior'],
            'love': ['like', 'enjoy', 'adore'],
            'hate': ['dislike', 'detest'],
            'delicious': ['tasty', 'yummy', 'flavorful'],
            'terrible': ['awful', 'horrible', 'dreadful', 'bad'],
            'amazing': ['wonderful', 'fantastic', 'incredible'],
            'perfect': ['excellent', 'ideal', 'flawless'],
            'worst': ['terrible', 'awful', 'poorest'],
            'best': ['finest', 'greatest', 'top'],
        }
    
    def synonym_replacement(self, text, n=2):
        """Replace n words with their synonyms"""
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word.lower() for word in words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            if random_word in self.synonyms:
                synonym = random.choice(self.synonyms[random_word])
                new_words = [synonym if word.lower() == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        
        return ' '.join(new_words)
    
    def random_deletion(self, text, p=0.1):
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_swap(self, text, n=2):
        """Randomly swap two words n times"""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def augment(self, text, num_aug=1):
        """Apply random augmentation techniques"""
        augmented_texts = []
        
        for _ in range(num_aug):
            aug_type = random.choice(['synonym', 'deletion', 'swap', 'none'])
            
            if aug_type == 'synonym':
                aug_text = self.synonym_replacement(text, n=random.randint(1, 3))
            elif aug_type == 'deletion':
                aug_text = self.random_deletion(text, p=0.1)
            elif aug_type == 'swap':
                aug_text = self.random_swap(text, n=random.randint(1, 2))
            else:
                aug_text = text
            
            augmented_texts.append(aug_text)
        
        return augmented_texts

import torch
import clip
from PIL import Image
import pandas as pd
import random
from datasets import load_dataset
from pydub import AudioSegment
from pydub.playback import play


class AudioGen:
    def __init__(self, image_dataset_name='mdroth/landscapes', audio_metadata_path='ESC-50/meta/esc50.csv'):
        """
        Initialize the AudioGen class.
        
        Args:
            image_dataset_name (str): Name of the image dataset to load.
            audio_metadata_path (str): Path to the audio metadata CSV file.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load datasets during initialization
        self.image_dataset = load_dataset(image_dataset_name)
        self.audio_metadata = pd.read_csv(audio_metadata_path)
        
        print("Datasets loaded successfully.")

        # Establish only the relevant audio keywords from the dataset
        self.keywords = [
            "rain", "sea_waves", "thunderstorm", "wind",
            "cat", "chirping_birds", "cow", "crickets", "crow", "dog", "frog", "hen",
            "insects", "pig", "rooster", "sheep"
        ]

        # Tokenize audio keywords
        self.text_tokens = clip.tokenize(self.keywords).to(self.device)

        # Get only the audio metadata for the relevant keywords
        self.filtered_audio_metadata = self.audio_metadata[self.audio_metadata['category'].isin(self.keywords)]

        # Get the file names for those audio keywords
        self.category_to_filenames = self.filtered_audio_metadata.groupby('category')['filename'].apply(list).to_dict()

    def select_random_image(self):
        """
        Select a random image from the loaded dataset and preprocess it.
        
        Returns:
            PIL Image: Random image selection from the dataset.
        """
        # Randomly select an image from the dataset
        image_data = random.choice(self.image_dataset['train'])
        image = image_data['image']
        return image

    def generate_white_noise(self, image=None):
        """
        Generate and return a white noise audio segment for a given image.
        
        Returns:
            pydub.AudioSegment: White noise audio segment.
        """
        if not image:
            image = self.select_random_image()
            
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(self.text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze()
        top_keywords_idx = similarities.topk(3).indices.tolist()
        top_keywords = [self.keywords[i] for i in top_keywords_idx]
        print(f"Top keywords: {top_keywords}")

        # Print each keyword with its similarity score
        keyword_similarities = list(zip(self.keywords, similarities))
        keyword_similarities.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity in descending order

        print("Keyword Similarities:")
        for keyword, similarity in keyword_similarities:
            print(f"{keyword}: {similarity:.4f}")

        # Overlay sounds from relevant keywords
        sounds = []
        for key_word in top_keywords:
            # Select a random audio file for that keyword for output diversity
            sound = AudioSegment.from_wav(f'ESC-50/audio/{random.choice(self.category_to_filenames[key_word])}')

            # Reduce the noise of animal sounds
            animals = ['dog', 'chirping_birds', 'crow', 'sheep', 'frog', 'cow', 'insects', 'hen', 'pig', 'rooster', 'cat', 'crickets']
            if keyword in animals:
                sound -= 20
                
            sounds.append(sound)

        base_sound = sounds[0]
        for next_sound in sounds[1:]:
            base_sound = base_sound.overlay(next_sound)
        
        print("White noise audio segment generated.")
        return base_sound

    def get_metrics(self):
        """
        Compute the average cosine similarity of each keyword across all images
        and count the number of times each keyword appeared in 1st, 2nd, and 3rd positions.
        
        Returns:
            pd.DataFrame: A dataframe with average cosine similarity and counts of 1st, 2nd, and 3rd positions per keyword.
        """
        total_similarities = {keyword: 0 for keyword in self.keywords}
        count_1st = {keyword: 0 for keyword in self.keywords}
        count_2nd = {keyword: 0 for keyword in self.keywords}
        count_3rd = {keyword: 0 for keyword in self.keywords}

        # Iterate over all images in the train split
        for image_data in self.image_dataset["train"]:
            image = image_data["image"]
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(self.text_tokens)

            # Normalize image and text features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarities
            similarities = (image_features @ text_features.T).squeeze()
            top_keywords_idx = similarities.topk(3).indices.tolist()
            top_keywords = [self.keywords[i] for i in top_keywords_idx]

            # Aggregate cosine similarity for each keyword
            for keyword in self.keywords:
                total_similarities[keyword] += similarities[self.keywords.index(keyword)].item()

            # Count positions
            if len(top_keywords) > 0:
                count_1st[top_keywords[0]] += 1
            if len(top_keywords) > 1:
                count_2nd[top_keywords[1]] += 1
            if len(top_keywords) > 2:
                count_3rd[top_keywords[2]] += 1

        # Normalize similarity sums to compute averages
        num_images = len(self.image_dataset["train"])
        avg_similarities = {keyword: total_similarities[keyword] / num_images for keyword in self.keywords}

        # Convert to DataFrame
        data = {
            "Keyword": self.keywords,
            "Average_Cosine_Similarity": [avg_similarities[keyword] for keyword in self.keywords],
            "1st_Position_Count": [count_1st[keyword] for keyword in self.keywords],
            "2nd_Position_Count": [count_2nd[keyword] for keyword in self.keywords],
            "3rd_Position_Count": [count_3rd[keyword] for keyword in self.keywords]
        }

        df = pd.DataFrame(data)
        df = df.sort_values(by='Average_Cosine_Similarity', ascending=False).reset_index(drop=True)

        print("Metrics computed and returned as DataFrame.")
        return df
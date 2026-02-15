import random
import pandas as pd
from typing import List, Tuple

class DataCollector:    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Sports vocabulary and patterns
        self.sports_keywords = [
            'goal', 'score', 'team', 'player', 'match', 'game', 'championship', 
            'victory', 'defeat', 'coach', 'tournament', 'league', 'season',
            'basketball', 'football', 'soccer', 'tennis', 'cricket', 'baseball',
            'referee', 'penalty', 'stadium', 'fans', 'trophy', 'medal', 'athletic',
            'training', 'competition', 'opponent', 'defense', 'offense', 'strategy'
        ]
        
        # Politics vocabulary and patterns
        self.politics_keywords = [
            'government', 'election', 'policy', 'minister', 'president', 'parliament',
            'legislation', 'democracy', 'vote', 'campaign', 'senator', 'congress',
            'bill', 'law', 'political', 'party', 'cabinet', 'reform', 'budget',
            'treaty', 'diplomat', 'constitution', 'governor', 'mayor', 'council',
            'committee', 'amendment', 'veto', 'administration', 'debate', 'opposition'
        ]
        
        # Sports templates
        self.sports_templates = [
            "The {team} won the {sport} {event} with a {score} victory over their rivals.",
            "{Player} scored {number} {action} in yesterday's crucial {sport} match.",
            "The {sport} season opens next week with high expectations for the defending champions.",
            "Coach announced the starting lineup for tomorrow's championship game.",
            "Fans celebrated as their team secured a spot in the playoffs with a dominant performance.",
            "{Team} defeated {opponent} in overtime to advance to the finals.",
            "The referee's controversial decision changed the outcome of the match.",
            "Record-breaking attendance at the stadium as {team} faced {opponent}.",
            "{Player} signed a new contract extension worth millions after stellar performance.",
            "The tournament bracket was announced with several exciting matchups ahead.",
            "Training camp begins as teams prepare for the upcoming competitive season.",
            "Injury concerns as star {player} misses practice before crucial game.",
            "The {sport} league announced new rules to improve player safety.",
            "Spectacular {action} in the final minutes secured the victory.",
            "Fans were disappointed as {team} suffered their third consecutive loss.",
            "The coach praised the team's defensive strategy after the win.",
            "{Player} broke the league record with an outstanding performance.",
            "The stadium erupted as the game-winning {action} was scored.",
            "Team management announced major roster changes for next season.",
            "The rivalry game lived up to expectations with intense competition throughout."
        ]
        
        # Politics templates
        self.politics_templates = [
            "The {body} passed new {legislation} aimed at reforming the healthcare system.",
            "{Official} announced plans for comprehensive economic policy changes.",
            "The election campaign enters its final week with candidates making last appeals to voters.",
            "Parliament debated the controversial bill for hours before reaching a decision.",
            "{Government} unveiled the annual budget with increased spending on infrastructure.",
            "Diplomatic relations improved after the treaty was signed by both nations.",
            "The opposition party criticized the administration's handling of the crisis.",
            "Voters will decide on the constitutional amendment in next month's referendum.",
            "{Official} met with international leaders to discuss climate policy.",
            "The committee released its findings after a thorough investigation.",
            "New legislation aims to address growing concerns about data privacy.",
            "The governor vetoed the controversial bill citing constitutional concerns.",
            "Political analysts predict a close race in the upcoming gubernatorial election.",
            "The administration announced plans to reform the immigration system.",
            "Parliament approved the treaty after weeks of intense debate.",
            "Local council voted to increase funding for public education.",
            "{Official} announced candidacy for the upcoming presidential election.",
            "The bill passed with bipartisan support after months of negotiations.",
            "Government officials met to discuss the proposed tax reform legislation.",
            "The debate focused on key policy differences between the candidates."
        ]
        
    def generate_sports_text(self) -> str:
        """Generate a realistic sports article snippet"""
        template = random.choice(self.sports_templates)
        
        # Replace placeholders
        replacements = {
            '{team}': random.choice(['Lakers', 'United', 'Eagles', 'Yankees', 'Warriors', 'Tigers']),
            '{opponent}': random.choice(['Celtics', 'City', 'Patriots', 'Red Sox', 'Heat', 'Lions']),
            '{sport}': random.choice(['basketball', 'football', 'soccer', 'baseball', 'hockey']),
            '{event}': random.choice(['championship', 'tournament', 'finals', 'playoff', 'cup']),
            '{score}': random.choice(['decisive', 'narrow', 'commanding', 'dramatic', 'convincing']),
            '{player}': random.choice(['Johnson', 'Smith', 'Williams', 'Martinez', 'Davis']),
            '{Player}': random.choice(['Johnson', 'Smith', 'Williams', 'Martinez', 'Davis']),
            '{number}': str(random.randint(1, 5)),
            '{action}': random.choice(['goals', 'points', 'touchdowns', 'runs', 'baskets'])
        }
        
        text = template
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        # Add some sports keywords
        keywords = random.sample(self.sports_keywords, k=random.randint(2, 4))
        text += " " + " ".join([f"The {kw} was impressive." for kw in keywords[:2]])
        
        return text
    
    def generate_politics_text(self) -> str:
        """Generate a realistic politics article snippet"""
        template = random.choice(self.politics_templates)
        
        # Replace placeholders
        replacements = {
            '{body}': random.choice(['Senate', 'House', 'Parliament', 'Congress', 'Assembly']),
            '{legislation}': random.choice(['legislation', 'bill', 'policy', 'reform', 'act']),
            '{Official}': random.choice(['Senator Thompson', 'Governor Smith', 'President Johnson', 'Minister Brown']),
            '{official}': random.choice(['senator', 'governor', 'president', 'minister']),
            '{Government}': random.choice(['The administration', 'The government', 'Officials', 'The cabinet'])
        }
        
        text = template
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        # Add some politics keywords
        keywords = random.sample(self.politics_keywords, k=random.randint(2, 4))
        text += " " + " ".join([f"The {kw} is significant." for kw in keywords[:2]])
        
        return text
    
    def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:

        data = []
        
        # Generate equal number of sports and politics samples
        for _ in range(n_samples // 2):
            # Sports sample
            data.append({
                'text': self.generate_sports_text(),
                'label': 'sports'
            })
            
            # Politics sample
            data.append({
                'text': self.generate_politics_text(),
                'label': 'politics'
            })
        
        df = pd.DataFrame(data)
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df

def main():
    """Generate and save the dataset"""
    collector = DataCollector(seed=42)
    
    # Generate datasets
    print("Generating training dataset...")
    train_df = collector.generate_dataset(n_samples=1200)
    
    print("Generating test dataset...")
    test_df = collector.generate_dataset(n_samples=400)
    
    # Save to CSV
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"\nLabel distribution (train):")
    print(train_df['label'].value_counts())

if __name__ == "__main__":
    main()

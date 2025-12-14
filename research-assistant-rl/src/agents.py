from collections import defaultdict
import random
import math
from typing import Dict, List, Tuple


class QueryStrategyAgent:
    """
    Q-Learning agent with intrinsic motivation.
    Learns optimal query formulation strategies.
    
    State: (topic, difficulty)
    Actions: (strategy, source) pairs
    """
    
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.strategies = ['broad', 'specific', 'narrow']
        self.sources = ['openalex', 'arxiv']
        
        self.episode_count = 0
        self.rewards_history = []
        
        # Intrinsic motivation: Track state-action novelty
        self.visit_counts = defaultdict(int)
        self.intrinsic_bonus = 0.5
    
    def get_state(self, task) -> Tuple:
        """Extract state from current task"""
        return (task.topic, task.difficulty)
    
    def get_intrinsic_reward(self, state, action):
        """
        Intrinsic motivation: Bonus for exploring novel state-actions.
        Encourages exploration early, diminishes over time.
        """
        key = (state, action)
        count = self.visit_counts[key]
        return self.intrinsic_bonus / (1 + count)
    
    def choose_action(self, state) -> Tuple[str, str]:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            strategy = random.choice(self.strategies)
            source = random.choice(self.sources)
        else:
            best_value = float('-inf')
            best_action = (self.strategies[0], self.sources[0])
            
            for strategy in self.strategies:
                for source in self.sources:
                    action = (strategy, source)
                    q_value = self.q_table[(state, action)]
                    
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
            
            strategy, source = best_action
        
        return strategy, source
    
    def update(self, state, action, reward, next_state):
        """
        Q-learning update with intrinsic motivation.
        Adds curiosity bonus to encourage exploration.
        """
        # Track visits for intrinsic motivation
        self.visit_counts[(state, action)] += 1
        
        # Add intrinsic reward (curiosity bonus)
        intrinsic = self.get_intrinsic_reward(state, action)
        total_reward = reward + intrinsic
        
        current_q = self.q_table[(state, action)]
        
        max_next_q = float('-inf')
        for strategy in self.strategies:
            for source in self.sources:
                next_action = (strategy, source)
                q = self.q_table[(next_state, next_action)]
                max_next_q = max(max_next_q, q)
        
        if max_next_q == float('-inf'):
            max_next_q = 0.0
        
        # Update with intrinsic + extrinsic reward
        new_q = current_q + self.alpha * (total_reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
        
        self.rewards_history.append(reward)
        self.episode_count += 1
    
    def get_policy(self) -> Dict:
        """
        Extract learned policy for analysis.
        Converts tuple keys to strings for JSON serialization.
        """
        policy = {}
        states = set(k[0] for k in self.q_table.keys())
        
        for state in states:
            best_value = float('-inf')
            best_action = None
            
            for strategy in self.strategies:
                for source in self.sources:
                    action = (strategy, source)
                    q = self.q_table[(state, action)]
                    
                    if q > best_value:
                        best_value = q
                        best_action = action
            
            if best_action:
                # Convert tuple keys to strings for JSON (FIX APPLIED)
                state_str = f"{state[0]}_{state[1]}"
                policy[state_str] = {
                    'topic': state[0],
                    'difficulty': state[1],
                    'strategy': best_action[0],
                    'source': best_action[1],
                    'q_value': best_value
                }
        
        return policy


class SourceSelectorAgent:
    """
    UCB (Upper Confidence Bound) bandit for source selection.
    Learns which database is better for each topic.
    """
    
    def __init__(self, exploration_param=2.0):
        self.c = exploration_param
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.total_pulls = defaultdict(int)
        self.sources = ['openalex', 'arxiv']
    
    def choose_source(self, topic: str) -> str:
        """UCB algorithm: balance exploration vs exploitation"""
        for source in self.sources:
            if self.counts[(topic, source)] == 0:
                return source
        
        ucb_scores = {}
        total = self.total_pulls[topic]
        
        for source in self.sources:
            key = (topic, source)
            mean_reward = self.values[key]
            count = self.counts[key]
            exploration = self.c * math.sqrt(math.log(total) / count)
            ucb_scores[source] = mean_reward + exploration
        
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, topic: str, source: str, reward: float):
        """Update statistics after observing reward"""
        key = (topic, source)
        self.counts[key] += 1
        self.total_pulls[topic] += 1
        
        n = self.counts[key]
        old_mean = self.values[key]
        self.values[key] = old_mean + (reward - old_mean) / n
    
    def get_preferences(self) -> Dict:
        """Get learned source preferences by topic"""
        preferences = {}
        topics = set(k[0] for k in self.counts.keys())
        
        for topic in topics:
            source_stats = {}
            for source in self.sources:
                key = (topic, source)
                source_stats[source] = {
                    'avg_reward': self.values[key],
                    'pull_count': self.counts[key]
                }
            
            best = max(source_stats, key=lambda s: source_stats[s]['avg_reward'])
            
            preferences[topic] = {
                'best_source': best,
                'sources': source_stats
            }
        
        return preferences

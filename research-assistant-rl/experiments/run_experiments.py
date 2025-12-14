import os
import sys
import json
import time
from datetime import datetime
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import ResearchEnvironment
from src.coordinator import EnhancedCoordinator


def run_random_baseline(env, num_episodes=30):
    """Baseline: random strategy and source selection"""
    print("\n" + "="*60)
    print("BASELINE: Random Search Strategy")
    print("="*60)
    
    results = []
    
    for i in range(num_episodes):
        task = env.generate_task()
        strategy = random.choice(['broad', 'specific', 'narrow'])
        
        # Prefer arXiv to reduce S2 rate limiting
        source = random.choices(['openalex', 'arxiv'], weights=[0.5, 0.5])[0]
        
        papers, cost = env.execute_search(strategy, source)
        reward = env.get_reward(papers, cost)
        relevance = task.evaluate_results(papers)
        
        results.append({
            'episode': i,
            'reward': reward,
            'relevance': relevance,
            'cost': cost,
            'papers_count': len(papers),
            'strategy': strategy,
            'source': source
        })
        
        if (i + 1) % 10 == 0:
            recent = results[-10:]
            avg_reward = sum(r['reward'] for r in recent) / len(recent)
            avg_relevance = sum(r['relevance'] for r in recent) / len(recent)
            print(f"Episode {i+1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Relevance: {avg_relevance:.2f}")
    
    return results


def run_rl_training(env, num_episodes=200):
    """Train multi-agent RL system with enhanced coordinator"""
    print("\n" + "="*60)
    print("RL TRAINING: Multi-Agent System")
    print("="*60)
    
    coordinator = EnhancedCoordinator()
    results = []
    
    for i in range(num_episodes):
        task = env.generate_task()
        papers, reward, metadata = coordinator.research_with_fallback(env, task)
        
        results.append({
            'episode': i,
            'reward': reward,
            'relevance': metadata['relevance'],
            'cost': metadata['cost'],
            'papers_count': metadata['papers_count'],
            'strategy': metadata['strategy'],
            'source': metadata['source'],
            'synthesis_quality': metadata.get('synthesis_quality', 0),
            'allocation': metadata.get('allocation', 'both'),
            'fallback_used': metadata.get('fallback_used', False)
        })
        
        if (i + 1) % 25 == 0:
            recent = results[-25:]
            avg_reward = sum(r['reward'] for r in recent) / 25
            avg_relevance = sum(r['relevance'] for r in recent) / 25
            print(f"Episode {i+1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Relevance: {avg_relevance:.2f}")
    
    return results, coordinator


def save_results(baseline_results, rl_results, coordinator, elapsed_time):
    """Save all experimental data to JSON"""
    os.makedirs('results', exist_ok=True)
    
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_time,
            'baseline_episodes': len(baseline_results),
            'rl_episodes': len(rl_results)
        },
        'baseline_results': baseline_results,
        'rl_results': rl_results,
        'learned_policy': coordinator.q_agent.get_policy(),
        'source_preferences': coordinator.ucb_agent.get_preferences(),
        'task_allocation': coordinator.task_allocation_history,
        'synthesis_improvement': coordinator.synthesizer.get_improvement()
    }
    
    filepath = 'results/experiment_data.json'
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def print_summary(baseline_results, rl_results, coordinator):
    """Print summary of results"""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    baseline_reward = sum(r['reward'] for r in baseline_results) / len(baseline_results)
    baseline_relevance = sum(r['relevance'] for r in baseline_results) / len(baseline_results)
    
    rl_final = rl_results[-50:]
    rl_reward = sum(r['reward'] for r in rl_final) / len(rl_final)
    rl_relevance = sum(r['relevance'] for r in rl_final) / len(rl_final)
    
    improvement = ((rl_reward - baseline_reward) / abs(baseline_reward)) * 100
    
    print(f"\nBaseline (Random):")
    print(f"  Avg Reward: {baseline_reward:.2f}")
    print(f"  Avg Relevance: {baseline_relevance:.2f}")
    
    print(f"\nRL Agent (After Training):")
    print(f"  Avg Reward: {rl_reward:.2f}")
    print(f"  Avg Relevance: {rl_relevance:.2f}")
    
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Task allocation stats
    print(f"\nTask Allocation:")
    total_tasks = sum(coordinator.task_allocation_history.values())
    for agent, count in coordinator.task_allocation_history.items():
        pct = (count / total_tasks) * 100 if total_tasks > 0 else 0
        print(f"  {agent}: {count} ({pct:.1f}%)")
    
    # Synthesis improvement
    synthesis_improvement = coordinator.synthesizer.get_improvement()
    print(f"\nSynthesis Quality Improvement: {synthesis_improvement:+.3f}")
    
    print("\n" + "="*60)


def main():
    """Main experiment pipeline"""
    print("\n" + "="*60)
    print("ADAPTIVE RESEARCH ASSISTANT")
    print("Reinforcement Learning Experiment")
    print("="*60)
    
    print("\nInitializing research environment...")
    env = ResearchEnvironment()
    print("✓ Environment ready")
    print(f"✓ API toolkit initialized")
    print(f"✓ Task templates loaded: {len(env.task_templates)} topics")
    
    start_time = time.time()
    
    baseline_results = run_random_baseline(env)
    rl_results, coordinator = run_rl_training(env)
    
    elapsed_time = time.time() - start_time
    
    save_results(baseline_results, rl_results, coordinator, elapsed_time)
    print_summary(baseline_results, rl_results, coordinator)
    
    print(f"\nTotal experiment time: {elapsed_time:.1f} seconds")
    print(f"API usage: {env.toolkit.get_stats()}")
    
    print("\n✓ Experiments complete!")
    print("  Run analysis scripts for full validation")


if __name__ == '__main__':
    main()

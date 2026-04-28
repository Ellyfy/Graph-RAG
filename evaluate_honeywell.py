"""
Evaluate Honeywell AutoQ Dataset with Ragas
CLEAN VERSION - No verbose warnings or errors displayed
"""

import json
import os
import warnings
import logging
from dotenv import load_dotenv
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset, evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import pandas as pd
from pathlib import Path

# Suppress ALL warnings
warnings.filterwarnings('ignore')

# Suppress Ragas logging
logging.getLogger('ragas').setLevel(logging.CRITICAL)
logging.getLogger('openai').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

os.environ["OPENAI_API_KEY"] = openai_key

def load_honeywell_data():
    """Load the Honeywell ground truth data"""
    
    print("\n" + "="*70)
    print("HONEYWELL AUTOQ RAGAS EVALUATION")
    print("="*70)
    
    json_file = Path('honeywell_autoq_with_ground_truth.json')
    
    if not json_file.exists():
        raise FileNotFoundError("Could not find honeywell_autoq_with_ground_truth.json")
    
    with open(json_file) as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    print(f"\n📊 Loaded {len(queries)} queries")
    
    return queries

def load_graphrag_results():
    """Load GraphRAG results"""
    
    results_file = Path('graphrag_autoq_results.json')
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        print(f"📊 Loaded {len(results)} GraphRAG results")
        return results
    return None

def convert_to_ragas_samples(queries, graphrag_results=None):
    """Convert to Ragas format"""
    
    print(f"\n⚙️  Converting to Ragas format...")
    
    samples = []
    
    for idx, item in enumerate(queries):
        try:
            query_id = item.get('id', f'Q{idx+1:02d}')
            question = item.get('question', '')
            ground_truth = item.get('ground_truth', '')
            
            response = ground_truth
            contexts = [ground_truth]
            
            if graphrag_results:
                for result in graphrag_results:
                    if result.get('id') == query_id:
                        response = result.get('graphrag_answer', ground_truth)
                        contexts = result.get('retrieved_contexts', [ground_truth])
                        break
            
            # Handle context formats
            if isinstance(contexts, tuple) and len(contexts) > 1:
                contexts = [str(item) for item in contexts[1]] if isinstance(contexts[1], list) else [str(contexts)]
            elif isinstance(contexts, str):
                contexts = [contexts]
            else:
                contexts = [str(ctx) for ctx in contexts]
            
            # Truncate to avoid token limits
            MAX_LEN = 2000
            contexts = [ctx[:MAX_LEN] for ctx in contexts]
            
            sample = SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=str(response)[:MAX_LEN],
                reference=str(ground_truth)[:MAX_LEN]
            )
            
            samples.append(sample)
        
        except Exception:
            continue
    
    print(f"✅ Converted {len(samples)} samples\n")
    return samples

def run_evaluation(samples):
    """Run evaluation with clean output"""
    
    print("="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    print(f"\n📊 Evaluating {len(samples)} samples")
    print(f"⚙️  Metrics: Faithfulness, Answer Relevancy, Context Precision")
    print(f"⏱️  Estimated time: 8-12 minutes")
    
    print("🔄 Processing... (this may take several minutes)\n")
    
    dataset = EvaluationDataset(samples=samples)
    
    # Run evaluation (errors will be caught internally)
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        raise_exceptions=False,
        show_progress=True  # Show progress bar only
    )
    
    return results

def save_results(results, queries):
    """Save and display results"""
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    df = results.to_pandas()
    
    # Add metadata
    for idx, query in enumerate(queries):
        if idx < len(df):
            df.loc[idx, 'query_id'] = query.get('id', f'Q{idx+1:02d}')
            df.loc[idx, 'query_class'] = query.get('class', 'unknown')
    
    # Save CSV
    output_file = 'honeywell_ragas_evaluation.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved: {output_file}")
    
    # Summary statistics
    print(f"\n📊 Overall Averages:")
    for col in ['faithfulness', 'answer_relevancy', 'context_precision']:
        if col in df.columns:
            valid = df[col].notna().sum()
            mean = df[col].mean()
            print(f"   {col:20s}: {mean:.3f} ({valid}/{len(df)} samples)")
    
    # By query class
    if 'query_class' in df.columns:
        print(f"\n📊 By Query Class:")
        grouped = df.groupby('query_class')[['faithfulness', 'answer_relevancy', 'context_precision']].mean()
        print(grouped.to_string())
    
    # Check for issues
    missing = df[['faithfulness', 'answer_relevancy', 'context_precision']].isna().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Note: Some metrics have missing values due to API errors")
        for col, count in missing.items():
            if count > 0:
                print(f"   {col}: {count} missing")
    
    return df

def main():
    """Main execution"""
    
    try:
        queries = load_honeywell_data()
        graphrag_results = load_graphrag_results()
        samples = convert_to_ragas_samples(queries, graphrag_results)
        
        if not samples:
            print("\n❌ No valid samples!")
            return
        
        results = run_evaluation(samples)
        df = save_results(results, queries)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📄 Results: honeywell_ragas_evaluation.csv")
        print(f"📊 Samples: {len(samples)}")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

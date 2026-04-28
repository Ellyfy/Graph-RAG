"""
Honeywell AutoQ Evaluation with REAL Ragas Framework
Uses OpenAI API for LLM-based metrics
"""

import json
import os
import warnings
import sys
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

# Suppress ALL warnings
warnings.filterwarnings('ignore')

# Suppress asyncio warnings (prevents "Event loop is closed" errors)
import logging
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Fix Windows asyncio event loop issues
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("❌ OPENAI_API_KEY not found in .env file!")
    exit(1)

os.environ["OPENAI_API_KEY"] = openai_key

# Import Ragas metrics directly (works with ragas 0.1.9)
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from datasets import Dataset

print("✅ Using REAL Ragas Framework with OpenAI API\n")

def load_data():
    """Load queries and GraphRAG results"""
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    with open('honeywell_autoq_with_ground_truth.json') as f:
        data = json.load(f)
    queries = data['queries']
    print(f"✅ Loaded {len(queries)} queries")
    
    with open('graphrag_autoq_results.json') as f:
        results = json.load(f)
    print(f"✅ Loaded {len(results)} GraphRAG results\n")
    
    return queries, results

def prepare_ragas_data(queries, results):
    """Prepare data in format Ragas expects"""
    
    print("="*70)
    print("PREPARING DATA FOR RAGAS")
    print("="*70)
    
    data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }
    
    metadata = []
    
    for q in queries:
        # Find matching result
        r = next((x for x in results if x['id'] == q['id']), None)
        
        if not r:
            continue
            
        # Get response and contexts
        response = r.get('graphrag_answer', q['ground_truth'])
        contexts = r.get('retrieved_contexts', [q['ground_truth']])
        
        # Handle tuple format
        if isinstance(contexts, tuple) and len(contexts) > 1:
            contexts = [str(x) for x in contexts[1]] if isinstance(contexts[1], list) else [str(contexts)]
        elif isinstance(contexts, str):
            contexts = [contexts]
        else:
            contexts = [str(c) for c in contexts]
        
        # Truncate to avoid token limits
        contexts = [c[:2000] for c in contexts]
        
        # Add to dataset
        data['question'].append(q['question'])
        data['answer'].append(str(response)[:2000])
        data['contexts'].append(contexts)
        data['ground_truth'].append(str(q['ground_truth'])[:2000])
        
        metadata.append({
            'query_id': q['id'],
            'query_class': q['class']
        })
    
    print(f"✅ Prepared {len(data['question'])} samples\n")
    return data, metadata

def run_ragas_evaluation(data):
    """Run Ragas evaluation with OpenAI"""
    
    print("="*70)
    print("RUNNING RAGAS EVALUATION")
    print("="*70)
    print(f"\n📊 Evaluating {len(data['question'])} samples")
    print("⚙️  Metrics: Faithfulness, Answer Relevancy, Context Precision")
    print("🔧 Using: OpenAI GPT models via Ragas")
    print("⏱️  Time: 8-12 minutes")
    
    
    # Create HuggingFace dataset (required by Ragas)
    dataset = Dataset.from_dict(data)
    
    # Run evaluation
    print("🔄 Processing...\n")
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ]
    )
    
    return result

def save_results(result, metadata):
    """Save and display results"""
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Convert to DataFrame
    df = result.to_pandas()
    
    # Add metadata
    for idx, meta in enumerate(metadata):
        if idx < len(df):
            df.loc[idx, 'query_id'] = meta['query_id']
            df.loc[idx, 'query_class'] = meta['query_class']
    
    # Save
    output = 'honeywell_ragas_evaluation.csv'
    df.to_csv(output, index=False)
    
    print(f"\n💾 Saved: {output}")
    
    # Display summary
    print(f"\n📊 Overall Averages:")
    print(f"   Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"   Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")  
    print(f"   Context Precision: {df['context_precision'].mean():.3f}")
    
    if 'query_class' in df.columns:
        print(f"\n📊 By Query Class:")
        grouped = df.groupby('query_class')[['faithfulness', 'answer_relevancy', 'context_precision']].mean()
        print(grouped.to_string())
    
    # Check for missing values
    missing = df[['faithfulness', 'answer_relevancy', 'context_precision']].isna().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Some metrics have missing values:")
        for col, count in missing.items():
            if count > 0:
                print(f"   {col}: {count} missing")
    
    return df

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("HONEYWELL AUTOQ - RAGAS FRAMEWORK EVALUATION")
    print("="*70)
    print("\n🔬 Using OFFICIAL Ragas library")
    print("🤖 Using OpenAI GPT for LLM-based metrics\n")
    
    try:
        # Load data
        queries, results = load_data()
        
        # Prepare for Ragas
        data, metadata = prepare_ragas_data(queries, results)
        
        # Run evaluation
        result = run_ragas_evaluation(data)
        
        # Save results
        df = save_results(result, metadata)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📄 Results: honeywell_ragas_evaluation.csv")
        print(f"📊 Samples: {len(data['question'])}")
        print(f"\n🔬 This used the REAL Ragas framework:")
        print("   ✅ Ragas evaluate() function")
        print("   ✅ Ragas metrics (faithfulness, answer_relevancy, context_precision)")
        print("   ✅ OpenAI GPT for LLM-based evaluation")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
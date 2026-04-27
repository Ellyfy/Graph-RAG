"""
Evaluate Honeywell AutoQ Dataset - Simple Metrics
NO API NEEDED - Uses text similarity instead of LLM-based evaluation
"""

import json
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

def load_honeywell_data():
    """Load the Honeywell ground truth data"""
    
    print("="*70)
    print("Loading Honeywell AutoQ Dataset")
    print("="*70)
    
    json_file = Path('honeywell_autoq_with_ground_truth.json')
    
    if not json_file.exists():
        raise FileNotFoundError(
            "Could not find honeywell_autoq_with_ground_truth.json\n"
            "Make sure you're running from the Graph-RAG-main directory!"
        )
    
    print(f"\n✅ Found: {json_file}")
    with open(json_file) as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    
    print(f"\n📊 Dataset Information:")
    print(f"   Domain: {data.get('dataset', 'unknown')}")
    print(f"   Total queries: {data.get('total_queries', len(queries))}")
    print(f"   Source: {data.get('source_document', 'unknown')}")
    print(f"   Queries loaded: {len(queries)}")
    
    return queries

def load_graphrag_results():
    """Load GraphRAG results if available"""
    
    results_file = Path('graphrag_autoq_results.json')
    
    if results_file.exists():
        print(f"\n✅ Found GraphRAG results: {results_file}")
        with open(results_file) as f:
            results = json.load(f)
        
        if isinstance(results, dict):
            results = results.get('results', results.get('queries', []))
        
        print(f"   Loaded {len(results)} results")
        return results
    else:
        print(f"\n⚠️  No GraphRAG results found")
        return None

def text_similarity(text1, text2):
    """Calculate similarity between two texts (0-1)"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def calculate_faithfulness(response, ground_truth):
    """
    Simple faithfulness: How similar is response to ground truth
    Score: 0-1 (higher = more faithful)
    """
    return text_similarity(response, ground_truth)

def calculate_answer_relevancy(response, question):
    """
    Simple relevancy: Does response contain key terms from question
    Score: 0-1 (higher = more relevant)
    """
    # Extract key words from question (longer than 3 chars, not common words)
    common_words = {'what', 'when', 'where', 'which', 'how', 'the', 'is', 'are', 'for', 'and', 'or'}
    question_words = [w.lower() for w in question.split() if len(w) > 3 and w.lower() not in common_words]
    
    if not question_words:
        return 0.5
    
    # Count how many question keywords appear in response
    response_lower = response.lower()
    matches = sum(1 for word in question_words if word in response_lower)
    
    return min(matches / len(question_words), 1.0)

def calculate_context_precision(contexts, ground_truth):
    """
    Simple precision: How well do contexts match ground truth
    Score: 0-1 (higher = better context quality)
    """
    if not contexts:
        return 0.0
    
    # Check similarity of contexts to ground truth
    similarities = [text_similarity(ctx, ground_truth) for ctx in contexts]
    return max(similarities) if similarities else 0.0

def evaluate_sample(query_id, question, response, ground_truth, contexts, query_class):
    """Evaluate a single sample"""
    
    # Calculate metrics
    faithfulness = calculate_faithfulness(response, ground_truth)
    answer_relevancy = calculate_answer_relevancy(response, question)
    context_precision = calculate_context_precision(contexts, ground_truth)
    
    return {
        'query_id': query_id,
        'query_class': query_class,
        'question': question,
        'faithfulness': round(faithfulness, 3),
        'answer_relevancy': round(answer_relevancy, 3),
        'context_precision': round(context_precision, 3),
        'response_length': len(response),
        'gt_length': len(ground_truth)
    }

def run_evaluation(queries, graphrag_results):
    """Run evaluation on all samples"""
    
    print("\n" + "="*70)
    print("Running Simple Evaluation (No API Required)")
    print("="*70)
    
    print(f"\n📊 Evaluating {len(queries)} samples...")
    print("   Using text similarity metrics\n")
    
    results = []
    
    for idx, item in enumerate(queries, 1):
        query_id = item.get('id', f'Q{idx:02d}')
        query_class = item.get('class', 'unknown')
        question = item.get('question', '')
        ground_truth = item.get('ground_truth', '')
        
        # Find matching GraphRAG result
        response = None
        contexts = []
        
        if graphrag_results:
            for result in graphrag_results:
                if (result.get('id') == query_id or 
                    result.get('question') == question):
                    response = result.get('response', result.get('answer', ''))
                    contexts = result.get('contexts', result.get('retrieved_contexts', []))
                    break
        
        # If no GraphRAG result, use ground truth as response
        if not response:
            response = ground_truth
            contexts = [ground_truth]
        
        # Ensure contexts is a list
        if isinstance(contexts, str):
            contexts = [contexts]
        
        # Evaluate
        result = evaluate_sample(
            query_id, question, response, ground_truth, contexts, query_class
        )
        results.append(result)
        
        # Show progress
        if idx % 5 == 0 or idx <= 3:
            print(f"  [{idx}/{len(queries)}] {query_id}: "
                  f"F={result['faithfulness']:.3f}, "
                  f"R={result['answer_relevancy']:.3f}, "
                  f"P={result['context_precision']:.3f}")
    
    print(f"\n✅ Evaluation complete!")
    return results

def save_results(results):
    """Save and display results"""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate overall statistics
    print(f"\nOverall Averages:")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision: {df['context_precision'].mean():.3f}")
    
    # Group by query class
    print(f"\n📊 Results by Query Class:")
    grouped = df.groupby('query_class')[['faithfulness', 'answer_relevancy', 'context_precision']].mean()
    print(grouped.to_string())
    
    # Show top and bottom performers
    print(f"\n🏆 Top 5 by Faithfulness:")
    top = df.nlargest(5, 'faithfulness')[['query_id', 'faithfulness', 'answer_relevancy', 'context_precision']]
    print(top.to_string(index=False))
    
    print(f"\n⚠️  Bottom 5 by Faithfulness:")
    bottom = df.nsmallest(5, 'faithfulness')[['query_id', 'faithfulness', 'answer_relevancy', 'context_precision']]
    print(bottom.to_string(index=False))
    
    # Save to CSV
    output_file = 'honeywell_evaluation_simple.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved results to: {output_file}")
    
    return df

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("HONEYWELL AUTOQ SIMPLE EVALUATION")
    print("="*70)
    print("\nEvaluating Honeywell queries using text similarity metrics")
    print("(No API required - works offline!)\n")
    
    try:
        # Load data
        queries = load_honeywell_data()
        graphrag_results = load_graphrag_results()
        
        # Run evaluation
        results = run_evaluation(queries, graphrag_results)
        
        # Save and display
        df = save_results(results)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nTotal samples evaluated: {len(results)}")
        print(f"Results saved to: honeywell_evaluation_simple.csv")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
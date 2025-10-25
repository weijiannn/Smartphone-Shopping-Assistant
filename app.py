from flask import Flask, render_template, request, jsonify
from shopping_assistant import ShoppingCopilot
import time

app = Flask(__name__)
copilot = ShoppingCopilot()
# Warm up models once so first request is fast
try:
    _ = copilot.intent_classifier.predict("warm up")
    _ = copilot.sentiment_analyzer.analyze("warm up")
except Exception:
    pass

# Store rolling latencies to compute average
latencies = []
intent_cv_accuracy = None
try:
    # Compute cross-validated intent accuracy once at startup
    intent_cv_accuracy = round(copilot.intent_classifier.evaluate_kfold(k=5), 1)
except Exception:
    intent_cv_accuracy = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Start total request timer
    total_start = time.time()
    
    # Measure only the processing time of copilot.process_query
    process_start = time.time()
    response = copilot.process_query(user_message)
    process_end = time.time()
    
    process_latency = round(process_end - process_start, 3)
    
    # End total request timer (includes Flask handling)
    total_end = time.time()
    total_latency = round(total_end - total_start, 3)
    
    latencies.append(total_latency)
    avg_latency = round(sum(latencies) / len(latencies), 3)
    
    formatted_response = response['response'].replace('\n', '<br>')
    
    # Log latencies to console
    print(f"Processing latency: {process_latency}s | Total latency: {total_latency}s | Avg latency: {avg_latency}s")
    
    return jsonify({
        'response': formatted_response,
        'process_latency': process_latency,
        'total_latency': total_latency,
        'avg_latency': avg_latency,
        'intent_accuracy': intent_cv_accuracy
    })

# JSON APIs for direct integration
@app.route('/metrics', methods=['GET'])
def metrics():
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    return jsonify({
        'intent_accuracy': intent_cv_accuracy,
        'avg_latency': avg_latency
    })

@app.route('/api/phones', methods=['GET'])
def api_phones():
    try:
        df = copilot.db.df[['name','brand','price','review']].copy()
        rows = df.to_dict(orient='records')
        return jsonify({'phones': rows, 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json(force=True)
    q = data.get('q', '')
    t0 = time.time()
    res = copilot.search(q, copilot.get_or_create_user('api_search'))
    return jsonify({'response': res, 'latency': round(time.time()-t0, 3)})

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json(force=True)
    q = data.get('q', '')
    t0 = time.time()
    res = copilot.recommend(q, copilot.get_or_create_user('api_recommend'))
    return jsonify({'response': res, 'latency': round(time.time()-t0, 3)})

@app.route('/api/compare', methods=['POST'])
def api_compare():
    data = request.get_json(force=True)
    q = data.get('q', '')
    t0 = time.time()
    res = copilot.compare(q, copilot.get_or_create_user('api_compare'))
    return jsonify({'response': res, 'latency': round(time.time()-t0, 3)})

@app.route('/reset', methods=['GET'])
def reset():
    # Clear conversation context so new sessions don't inherit old filters
    try:
        copilot.reset_context()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Use production-like flags for lower latency
    app.run(debug=False, use_reloader=False, threaded=True)

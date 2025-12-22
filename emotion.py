from transformers import pipeline

def emotion_analyze(text):
    classifier = pipeline('sentiment-analysis', model='alsgyu/sentiment-analysis-fine-tuned-model', truncation=True, max_length=256)
    result = classifier(text)[0]

    if result['label'] == 'negative':
        emotion = "부정적"
    else:
        emotion = "긍정적"
        
    e_score = result['score']*100

    return emotion, round(e_score,1)
from bertopic import BERTopic

topic_model = BERTopic()

def detect_topics(text):
    topics, _ = topic_model.fit_transform([text])
    topic_info = topic_model.get_topic(topics[0])
    if topic_info:
        return [word for word, _ in topic_info]
    return []

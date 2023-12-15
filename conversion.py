def convert_labels(emotion, agents):
    filtered_emotion = [emotion[i] for i in range(len(agents)) if agents[i] == 1] #when human is speaking 
    mid = len(filtered_emotion) // 2
    print(mid)
    buffer = 0.2
    first_half = filtered_emotion[:int(mid * (1 - buffer))]
    second_half = filtered_emotion[int(mid * (1 + buffer)):]
    
    def most_common(emotions):
        # Filter out 'Neutral' from the list
        emotions_ex_neutral = [e for e in emotions if e != 'Neutral']
        # If the list is empty after filtering (all were 'Neutral'), return 'Neutral' as default
        if not emotions_ex_neutral: 
            return 'Neutral'
        return max(emotions_ex_neutral, key = emotions_ex_neutral.count)

    # For both halves, output the emotion label with the most overlap, excluding 'Neutral'
    first_emotion = most_common(first_half)
    second_emotion = most_common(second_half)
    print(first_emotion, second_emotion)
    return [first_emotion, second_emotion]

#Test case: 
convert_labels(['Neutral', 'Anger', 'Neutral', 'Anger', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Happy', 'Neutral', 'Happy', 'Neutral'],[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
convert_labels(['Neutral', 'Surprise', 'Neutral', 'Surprise', 'Neutral', 'Happiness', 'Neutral', 'Happiness', 'Neutral', 'Happiness', 'Neutral', 'Sadness', 'Neutral', 'Sadness', 'Neutral', 'Sadness', 'Neutral', 'Sadness', 'Neutral', 'Sadness', 'Neutral'], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
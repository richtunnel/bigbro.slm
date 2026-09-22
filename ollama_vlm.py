import ollama

response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'What is written on the sign in this image?',
        'images': ['/Users/rstokes/Sites/slm.main/slm-madness/src/static/assets/img/sunset_friendship.jpeg'] 
    }]
)
print(response['message']['content'])

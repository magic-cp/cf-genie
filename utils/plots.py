import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(text: str) -> None:
    wordcloud = WordCloud(
        background_color='white',
        width=800,
        height=800,
        max_words=200,
        max_font_size=40,
        random_state=42
    ).generate(text)

    fig = plt.figure(1, figsize=(10, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

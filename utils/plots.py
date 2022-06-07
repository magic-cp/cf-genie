import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(text: str) -> None:
    wordcloud = WordCloud(
        background_color='black',
        width=1000,
        height=800,
        max_words=100,
        random_state=42,
        min_word_length=4
    ).generate(text)

    plt.figure()
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

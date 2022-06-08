from typing import Optional

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .read_write_files import write_plot


def plot_wordcloud(
        text: str,
        file_name: Optional[str] = None,
        plot_title: Optional[str] = None):
    wordcloud = WordCloud(
        background_color='black',
        width=1000,
        height=800,
        max_words=100,
        random_state=42,
        min_word_length=4
    ).generate(text)

    plt.figure()
    if plot_title:
        plt.title(plot_title)
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')

    if file_name:
        write_plot(file_name, plt)
    else:
        plt.show()
